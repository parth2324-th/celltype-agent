# ct Agent Architecture: MCP Server + Runner

## Overview

The agent pipeline has three layers:

```
User query
    │
    ▼
runner.py (AgentRunner)
    │  spawns subprocess
    ▼
claude -p  (Claude Code CLI — owns the agentic loop)
    │  MCP protocol over stdio
    ▼
mcp_stdio_server.py  (standalone process)
    │  delegates to
    ▼
mcp_server.py  (shared tool wiring logic)
    │  calls
    ▼
ct tool registry (190+ domain tools)
```

---

## `runner.py` — AgentRunner

**Role:** Orchestrates a single query by spawning `claude -p` as a subprocess, streaming its output, and rendering results to the terminal.

### Construction

```python
AgentRunner(session, trajectory=None, headless=False, trace_store=None)
```

- `session` — holds `Config` and `Console`; passed into the MCP server for tool calls
- `trajectory` — optional multi-turn memory; last 5 turns are injected into the system prompt as text
- `headless` — suppresses all terminal output (used by the benchmark runner)
- `trace_store` — if set, all events are serialized to a `.trace.jsonl` file

### `run()` / `_run_async()`

`run()` is a synchronous wrapper that calls `asyncio.run(_run_async(...))`. Each query gets a **fresh event loop** — nothing carries over between queries.

#### Setup phase

```python
ensure_loaded()
tool_names = [t.name for t in registry.list_tools()] + ["run_python", "run_r"]
```
All registered tools are listed so the system prompt can enumerate them for Claude.

```python
system_prompt = build_system_prompt(
    self.session, tool_names=tool_names, data_context=data_context, history=history
)
```
Builds a large system prompt describing ct's role, all available tools, and any trajectory history. Written to a **temp file** to avoid hitting ARG_MAX shell argument limits.

```python
mcp_config = {
    "mcpServers": {
        "ct-tools": {
            "command": sys.executable,
            "args": ["-m", "ct.agent.mcp_stdio_server"],
        }
    }
}
```
Tells Claude Code exactly one MCP server to connect to: the ct stdio server, launched as a Python subprocess using the **same venv interpreter** (`sys.executable`) so all ct packages are importable.

#### The subprocess command

```python
cmd = [
    "claude", "-p",
    "--bare",                          # no extra formatting wrappers
    "--no-session-persistence",        # fresh Claude Code session every query
    "--strict-mcp-config",             # only allow MCP servers in our config
    "--permission-mode", "bypassPermissions",  # auto-approve all MCP tool calls (no TTY available)
    "--tools", "",                     # disable all built-in Claude Code tools (Read, Edit, Bash, etc.)
    "--mcp-config", mcp_config_path,
    "--system-prompt-file", system_prompt_path,
    "--output-format", "stream-json",  # structured JSON events on stdout
    "--verbose",                       # required for stream-json to emit tool events
    user_prompt,
]
```

Key design decisions:
- `--strict-mcp-config` + `--tools ""` together mean Claude can **only** call ct MCP tools — no file system access, no bash, no internet via built-ins
- `--permission-mode bypassPermissions` is necessary because the subprocess has no TTY to prompt the user for permission approvals
- `--no-session-persistence` ensures Claude Code doesn't try to resume a prior conversation from its own session store

#### Concurrency control

```python
from ct.models.llm import _semaphore as _llm_semaphore
with _llm_semaphore:
    proc = await asyncio.create_subprocess_exec(...)
```

Acquires the module-level `threading.Semaphore` from `llm.py` (default cap: 8, overridable via `CT_MAX_LLM_SUBPROCESSES`). This is a blocking acquire — safe here because the event loop has no other concurrent coroutines (each query runs in its own `asyncio.run()`). The semaphore is released when the `with` block exits, which only happens after `await proc.wait()`.

#### Stream-JSON event loop

Claude Code emits newline-delimited JSON events on stdout. The runner processes three event types:

**`assistant` events** — Claude's responses:
- `text` blocks: rendered to terminal via `TraceRenderer`, appended to `full_text`
- `tool_use` blocks: logs the tool name + args, restarts the spinner, calls `trace_renderer.render_tool_start()`

**`user` events** — tool results returning to Claude:
- Matches the `tool_use_id` back to the pending tool call in `tool_calls_log` (reverse scan to handle retries)
- Records `result_text` and `duration_s`
- Renders success or error via `TraceRenderer`

**`result` events** — Claude's final summary:
- Used as fallback `full_text` if Claude produced no intermediate text blocks

#### Stderr drain

```python
stderr_task = asyncio.create_task(proc.stderr.read())
```

Started immediately after spawn. Draining stderr **concurrently** with stdout is critical — if the stderr pipe buffer fills up while the runner is blocked reading stdout, both pipes deadlock. The full stderr content is only inspected after `proc.wait()` to produce error messages.

#### Cleanup

Always in `finally`:
- Stops the spinner if still running
- Deletes both temp files (`mcp_config_path`, `system_prompt_path`) via `os.unlink`

---

## `mcp_stdio_server.py` — Standalone MCP Process

**Role:** A self-contained process spawned by Claude Code (via `--mcp-config`). Speaks the MCP protocol over stdin/stdout. Claude Code is the client; this server exposes all ct tools as MCP-callable functions.

### Lifecycle

1. Claude Code reads the MCP config JSON and spawns: `python -m ct.agent.mcp_stdio_server`
2. The process runs `main()` inside `asyncio.run()`
3. `stdio_server()` wraps stdin/stdout in async streams for the MCP protocol
4. The process lives for the duration of the `claude -p` session, then exits

### Session

```python
session = Session(mode="batch")
```

Creates a fresh session with default config. No API key or LLM client needed — this process only runs ct domain tools, not LLM calls.

### Tool registration

```python
@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
```

Called once by Claude Code during MCP initialization. Returns all registered ct tools (excluding experimental categories) plus `run_python` and optionally `run_r`.

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    handler = handlers.get(name)
    result = await handler(arguments)
    text = (result.get("content") or [{}])[0].get("text", "")
    is_error = result.get("is_error", False)
```

Dispatches tool calls to pre-built handlers. Returns `mcp_types.CallToolResult` with `isError=True` for failed tools so Claude knows to retry or adapt — rather than treating an error as a successful empty result.

### Handler pre-building

All handlers are built **once at startup** into a `handlers: dict` before any tool calls arrive:

```python
rp_handler, _sandbox = _make_run_python_handler(session, code_trace_buffer)
handlers["run_python"] = rp_handler

for tool_obj in registry.list_tools():
    handlers[tool_obj.name] = _make_tool_handler(tool_obj, session)
```

The sandbox is created once and reused across all `run_python` calls within a query — **variables persist between calls**. A new MCP server process is spawned per query, so state resets automatically between queries.

---

## `mcp_server.py` — Shared Tool Wiring

**Role:** Contains all the logic for wrapping ct registry tools into MCP-compatible handlers. Used by both `mcp_stdio_server.py` (stdio mode) and the old SDK-based `mcp_server.py` (in-process mode).

### `_params_to_json_schema(parameters)`

ct tools describe parameters as `{name: "description string"}`. This converts them to JSON Schema `{"type": "object", "properties": {...}}` so the MCP protocol can validate inputs. If the parameters dict is already a full JSON Schema (has `"type": "object"` and `"properties"`), it's passed through unchanged.

### `_make_tool_handler(tool_obj, session)`

Returns a closure `handler(args: dict) -> dict`. Key behaviors:

**Session injection:**
```python
call_args["_session"] = session
call_args["_prior_results"] = {}
```
Every ct tool optionally accepts `_session` for access to config and console.

**String coercion** (for legacy flat-parameter tools):
```python
try:
    call_args[key] = int(val)   # "5" → 5
except ValueError:
    call_args[key] = float(val) # "0.5" → 0.5
# "true"/"false" → bool
```
Claude Code always sends MCP arguments as strings when using legacy-style tool schemas. This coercion ensures tools that expect `int`/`float`/`bool` don't break.

**GPU routing:**
```python
if getattr(tool_obj, "requires_gpu", False) or getattr(tool_obj, "cpu_only", False):
    from ct.cloud.router import ComputeRouter
    router = ComputeRouter(config=session.config)
    result = await asyncio.to_thread(router.route, tool_obj, **call_args)
```
GPU tools go through `ComputeRouter` which decides local Docker vs CellType Cloud. If local GPU isn't available and compute mode is `"local"`, the router returns `needs_user_prompt=True` — handled by `_prompt_cloud_fallback()` which calls `input()` on the main thread.

**Thread offloading:**
```python
result = await asyncio.to_thread(tool_obj.run, **call_args)
```
All ct tools are synchronous. `asyncio.to_thread` runs them in the default thread pool so the MCP event loop stays responsive for protocol messages.

### `_make_run_python_handler(session, code_trace_buffer)`

Creates a `Sandbox` instance with config-driven settings (timeout, output dir, extra read dirs). The sandbox is returned alongside the handler so the caller can inspect it post-query (e.g., to extract generated plots/exports).

`code_trace_buffer` is a shared list: the handler appends `{tool, code, stdout, plots, error}` dicts after each execution. This is a workaround for the MCP protocol potentially truncating large tool results in the stream — the runner reads the full untruncated data directly from the buffer.

### `_format_tool_result(result, max_chars=8000)`

Converts a ct tool's return dict to a string for MCP. Always leads with `summary`, then appends other fields. Large fields are truncated at 1500 chars each, and the whole result is capped at 8000 chars to keep Claude's context window manageable.

---

## Data Flow Summary

```
ct "what proteins interact with p53?"
    │
    ├─ AgentRunner._run_async()
    │       builds system prompt (tool list + history)
    │       writes 2 temp files (system prompt, MCP config JSON)
    │       acquires _llm_semaphore (cap=8)
    │       spawns: claude -p --bare --strict-mcp-config --permission-mode bypassPermissions ...
    │
    ├─ Claude Code (claude -p subprocess)
    │       reads system prompt
    │       spawns MCP server: python -m ct.agent.mcp_stdio_server
    │       ← MCP: list_tools() → gets all 205 ct tools
    │       plans tool calls, emits stream-json on stdout
    │
    ├─ mcp_stdio_server (separate subprocess, lives with claude -p)
    │       receives MCP call_tool(name, args)
    │       handler(args) → asyncio.to_thread(tool.run, **args)
    │       returns TextContent result → Claude Code
    │
    └─ AgentRunner event loop (async for raw_line in proc.stdout)
            parses stream-json events
            renders spinner / tool traces to terminal
            collects full_text, tool_calls_log, trace_events
            on proc.wait() → builds ExecutionResult
            deletes temp files
            releases _llm_semaphore
```
