"""
AgentRunner: runs queries by spawning ``claude -p --mcp-config`` as a subprocess.

Claude Code owns the full agentic loop — multi-turn conversations, tool
calls, and error recovery — while the ct MCP server (mcp_stdio_server.py)
provides all domain tools.  The runner just streams and renders the output.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import traceback

from ct.agent.types import ExecutionResult, Plan, Step

logger = logging.getLogger("ct.runner")


class AgentRunner:
    """Run queries by delegating the full agentic loop to Claude Code CLI.

    ``claude -p --mcp-config <mcp.json> --system-prompt <prompt> <query>``
    starts a clean session for each query.  The ct MCP stdio server
    (python -m ct.agent.mcp_stdio_server) exposes all domain tools.
    """

    def __init__(
        self,
        session,
        trajectory=None,
        headless: bool = False,
        trace_store=None,
    ):
        self.session = session
        self.trajectory = trajectory
        self._headless = headless
        self.trace_store = trace_store
        self._active_spinner = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        context: dict | None = None,
        progress_callback=None,
    ) -> ExecutionResult:
        """Execute a query synchronously (blocking wrapper around async)."""
        return asyncio.run(self._run_async(query, context, progress_callback))

    async def _run_async(
        self,
        query: str,
        context: dict | None = None,
        progress_callback=None,
    ) -> ExecutionResult:
        from ct.agent.system_prompt import build_system_prompt
        from ct.tools import ensure_loaded, registry
        from ct.ui.traces import TraceRenderer

        t0 = time.time()
        config = self.session.config
        ctx = context or {}

        # Spinner
        thinking_status = None
        if not self._headless:
            from ct.ui.status import ThinkingStatus
            thinking_status = ThinkingStatus(self.session.console, phase="planning")
            thinking_status.__enter__()
            thinking_status.start_async_refresh()
            self._active_spinner = thinking_status

        mcp_config_path = None
        system_prompt_path = None

        try:
            # Build tool list and system prompt
            ensure_loaded()
            tool_names = [t.name for t in registry.list_tools()] + ["run_python", "run_r"]

            history = None
            if self.trajectory and self.trajectory.turns:
                history = self.trajectory.context_for_planner()

            data_context = None
            data_dir = ctx.get("data_dir")
            if data_dir:
                data_context = f"Data directory: {data_dir}\n"
                config.set("sandbox.extra_read_dirs", str(data_dir))

            system_prompt = build_system_prompt(
                self.session, tool_names=tool_names, data_context=data_context, history=history
            )

            # Build user prompt with optional context
            user_prompt = query
            context_parts = []
            if ctx.get("compound_smiles"):
                context_parts.append(f"Compound SMILES: {ctx['compound_smiles']}")
            if ctx.get("target"):
                context_parts.append(f"Target: {ctx['target']}")
            if ctx.get("indication"):
                context_parts.append(f"Indication: {ctx['indication']}")
            if ctx.get("mention_context"):
                context_parts.append(ctx["mention_context"])
            if context_parts:
                user_prompt = query + "\n\nContext:\n" + "\n".join(context_parts)

            # Write system prompt to temp file (avoids ARG_MAX limits)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as sp_f:
                sp_f.write(system_prompt)
                system_prompt_path = sp_f.name

            # Write MCP config pointing to this Python executable
            mcp_config = {
                "mcpServers": {
                    "ct-tools": {
                        "command": sys.executable,
                        "args": ["-m", "ct.agent.mcp_stdio_server"],
                    }
                }
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as mc_f:
                json.dump(mcp_config, mc_f)
                mcp_config_path = mc_f.name

            cmd = [
                "claude", "-p",
                "--bare",
                "--no-session-persistence",
                "--strict-mcp-config",
                "--tools", "",
                "--mcp-config", mcp_config_path,
                "--system-prompt-file", system_prompt_path,
                "--output-format", "stream-json",
                "--verbose",
                user_prompt,
            ]

            env = self._build_env(config)

            from ct.models.llm import _semaphore as _llm_semaphore
            with _llm_semaphore:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )

                # Drain stderr concurrently to prevent pipe buffer deadlock
                stderr_task = asyncio.create_task(proc.stderr.read())

                trace_renderer = TraceRenderer(self.session.console)
                trace_events: list[dict] | None = [] if self.trace_store is not None else None
                full_text: list[str] = []
                tool_calls_log: list[dict] = []
                n_turns = 0

                async for raw_line in proc.stdout:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "assistant":
                        print(f"[mcp] <- assistant response (turn {n_turns + 1})", flush=True)
                        n_turns += 1
                        message = event.get("message", {})
                        for block in message.get("content", []):
                            btype = block.get("type")

                            if btype == "text":
                                text = block.get("text", "")
                                if text.strip():
                                    full_text.append(text)
                                    if thinking_status is not None:
                                        thinking_status.stop()
                                        thinking_status = None
                                        self._active_spinner = None
                                    if not self._headless:
                                        trace_renderer.render_reasoning(text)
                                    if progress_callback:
                                        progress_callback(text.strip().replace("\n", " ")[:40])
                                    if trace_events is not None:
                                        trace_events.append({
                                            "type": "text",
                                            "content": text,
                                            "timestamp": time.time(),
                                        })

                            elif btype == "tool_use":
                                tool_name = block.get("name", "").replace("mcp__ct-tools__", "")
                                tool_input = block.get("input", {})
                                tool_id = block.get("id", "")
                                now = time.time()
                                print(f"[mcp] -> {tool_name}  args={tool_input}", flush=True)
                                tool_calls_log.append({
                                    "name": tool_name,
                                    "input": tool_input,
                                    "tool_use_id": tool_id,
                                    "start_time": now,
                                })
                                # Restart spinner while tool runs
                                if thinking_status is None and not self._headless:
                                    try:
                                        from ct.ui.status import ThinkingStatus
                                        thinking_status = ThinkingStatus(
                                            self.session.console, phase="evaluating"
                                        )
                                        thinking_status.__enter__()
                                        thinking_status.start_async_refresh()
                                        self._active_spinner = thinking_status
                                    except ImportError:
                                        pass
                                if not self._headless:
                                    trace_renderer.render_tool_start(tool_name, tool_input)
                                if progress_callback:
                                    progress_callback(f"\u25b8 {tool_name}")
                                if trace_events is not None:
                                    trace_events.append({
                                        "type": "tool_start",
                                        "tool": tool_name,
                                        "input": tool_input,
                                        "tool_use_id": tool_id,
                                        "timestamp": now,
                                    })

                    elif etype == "user":
                        # Tool results arrive as user messages
                        print(f"[mcp] <- tool result", flush=True)
                        message = event.get("message", {})
                        for block in message.get("content", []):
                            if block.get("type") != "tool_result":
                                continue
                            tool_id = block.get("tool_use_id", "")
                            result_content = block.get("content", "")
                            if isinstance(result_content, list):
                                result_content = " ".join(
                                    c.get("text", "") for c in result_content if isinstance(c, dict)
                                )
                            is_error = block.get("is_error", False)

                            for tc in reversed(tool_calls_log):
                                if tc.get("tool_use_id") == tool_id and "result_text" not in tc:
                                    duration = time.time() - tc["start_time"]
                                    tc["result_text"] = result_content
                                    tc["duration_s"] = duration
                                    if not self._headless:
                                        if is_error:
                                            trace_renderer.render_tool_error(tc["name"], result_content)
                                        else:
                                            trace_renderer.render_tool_complete(
                                                tc["name"], tc["input"], result_content, duration
                                            )
                                    if trace_events is not None:
                                        trace_events.append({
                                            "type": "tool_result",
                                            "tool": tc["name"],
                                            "tool_use_id": tool_id,
                                            "result_text": result_content,
                                            "is_error": is_error,
                                            "duration_s": duration,
                                            "timestamp": time.time(),
                                        })
                                    break

                    elif etype == "result":
                        if thinking_status is not None:
                            thinking_status.stop()
                            thinking_status = None
                            self._active_spinner = None
                        # Use the result field as a fallback summary if Claude produced no text
                        result_text = event.get("result", "")
                        if result_text and not full_text:
                            full_text.append(result_text)

                await proc.wait()
                stderr_bytes = await stderr_task
                err_msg = stderr_bytes.decode("utf-8", errors="replace")

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"claude -p exited {proc.returncode}: {err_msg[:600]}"
                    )

        except Exception as exc:
            logger.error("AgentRunner failed: %s\n%s", exc, traceback.format_exc())
            return self._make_error_result(query, str(exc), time.time() - t0)
        finally:
            if thinking_status is not None:
                thinking_status.stop()
                self._active_spinner = None
            for path in (mcp_config_path, system_prompt_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        duration = time.time() - t0
        summary = "\n".join(full_text).strip() or "(Agent produced no text output)"

        steps = []
        for i, tc in enumerate(tool_calls_log, 1):
            step = Step(
                id=i, tool=tc["name"],
                description=f"Called {tc['name']}",
                tool_args=tc.get("input", {}),
            )
            step.status = "completed"
            steps.append(step)

        tool_calls_for_result = [
            {
                "name": tc["name"],
                "input": tc.get("input", {}),
                "result_text": tc.get("result_text", ""),
                "duration_s": tc.get("duration_s", 0.0),
            }
            for tc in tool_calls_log
        ]

        exec_result = ExecutionResult(
            plan=Plan(query=query, steps=steps),
            summary=summary,
            raw_results={"tool_calls": tool_calls_for_result},
            duration_s=duration,
            iterations=1,
            metadata={
                "sdk_cost_usd": 0.0,
                "sdk_turns": n_turns,
                "tool_call_count": len(tool_calls_log),
            },
        )

        if self.trace_store is not None and trace_events:
            try:
                model_name = config.get("llm.model", "__local_server__")
                self.trace_store.add_events(
                    trace_events, query=query, model=model_name,
                    duration_s=duration, cost_usd=0.0,
                )
                self.trace_store.flush()
            except Exception as exc:
                logger.warning("Failed to flush trace: %s", exc)

        if not self._headless:
            self.session.console.print(
                f"\n  [dim]{duration:.1f}s | {n_turns} turns | {len(tool_calls_log)} tool calls[/dim]"
            )

        return exec_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_env(*_) -> dict:
        from ct.models.llm import build_claude_env
        return build_claude_env()

    @staticmethod
    def _make_error_result(query: str, error: str, duration: float) -> ExecutionResult:
        return ExecutionResult(
            plan=Plan(query=query, steps=[]),
            summary=f"Agent error: {error}",
            raw_results={"error": error},
            duration_s=duration,
            iterations=1,
        )
