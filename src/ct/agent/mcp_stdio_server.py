"""Standalone MCP stdio server exposing all ct tools.

Spawned by AgentRunner as an MCP server subprocess:
  python3 -m ct.agent.mcp_stdio_server

Claude Code connects to it via --mcp-config and handles the full
agentic loop natively, preserving multi-turn context correctly.
"""

import asyncio
import logging

logger = logging.getLogger("ct.mcp_stdio")


async def main():
    from mcp import types as mcp_types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    from ct.agent.mcp_server import (
        _make_run_python_handler,
        _make_run_r_handler,
        _make_tool_handler,
        _params_to_json_schema,
    )
    from ct.agent.session import Session
    from ct.tools import EXPERIMENTAL_CATEGORIES, ensure_loaded, registry

    session = Session(mode="batch")
    ensure_loaded()

    code_trace_buffer: list[dict] = []
    handlers: dict = {}

    app = Server("ct-tools")

    @app.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        tools = []
        for tool_obj in registry.list_tools():
            if tool_obj.category in EXPERIMENTAL_CATEGORIES:
                continue
            schema = _params_to_json_schema(tool_obj.parameters)
            tools.append(
                mcp_types.Tool(
                    name=tool_obj.name,
                    description=tool_obj.description or "",
                    inputSchema=schema,
                )
            )

        tools.append(
            mcp_types.Tool(
                name="run_python",
                description=(
                    "Execute Python code in a sandboxed environment. Variables persist "
                    "between calls. Pre-imported: pd, np, plt, sns, scipy_stats, sklearn, "
                    "json, re, math, os, Path. Save plots to OUTPUT_DIR. "
                    "Assign result = {'summary': '...', 'answer': '...'} when done."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                    "required": ["code"],
                },
            )
        )

        try:
            import rpy2.robjects  # noqa: F401
            tools.append(
                mcp_types.Tool(
                    name="run_r",
                    description=(
                        "Execute R code via rpy2. Use for: natural splines (ns()), "
                        "wilcox.test(), p.adjust(), survival analysis, KEGG pathway analysis."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "R code to execute"}},
                        "required": ["code"],
                    },
                )
            )
        except ImportError:
            pass

        return tools

    # Build handlers
    rp_handler, _sandbox = _make_run_python_handler(session, code_trace_buffer)
    handlers["run_python"] = rp_handler

    try:
        import rpy2.robjects  # noqa: F401
        handlers["run_r"] = _make_run_r_handler(code_trace_buffer)
    except ImportError:
        pass

    for tool_obj in registry.list_tools():
        if tool_obj.category in EXPERIMENTAL_CATEGORIES:
            continue
        handlers[tool_obj.name] = _make_tool_handler(tool_obj, session)

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        handler = handlers.get(name)
        if handler is None:
            return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            result = await handler(arguments)
            text = (result.get("content") or [{}])[0].get("text", "")
            is_error = result.get("is_error", False)
        except Exception as exc:
            text = f"Error: {exc}"
            is_error = True

        if is_error:
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=text)],
                isError=True,
            )
        return [mcp_types.TextContent(type="text", text=text)]

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
