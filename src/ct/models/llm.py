"""
LLM client backed by claude -p subprocess.

Every call spawns a fresh `claude -p` process with a temp system-prompt file
and `--output-format json` (chat) or `--output-format stream-json` (stream).

A module-level semaphore caps concurrent subprocesses so nested tool calls
(e.g. claude.reason firing while a thread pool is running) don't fan out
unboundedly. Default cap is 8; override with CT_MAX_LLM_SUBPROCESSES env var.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Generator

logger = logging.getLogger("ct.llm")

# ── Concurrency cap ──────────────────────────────────────────────────────────

_MAX_SUBPROCESSES = int(os.environ.get("CT_MAX_LLM_SUBPROCESSES", "8"))
_semaphore = threading.Semaphore(_MAX_SUBPROCESSES)


# ── Shared env builder ───────────────────────────────────────────────────────

def build_claude_env() -> dict:
    """Build a subprocess env suitable for any claude -p call.

    Resolution order for ANTHROPIC_API_KEY:
      1. Already in os.environ (real key)
      2. Claude Code OAuth token from macOS keychain

    """
    env = os.environ.copy()

    if not env.get("ANTHROPIC_API_KEY") and sys.platform == "darwin":
        try:
            raw = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()
            env["ANTHROPIC_API_KEY"] = json.loads(raw)["claudeAiOauth"]["accessToken"]
        except Exception:
            pass

    return env


# ── Response / usage types ───────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    usage: dict = None
    raw: object = None
    content_blocks: list = None


@dataclass
class UsageTracker:
    calls: list = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.get("input", 0) for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.get("output", 0) for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(c.get("cost", 0.0) for c in self.calls)

    def record(self, model: str, usage: dict):
        if usage:
            self.calls.append({"model": model, **usage})

    def summary(self) -> str:
        if not self.calls:
            return "No LLM calls made."
        return (
            f"{len(self.calls)} LLM calls | "
            f"{self.total_input_tokens:,} in + {self.total_output_tokens:,} out tokens"
        )

    def reset(self):
        self.calls.clear()


# ── Client ───────────────────────────────────────────────────────────────────

class LLMClient:
    """LLM client using claude -p subprocess.

    Presents the same chat/stream interface as the old LiteLLM-based client.
    All calls go through the claude CLI so auth is handled by Claude Code's
    own credential store — no API key configuration needed.

    Concurrent subprocesses are capped by a module-level semaphore
    (CT_MAX_LLM_SUBPROCESSES, default 8) so nested tool calls don't fan out.
    """

    def __init__(self, provider: str = "anthropic", model: str = None,
                 api_key: str = None, endpoint: str = None):
        self.provider = provider
        self.usage = UsageTracker()

    def chat(self, system: str, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, tools: list[dict] | None = None) -> LLMResponse:
        """Single-turn chat via claude -p --output-format json."""
        user_msg = self._flatten_messages(messages)
        text = self._run_subprocess(system, user_msg)
        return LLMResponse(content=text)

    def stream(self, system: str, messages: list[dict], temperature: float = 0.1,
               max_tokens: int = 4096) -> Generator[str, None, None]:
        """Stream response via claude -p --output-format stream-json."""
        user_msg = self._flatten_messages(messages)
        yield from self._run_subprocess_stream(system, user_msg)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _flatten_messages(self, messages: list[dict]) -> str:
        if not messages:
            return ""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                parts.append(content)
            else:
                parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    def _build_cmd(self, system_prompt_path: str, user_msg: str, output_format: str) -> list[str]:
        cmd = [
            "claude", "-p",
            "--output-format", output_format,
            "--no-session-persistence",
            "--system-prompt-file", system_prompt_path,
        ]
        if output_format == "stream-json":
            cmd.append("--verbose")
        cmd.append(user_msg)
        return cmd

    def _run_subprocess(self, system: str, user_msg: str) -> str:
        system_prompt_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(system)
                system_prompt_path = f.name

            cmd = self._build_cmd(system_prompt_path, user_msg, "json")
            with _semaphore:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=build_claude_env(),
                    timeout=120,
                )
        finally:
            if system_prompt_path:
                try:
                    os.unlink(system_prompt_path)
                except OSError:
                    pass

        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exited {proc.returncode}: {proc.stderr[:400]}")

        try:
            return json.loads(proc.stdout).get("result", "").strip()
        except (json.JSONDecodeError, AttributeError):
            return proc.stdout.strip()

    def _run_subprocess_stream(self, system: str, user_msg: str) -> Generator[str, None, None]:
        system_prompt_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(system)
                system_prompt_path = f.name

            cmd = self._build_cmd(system_prompt_path, user_msg, "stream-json")
            with _semaphore:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=build_claude_env(),
                )
                try:
                    for raw_line in proc.stdout:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        yield text
                finally:
                    proc.stdout.close()
                    proc.wait()
        finally:
            if system_prompt_path:
                try:
                    os.unlink(system_prompt_path)
                except OSError:
                    pass
