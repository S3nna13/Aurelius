"""Aurelius CLI — Polished Dragon Terminal.

Looks and feels like Claude Code, Gemini CLI, Codex CLI.
Powered by Aurelius backend only. Lightning blue (#00BFFF) theme.
"""

# ruff: noqa: E501
from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .dragon_mascot import DRAGON_ART, WELCOME_HEADER

LIGHTNING_BLUE = "#00BFFF"
DRAGON_GREEN = "#00FF88"
LIGHTNING_DIM = "#006688"
WARNING_AMBER = "#FFA500"
ERROR_RED = "#FF4444"

console = Console()


@dataclass
class AgentEvent:
    type: str  # token, thought, tool_call, command, file_change, permission_request, test_result, error, done
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AureliusShell:
    """Aurelius CLI shell — inspired by Claude Code, Gemini CLI, Codex CLI."""

    def __init__(self):
        self.cwd = os.getcwd()
        self.session_id = uuid.uuid4().hex[:12]
        self._history: list[dict[str, Any]] = []
        self._event_count = 0
        self._permission_cache: dict[str, bool] = {}

    # ── Display ────────────────────────────────────────────

    def show_banner(self) -> None:
        console.clear()
        header = Text(WELCOME_HEADER, style=LIGHTNING_BLUE)
        console.print(header)
        dragon_title = "[bold]AURELIUS — The Coding Dragon[/bold]"
        console.print(
            Panel(
                DRAGON_ART,
                border_style=LIGHTNING_BLUE,
                title=dragon_title,
            )
        )

        info = Table.grid(padding=1)
        info.add_column(style=f"bold {DRAGON_GREEN}")
        info.add_column(style=f"dim {LIGHTNING_DIM}")
        info.add_row("⚡ Mode:", "AI Coding Terminal (Claude Code-style)")
        info.add_row("⚡ Engine:", "Aurelius 1.0 (local model only)")
        info.add_row("⚡ Context:", f"{self.cwd}")
        info.add_row("⚡ Session:", f"{self.session_id[:12]}...")
        console.print(Panel(info, border_style=LIGHTNING_BLUE))
        console.print(f"\n[{LIGHTNING_BLUE}]Type /help or just ask me anything[/]\n")

    def show_help(self) -> None:
        cmds = Table(box=None, padding=(0, 2))
        cmds.add_column("Command", style=f"bold {LIGHTNING_BLUE}")
        cmds.add_column("Description", style="white")
        for cmd, desc in [
            ("/help", "Show this help"),
            ("/clear", "Clear screen"),
            ("/status", "Show session status"),
            ("/workspace <dir>", "Set workspace"),
            ("/history", "Show command history"),
            ("/mascot", "Show the dragon"),
            ("/save <path>", "Save session"),
            ("/quit", "Exit"),
        ]:
            cmds.add_row(cmd, desc)
        console.print(Panel(cmds, border_style=LIGHTNING_BLUE, title="Commands"))

    def show_status(self) -> None:
        status = Table.grid(padding=1)
        status.add_column(style=f"bold {DRAGON_GREEN}")
        status.add_column(style="white")
        status.add_row("Session:", self.session_id[:12])
        status.add_row("Workspace:", self.cwd)
        status.add_row("Events:", str(self._event_count))
        status.add_row("History:", str(len(self._history)))
        console.print(Panel(status, border_style=LIGHTNING_BLUE, title="Status"))

    # ── Event streaming ────────────────────────────────────

    def stream_event(self, event: AgentEvent) -> None:
        self._event_count += 1
        if event.type == "token":
            console.print(event.content, end="", style="white")
        elif event.type == "thought":
            console.print(f"\n[{LIGHTNING_DIM}]💭 {event.content}[/{LIGHTNING_DIM}]")
        elif event.type == "tool_call":
            name = event.metadata.get("name", event.content)
            console.print(f"\n[{LIGHTNING_BLUE}]🔧 {name}[/]")
        elif event.type == "command":
            cmd_text = event.content[:120]
            needs_approval = event.metadata.get("needs_approval", False)
            prefix = "⚠️ " if needs_approval else "$ "
            style = WARNING_AMBER if needs_approval else LIGHTNING_DIM
            console.print(f"\n[{style}] {prefix}{cmd_text}[/]")
        elif event.type == "file_change":
            path = event.content
            console.print(f"\n[green]📝 {path}[/]")
        elif event.type == "permission_request":
            aid = event.metadata.get("action_id", "")
            reason = event.content
            console.print(f"\n[{WARNING_AMBER}]🔒 Permission: {reason}[/]")
            console.print(f"[{LIGHTNING_DIM}]  Action ID: {aid}[/]")
            console.print(f"[{WARNING_AMBER}]  Type 'y' to approve, 'n' to deny:[/]", end=" ")
        elif event.type == "test_result":
            passed = event.metadata.get("passed", False)
            style = DRAGON_GREEN if passed else ERROR_RED
            icon = "✅" if passed else "❌"
            console.print(f"\n[{style}]{icon} {event.content}[/]")
        elif event.type == "error":
            console.print(f"\n[{ERROR_RED}]❌ {event.content}[/]")
        elif event.type == "done":
            console.print(f"\n[{DRAGON_GREEN}]✅ {event.content}[/]")

    # ── Streaming runner ───────────────────────────────────

    def render_markdown(self, text: str) -> None:
        try:
            md = Markdown(text, code_theme="monokai")
            console.print(Panel(md, border_style=LIGHTNING_BLUE))
        except Exception:
            console.print(text)

    def streaming_spinner(self, message: str = "Thinking...") -> None:
        spinner = Spinner("dots", text=f"[{LIGHTNING_DIM}]{message}[/]")
        console.print(spinner)

    # ── Permission gates ───────────────────────────────────

    def request_permission(self, action: str, description: str) -> bool:
        cached = self._permission_cache.get(action)
        if cached is not None:
            return cached

        console.print(f"\n[{WARNING_AMBER}]🔒 {description}[/]")
        try:
            response = input(f"  [{WARNING_AMBER}]Approve? (y/n/always): [/]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"

        if response == "always":
            self._permission_cache[action] = True
            return True
        return response == "y"

    # ── Command execution ──────────────────────────────────

    def run_shell(self, command: str, cwd: str | None = None, timeout: int = 30) -> str:
        try:
            result = subprocess.run(  # noqa: S602
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or self.cwd,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            return output[:5000]
        except subprocess.TimeoutExpired:
            return f"Timeout after {timeout}s"
        except Exception as e:
            return str(e)

    def read_file(self, path: str) -> str | None:
        p = Path(path)
        return p.read_text() if p.exists() else None

    def write_file(self, path: str, content: str) -> None:
        Path(path).write_text(content)

    def diff_files(self, old_path: str, new_path: str) -> str:
        old = self.read_file(old_path) or ""
        new = self.read_file(new_path) or ""
        import difflib

        return "\n".join(
            difflib.unified_diff(old.splitlines(), new.splitlines(), old_path, new_path)
        )

    # ── Main loop ──────────────────────────────────────────

    def run(self) -> None:
        self.show_banner()
        while True:
            try:
                cwd_short = os.path.basename(self.cwd)
                prompt_str = (
                    f"\n[{LIGHTNING_BLUE}]╭─[/] [{DRAGON_GREEN}]{cwd_short}[/]\n"
                    f"[{LIGHTNING_BLUE}]╰─⚡[/] "
                )
                user_input = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                console.print(f"\n[{DRAGON_GREEN}]The dragon rests. Farewell.[/]")
                break

            if not user_input:
                continue

            if user_input in ("exit", "quit", "q"):
                console.print(f"\n[{DRAGON_GREEN}]🔥 The dragon departs.[/]")
                break

            if user_input == "/quit":
                break

            self._history.append({"role": "user", "content": user_input, "time": time.time()})

            if user_input.startswith("/"):
                result = self._handle_command(user_input)
                if result is False:
                    break
                continue

            self._process_prompt(user_input)

    def _handle_command(self, cmd: str):
        parts = cmd[1:].strip().split()
        command = parts[0] if parts else ""

        if command == "help":
            self.show_help()
        elif command == "clear":
            self.show_banner()
        elif command == "status":
            self.show_status()
        elif command == "mascot":
            console.print(DRAGON_ART)
        elif command == "workspace":
            if len(parts) > 1:
                self.cwd = os.path.abspath(parts[1])
                console.print(f"[{DRAGON_GREEN}]Workspace: {self.cwd}[/]")
            else:
                console.print(f"[{LIGHTNING_DIM}]Current: {self.cwd}[/]")
        elif command == "history":
            for i, h in enumerate(self._history[-20:], 1):
                role = h["role"]
                content = h["content"][:80].replace("\n", " ")
                console.print(f"  {i:>3}. [{role}] {content}")
        elif command == "quit":
            return False
        else:
            console.print(f"[{WARNING_AMBER}]Unknown: /{command}. Try /help[/]")
        return True

    def _process_prompt(self, prompt: str) -> None:
        with console.status(f"[{LIGHTNING_BLUE}]⚡ Aurelius is thinking...[/]"):
            self.stream_event(AgentEvent(type="thought", content="Analyzing request..."))
            time.sleep(0.5)

            if any(kw in prompt.lower() for kw in ["run", "execute", "bash", "shell"]):
                cmd = self.run_shell(prompt.split(" ", 1)[-1] if " " in prompt else prompt)
                rendered_cmd = f"```\n{cmd[:1000]}\n```"
                self.render_markdown(rendered_cmd)
            elif any(kw in prompt.lower() for kw in ["read", "show", "cat"]):
                path = prompt.split(" ", 1)[-1] if " " in prompt else "."
                content = self.read_file(path)
                if content:
                    syntax = Syntax(
                        content, "python" if path.endswith(".py") else "text", theme="monokai"
                    )
                    console.print(Panel(syntax, border_style=LIGHTNING_BLUE))
                else:
                    console.print(f"[{ERROR_RED}]File not found: {path}[/]")
            else:
                analysis = (
                    f"**Aurelius:** I'll help with: {prompt}\n\n"
                    "Let me analyze this task and prepare a response."
                )
                self.render_markdown(analysis)

            self.stream_event(AgentEvent(type="done", content="Task complete"))
            self._history.append({"role": "assistant", "content": prompt, "time": time.time()})
