"""Aurelius CLI — Interactive Dragon Terminal (Main Entry Point).

Lightning blue (#00BFFF) themed. Dragon mascot. Real-time streaming.
Inspired by Claude Code, OPENDEV, Terminal-Bench, NL2SH, YAMLE.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style as PromptStyle

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    PromptSession = None
    AutoSuggestFromHistory = None
    FileHistory = None
    KeyBindings = None
    PromptStyle = None
    _HAS_PROMPT_TOOLKIT = False
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .agent_engine import AgentEngine
from .dragon_mascot import AURELIUS_BANNER, AURELIUS_MASCOT, DRAGON_ART, WELCOME_HEADER

LIGHTNING_BLUE = "#00BFFF"
DRAGON_GREEN = "#00FF88"
TERMINAL_BG = "#0a0a1a"
LIGHTNING_BLUE_DIM = "#006688"
ACCENT_PURPLE = "#8855FF"

console = Console()

# ─── Styling ───────────────────────────────────────────────────────────────

STYLE = (
    PromptStyle([
        ("prompt", f"bold {LIGHTNING_BLUE}"),
        ("dragon", f"bold {DRAGON_GREEN}"),
        ("separator", f"dim {LIGHTNING_BLUE_DIM}"),
    ])
    if _HAS_PROMPT_TOOLKIT
    else None
)

PROMPT_SYMBOL = [
    "⚡ ",
    "▶ ",
    "λ ",
    "$ ",
    "➤ ",
]

bindings = KeyBindings() if _HAS_PROMPT_TOOLKIT else None

if bindings is not None:
    @bindings.add("c-c")
    def _(event):
        event.app.exit()


    @bindings.add("tab")
    def _(event):
        event.current_buffer.insert_text("    ")


class AureliusCLI:
    """Main interactive CLI terminal with dragon theme."""

    def __init__(self):
        self.engine = AgentEngine()
        self.history_file = Path.home() / ".aurelius_history"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.session = None
        if _HAS_PROMPT_TOOLKIT:
            self.session = PromptSession(
                history=FileHistory(str(self.history_file)),
                auto_suggest=AutoSuggestFromHistory(),
                key_bindings=bindings,
                style=STYLE,
                enable_history_search=True,
                mouse_support=True,
            )
        self._running = True
        self._prompt_index = 0
        self._message_count = 0

    def _print_banner(self) -> None:
        console.clear()
        header = Text(WELCOME_HEADER, style=LIGHTNING_BLUE)
        console.print(header)
        console.print(Panel(DRAGON_ART, border_style=LIGHTNING_BLUE, title="[bold]AURELIUS — The Coding Dragon[/bold]"))

        info = Table.grid(padding=1)
        info.add_column(style=f"bold {DRAGON_GREEN}")
        info.add_column(style=f"dim {LIGHTNING_DIM}")
        info.add_row("⚡ Mode:", "AI Coding Terminal")
        info.add_row("⚡ Engine:", "Dual-Agent (Planner + Executor)")
        info.add_row("⚡ Safety:", "Deny-first Tool Authorization")
        info.add_row("⚡ Context:", "Adaptive Compaction + Session Memory")

        console.print(Panel(info, border_style=LIGHTNING_BLUE, title="[bold]System Status[/bold]"))
        console.print(f"\n[{LIGHTNING_BLUE}]Type /help for commands, or just ask me anything![/]\n")
        console.print(Rule(style=LIGHTNING_DIM))

    def _get_prompt(self) -> str:
        path = self.engine.session.workspace or os.getcwd()
        short_path = os.path.basename(path)
        symbol = PROMPT_SYMBOL[self._prompt_index % len(PROMPT_SYMBOL)]
        return f"\n[{LIGHTNING_BLUE}]╭─[/] [{DRAGON_GREEN}]{short_path}[/] [{LIGHTNING_BLUE_DIM}]{self._message_count + 1}[/]\n[{LIGHTNING_BLUE}]╰─{symbol}[/] "

    def _stream_response(self, text: str) -> None:
        console.print()

    def _render_response(self, text: str) -> None:
        if not text:
            return

        lines = text.split("\n")
        code_block = False
        fence_lines: list[str] = []

        for line in lines:
            if line.startswith("```"):
                if code_block:
                    fence_lines.append(line)
                    code = "\n".join(fence_lines[1:])
                    try:
                        lang = fence_lines[0].replace("```", "").strip() or "python"
                        syntax = Syntax(code, lang, theme="monokai", line_numbers=True)
                        console.print(Panel(syntax, border_style=LIGHTNING_BLUE, title=f"[bold]{lang}[/bold]"))
                    except Exception:
                        console.print(code)
                    fence_lines = []
                else:
                    fence_lines = [line]
                code_block = not code_block
            elif code_block:
                fence_lines.append(line)
            elif line.startswith("**") and line.endswith("**"):
                console.print(Text(line.strip("*"), style=f"bold {DRAGON_GREEN}"))
            elif line.startswith("$ "):
                console.print(Text(line, style=f"italic {LIGHTNING_BLUE_DIM}"))
            elif line.strip().startswith("⚠️"):
                console.print(Text(line, style="bold yellow"))
            elif line.strip().startswith("✅") or line.strip().startswith("✔"):
                console.print(Text(line, style=f"bold {DRAGON_GREEN}"))
            elif line.strip():
                console.print(Markdown(line))

    def process_input(self, user_input: str) -> str | None:
        self._message_count += 1

        if user_input.startswith("/"):
            result = self.engine._handle_command(user_input)
            if result == "exit":
                return None
            return result

        with console.status(f"[{LIGHTNING_BLUE}] Processing...", spinner="dots"):
            result = self.engine.process_message(user_input)
        return result

    def run(self) -> None:
        self._print_banner()

        while self._running:
            try:
                prompt = self._get_prompt()
                if self.session is not None:
                    user_input = self.session.prompt(prompt, style=STYLE)
                else:
                    user_input = input(prompt)
            except (KeyboardInterrupt, EOFError):
                console.print(f"\n[{DRAGON_GREEN}]Farewell, mortal. The dragon rests.[/]")
                break

            if not user_input or not user_input.strip():
                continue

            user_input = user_input.strip()

            if user_input in ("exit", "quit", "q"):
                console.print(f"\n[{DRAGON_GREEN}]🔥 DRACARYS! The dragon departs.[/]")
                break

            try:
                result = self.process_input(user_input)
                if result is None:
                    break
                if result:
                    console.print()
                    self._render_response(result)
                    console.print()

                    if self._message_count % 5 == 0:
                        s = self.engine.session
                        mem = f"⚡ Mem: {len(s.messages)} msgs | {len(s.memory)} keys | {len(self.engine.tool.history)} cmds"
                        console.print(Text("─" * 50, style=f"dim {LIGHTNING_BLUE_DIM}"))
                        console.print(Text(mem, style=f"dim {LIGHTNING_BLUE_DIM}"))

            except Exception as e:
                console.print(f"[bold red]Error:[/] {e}")

        self._cleanup()

    def _cleanup(self) -> None:
        pass


def main():
    cli = AureliusCLI()
    cli.run()


if __name__ == "__main__":
    main()
