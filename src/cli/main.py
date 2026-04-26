"""
src/cli/main.py

Aurelius CLI — interactive AI assistant and model toolkit.

Usage:
    aurelius                    # launch interactive chat
    aurelius chat               # interactive chat (explicit)
    aurelius chat -s coding     # use built-in persona
    aurelius serve              # start API server + web UI
    aurelius serve --port 8080  # custom port
    aurelius train              # launch training
    aurelius eval <ckpt>        # run evaluation
    aurelius --version          # print version
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from typing import Optional

__version__ = "0.1.0"

# ── rich imports (graceful fallback if not installed) ─────────────────────────

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.theme import Theme
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.table import Table
    import rich.box as box

    _RICH = True
except ImportError:
    _RICH = False

# ── console setup ─────────────────────────────────────────────────────────────

_THEME = None
if _RICH:
    _THEME = Theme(
        {
            "aurelius.dragon": "bold bright_red",
            "aurelius.user": "bold bright_white",
            "aurelius.assistant": "bold bright_green",
            "aurelius.system": "dim cyan",
            "aurelius.cmd": "bold bright_cyan",
            "aurelius.dim": "dim white",
            "aurelius.warn": "bold yellow",
            "aurelius.error": "bold red",
            "aurelius.ok": "bold green",
            "aurelius.border": "bright_red",
            "aurelius.header": "bold bright_red",
        }
    )

_console: "Console | None" = Console(theme=_THEME, highlight=False) if _RICH else None


def _print(msg: str = "", **kwargs) -> None:
    if _RICH and _console:
        _console.print(msg, **kwargs)
    else:
        print(msg)


def _print_rich(renderable) -> None:
    if _RICH and _console:
        _console.print(renderable)
    else:
        print(str(renderable))


# ── ANSI fallback (used only when rich is absent) ────────────────────────────

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if _RICH or not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


BOLD = lambda t: _c("1", t)
DIM = lambda t: _c("2", t)
GREEN = lambda t: _c("32", t)
CYAN = lambda t: _c("36", t)
YELLOW = lambda t: _c("33", t)
BLUE = lambda t: _c("34", t)
RED = lambda t: _c("31", t)
MAGENTA = lambda t: _c("35", t)


# ── Dragon ASCII art mascot ───────────────────────────────────────────────────

DRAGON = r"""
                                                    /===-_---~~~~~~~~~------____
                                                   |===-~___                _,-'
                    -==\\                         `//~\\   ~~~~`---.___.-~~
                ______-==|                         | |  \\           _-~`
          __--~~~  ,-/-==\\                        | |   `\        ,'
       _-~       /'    |  \\                      / /      \      /
     .'         /       |   \\                   / /        \   /'
    /  ____  / |          |   `\                / /          \,/
   \_'/  `~\/  /',  .     |    `\            __/ /            /'
    |   |    |    /'      |     `\       ___/ /             /'
    |   |    |  /'       |       '     /  ___/            /'
     \       /'          |            /          /'
      \     /            |           /          /'
       \  /'             |          /         /'
        \/               |         /        /'
                         |        /       /'
                         |       /      /'
                         |      /     /'
"""

DRAGON_SMALL = r"""
      /\_____/\
     /  o   o  \
    ( ==  ^  == )
     )         (
    (           )
   ( (  )   (  ) )
  (__(__)___(__)__)
"""

DRAGON_BANNER = r"""
    ___           _ _
   /   \_ __ __ _| (_) _   _ ___
  / /\ / '__/ _` | | || | | / __|
 / /_//| | | (_| | | || |_| \__ \
/___,' |_|  \__,_|_|_| \__,_|___/
"""


# ── banner ────────────────────────────────────────────────────────────────────


def _print_banner() -> None:
    if _RICH and _console:
        # Dragon + title panel
        title_text = Text()
        title_text.append("Aurelius", style="bold bright_red")
        title_text.append(f"  v{__version__}", style="dim white")
        title_text.append("  •  1.395B decoder-only LLM", style="dim white")

        dragon_text = Text(DRAGON_SMALL.rstrip(), style="bright_red")

        info_lines = Text()
        info_lines.append("\n  Type ", style="dim white")
        info_lines.append("/help", style="bold bright_cyan")
        info_lines.append(" for commands  •  ", style="dim white")
        info_lines.append("/quit", style="bold bright_cyan")
        info_lines.append(" to exit\n", style="dim white")

        combined = Text()
        combined.append_text(dragon_text)
        combined.append("\n")
        combined.append(DRAGON_BANNER, style="bold bright_red")
        combined.append_text(info_lines)

        panel = Panel(
            combined,
            border_style="bright_red",
            box=box.HEAVY,
            padding=(0, 2),
        )
        _console.print(panel)
    else:
        print(RED(BOLD(DRAGON_BANNER)))
        print(DIM(f"  version {__version__}  |  type /help for commands\n"))


# ── slash-command help ────────────────────────────────────────────────────────

CHAT_COMMANDS = {
    "/help": "show this help message",
    "/quit, /exit": "end the session",
    "/reset": "clear conversation history",
    "/history": "show conversation history",
    "/system <p>": "set a new system prompt",
    "/save <id>": "save conversation to disk",
    "/load <id>": "load a saved conversation",
    "/list": "list saved conversations",
    "/model": "show current model info",
    "/version": "show version",
    "/clear": "clear the screen",
}


def _print_help() -> None:
    if _RICH and _console:
        table = Table(
            show_header=True,
            header_style="bold bright_red",
            border_style="dim red",
            box=box.SIMPLE_HEAVY,
            padding=(0, 1),
            title="[bold bright_red]Aurelius Commands[/bold bright_red]",
        )
        table.add_column("Command", style="bold bright_cyan", no_wrap=True)
        table.add_column("Description", style="white")
        for cmd, desc in CHAT_COMMANDS.items():
            table.add_row(cmd, desc)
        _console.print()
        _console.print(table)
        _console.print()
    else:
        print(CYAN(BOLD("\n  Slash commands:")))
        for cmd, desc in CHAT_COMMANDS.items():
            print(f"    {GREEN(cmd):<26}  {desc}")
        print()


# ── default system prompt ─────────────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are Aurelius, a helpful, harmless, and honest AI assistant. "
    "You were built from scratch using pure PyTorch. "
    "Respond thoughtfully and accurately."
)


# ── model loader ──────────────────────────────────────────────────────────────


def _load_generate_fn(model_path: str, max_tokens: int, temperature: float):
    """Return a callable(prompt: str) -> str using a loaded AureliusTransformer."""
    import json
    import pathlib

    import torch

    from src.model.config import AureliusConfig
    from src.model.transformer import AureliusTransformer

    cfg_file = pathlib.Path(model_path) / "config.json"
    if cfg_file.exists():
        cfg_dict = json.loads(cfg_file.read_text())
        config = AureliusConfig(**cfg_dict)
    else:
        config = AureliusConfig()

    model = AureliusTransformer(config)

    p = pathlib.Path(model_path)
    if p.is_file():
        state = torch.load(str(p), map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=False)
    elif (p / "pytorch_model.bin").exists():
        state = torch.load(
            str(p / "pytorch_model.bin"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=False)

    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    def _generate(prompt: str) -> str:
        token_ids = list(prompt.encode("utf-8"))[: config.max_seq_len - max_tokens]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
        new_ids = out_ids[0, input_ids.shape[1] :].tolist()
        return bytes(new_ids).decode("utf-8", errors="replace")

    return _generate


def _mock_generate(prompt: str) -> str:
    """Placeholder response when no checkpoint is loaded."""
    last_user = ""
    for line in reversed(prompt.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("<|"):
            last_user = stripped
            break
    return (
        "[Aurelius is not connected to a trained checkpoint.]\n\n"
        "To load weights:  `aurelius chat --model-path <checkpoint>`\n\n"
        f'Your message was: "{last_user}"'
    )


# ── response rendering ────────────────────────────────────────────────────────


def _render_response(text: str) -> None:
    """Render an assistant response with markdown + code highlighting."""
    if _RICH and _console:
        # Detect if it's pure markdown-ish content
        has_code = "```" in text or "`" in text
        has_bullets = text.lstrip().startswith(("-", "*", "#"))
        if has_code or has_bullets or "\n" in text:
            try:
                md = Markdown(text, code_theme="monokai")
                panel = Panel(
                    md,
                    border_style="dim red",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
                _console.print(panel)
            except Exception:
                _console.print(text)
        else:
            _console.print(f"  {text}")
    else:
        print(text)
    print()


# ── loading spinner ───────────────────────────────────────────────────────────


def _thinking_spinner(generate_fn, prompt: str) -> str:
    """Run generate_fn with a spinner shown while waiting."""
    if _RICH and _console:
        result_box: list[str] = []

        def _run():
            result_box.append(generate_fn(prompt))

        import threading

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        with Live(
            Spinner("dots2", text=" [dim]Aurelius is thinking...[/dim]", style="bright_red"),
            console=_console,
            refresh_per_second=10,
            transient=True,
        ):
            t.join()

        return result_box[0] if result_box else ""
    else:
        return generate_fn(prompt)


# ── interactive chat REPL ─────────────────────────────────────────────────────


def _run_chat(
    system_prompt: str = DEFAULT_SYSTEM,
    model_path: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> None:
    try:
        from src.serving.conversation_store import ConversationStore

        store: Optional[ConversationStore] = ConversationStore()
    except Exception:
        store = None

    try:
        from src.serving.response_formatter import ResponseFormatter

        formatter = ResponseFormatter(max_length=max_tokens * 8)
    except Exception:
        formatter = None  # type: ignore[assignment]

    generate_fn = _mock_generate
    if model_path:
        try:
            generate_fn = _load_generate_fn(model_path, max_tokens, temperature)
            if _RICH and _console:
                _console.print(
                    f"  [aurelius.ok]Model loaded[/aurelius.ok] [dim]from {model_path}[/dim]\n"
                )
            else:
                print(DIM(f"  Model loaded from {model_path}\n"))
        except Exception as exc:
            if _RICH and _console:
                _console.print(
                    f"  [aurelius.warn]Warning:[/aurelius.warn] [dim]could not load model ({exc}). Running in mock mode.[/dim]\n"
                )
            else:
                print(YELLOW(f"  Warning: could not load model ({exc}). Running in mock mode.\n"))

    history: list[dict] = []
    current_conv_id: Optional[str] = None

    def _build_prompt() -> str:
        parts = [f"<|system|>\n{system_prompt}<|end|>\n"]
        for msg in history:
            tok = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
            parts.append(f"{tok}\n{msg['content']}<|end|>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    _print_banner()

    if _RICH and _console:
        _console.print(
            f"  [aurelius.system]System:[/aurelius.system] [dim]{textwrap.shorten(system_prompt, 80)}[/dim]\n"
        )
    else:
        print(f"  System: {DIM(textwrap.shorten(system_prompt, 80))}\n")

    while True:
        # ── prompt ────────────────────────────────────────────────────────
        try:
            if _RICH and _console:
                _console.print("[bold bright_white]You[/bold bright_white] [dim]>[/dim] ", end="")
                user_input = input().strip()
            else:
                user_input = input(f"{BLUE(BOLD('You'))} » ").strip()
        except (EOFError, KeyboardInterrupt):
            if _RICH and _console:
                _console.print("\n[dim]  Goodbye.[/dim]")
            else:
                print(f"\n{DIM('  Goodbye.')}")
            break

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                if _RICH and _console:
                    _console.print("[dim]  Goodbye.[/dim]")
                else:
                    print(DIM("  Goodbye."))
                break

            elif cmd == "/help":
                _print_help()

            elif cmd == "/clear":
                if os.name == "nt":
                    subprocess.run(["cmd", "/c", "cls"], check=False)
                else:
                    subprocess.run(["clear"], check=False)
                _print_banner()

            elif cmd == "/reset":
                history.clear()
                current_conv_id = None
                if _RICH and _console:
                    _console.print("[dim]  Conversation reset.[/dim]\n")
                else:
                    print(DIM("  Conversation reset.\n"))

            elif cmd == "/history":
                if not history:
                    if _RICH and _console:
                        _console.print("[dim]  No messages yet.[/dim]\n")
                    else:
                        print(DIM("  No messages yet.\n"))
                else:
                    if _RICH and _console:
                        table = Table(
                            show_header=False,
                            border_style="dim red",
                            box=box.SIMPLE,
                            padding=(0, 1),
                        )
                        table.add_column("Role", style="bold", no_wrap=True)
                        table.add_column("Content", style="white")
                        for m in history:
                            role_style = "bright_white" if m["role"] == "user" else "bright_green"
                            role_label = "You" if m["role"] == "user" else "Aurelius"
                            table.add_row(
                                f"[{role_style}]{role_label}[/{role_style}]",
                                textwrap.shorten(m["content"], 100),
                            )
                        _console.print()
                        _console.print(table)
                        _console.print()
                    else:
                        print()
                        for m in history:
                            who = BLUE("You") if m["role"] == "user" else GREEN("Aurelius")
                            print(f"  {who}: {textwrap.shorten(m['content'], 80)}")
                        print()

            elif cmd == "/system":
                if arg:
                    system_prompt = arg
                    history.clear()
                    if _RICH and _console:
                        _console.print("[dim]  System prompt updated. History cleared.[/dim]\n")
                    else:
                        print(DIM("  System prompt updated. History cleared.\n"))
                else:
                    if _RICH and _console:
                        _console.print(f"  [dim]Current: {system_prompt}[/dim]\n")
                    else:
                        print(DIM(f"  Current: {system_prompt}\n"))

            elif cmd == "/save":
                if store:
                    cid = arg or current_conv_id or "default"
                    store.save(cid, history)
                    current_conv_id = cid
                    if _RICH and _console:
                        _console.print(f"  [dim]Saved as '[bold]{cid}[/bold]'.[/dim]\n")
                    else:
                        print(DIM(f"  Saved as '{cid}'.\n"))
                else:
                    if _RICH and _console:
                        _console.print("[aurelius.warn]  Conversation store unavailable.[/aurelius.warn]\n")
                    else:
                        print(YELLOW("  Conversation store unavailable.\n"))

            elif cmd == "/load":
                if store and arg:
                    msgs = store.load(arg)
                    if msgs:
                        history = msgs
                        current_conv_id = arg
                        if _RICH and _console:
                            _console.print(
                                f"  [dim]Loaded '[bold]{arg}[/bold]' ({len(msgs)} messages).[/dim]\n"
                            )
                        else:
                            print(DIM(f"  Loaded '{arg}' ({len(msgs)} messages).\n"))
                    else:
                        if _RICH and _console:
                            _console.print(f"  [aurelius.warn]No conversation found: '{arg}'[/aurelius.warn]\n")
                        else:
                            print(YELLOW(f"  No conversation found: '{arg}'\n"))
                else:
                    if _RICH and _console:
                        _console.print("[aurelius.warn]  Usage: /load <conversation-id>[/aurelius.warn]\n")
                    else:
                        print(YELLOW("  Usage: /load <conversation-id>\n"))

            elif cmd == "/list":
                if store:
                    ids = store.list_conversations()
                    if ids:
                        if _RICH and _console:
                            _console.print()
                            for cid in sorted(ids):
                                mark = " [bold bright_red]●[/bold bright_red]" if cid == current_conv_id else ""
                                _console.print(f"  [bright_cyan]{cid}[/bright_cyan]{mark}")
                            _console.print()
                        else:
                            print(f"\n  {CYAN('Saved conversations:')}")
                            for cid in sorted(ids):
                                mark = " *" if cid == current_conv_id else ""
                                print(f"    {cid}{DIM(mark)}")
                            print()
                    else:
                        if _RICH and _console:
                            _console.print("[dim]  No saved conversations.[/dim]\n")
                        else:
                            print(DIM("  No saved conversations.\n"))
                else:
                    if _RICH and _console:
                        _console.print("[aurelius.warn]  Conversation store unavailable.[/aurelius.warn]\n")
                    else:
                        print(YELLOW("  Conversation store unavailable.\n"))

            elif cmd == "/model":
                mp = model_path or "none (mock mode)"
                if _RICH and _console:
                    _console.print(
                        f"\n  [bright_cyan]Model path:[/bright_cyan] [dim]{mp}[/dim]"
                        f"\n  [bright_cyan]Version:[/bright_cyan]    [dim]{__version__}[/dim]\n"
                    )
                else:
                    print(f"\n  {CYAN('Model path:')} {mp}")
                    print(f"  {CYAN('Version:')} {__version__}\n")

            elif cmd == "/version":
                if _RICH and _console:
                    _console.print(
                        f"  [bold bright_red]Aurelius[/bold bright_red] [dim]{__version__}[/dim]\n"
                    )
                else:
                    print(f"  Aurelius {__version__}\n")

            else:
                if _RICH and _console:
                    _console.print(
                        f"  [aurelius.warn]Unknown command:[/aurelius.warn] [bold]{cmd}[/bold]  "
                        "(type [bold bright_cyan]/help[/bold bright_cyan])\n"
                    )
                else:
                    print(YELLOW(f"  Unknown command: {cmd}  (type /help)\n"))

            continue

        # ── normal message ────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        prompt = _build_prompt()

        if _RICH and _console:
            _console.print(
                "[bold bright_green]Aurelius[/bold bright_green] [dim]>[/dim] ",
                end="",
            )
            try:
                response = _thinking_spinner(generate_fn, prompt)
                if formatter is not None:
                    response = formatter.format(response)
            except Exception as exc:
                response = f"[error: {exc}]"
            _console.print()
            _render_response(response)
        else:
            print(f"{GREEN(BOLD('Aurelius'))} » ", end="", flush=True)
            try:
                response = generate_fn(prompt)
                if formatter is not None:
                    response = formatter.format(response)
            except Exception as exc:
                response = RED(f"[error: {exc}]")
            print(response)
            print()

        history.append({"role": "assistant", "content": response})


# ── serve command ─────────────────────────────────────────────────────────────


def _run_serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    ui_port: int = 7860,
    model_path: Optional[str] = None,
) -> None:
    import threading
    import time
    import webbrowser

    if _RICH and _console:
        _console.print(
            Panel(
                "[bold bright_red]Starting Aurelius Servers[/bold bright_red]",
                border_style="bright_red",
                box=box.HEAVY,
                padding=(0, 2),
            )
        )
    else:
        print(CYAN(BOLD("\n  Starting Aurelius servers...\n")))

    try:
        from src.serving.api_server import create_server, make_mock_generate_fn

        api_server = create_server(host, port, make_mock_generate_fn())
        threading.Thread(target=api_server.serve_forever, daemon=True).start()
        if _RICH and _console:
            _console.print(
                f"  [aurelius.ok]✓[/aurelius.ok] API server  →  "
                f"[bright_cyan]http://{host}:{port}/v1/chat/completions[/bright_cyan]"
            )
        else:
            print(f"  {GREEN('OK')} API server  ->  http://{host}:{port}/v1/chat/completions")
    except Exception as exc:
        if _RICH and _console:
            _console.print(f"  [aurelius.error]✗  API server: {exc}[/aurelius.error]")
        else:
            print(RED(f"  FAIL  API server: {exc}"))

    try:
        from src.serving.web_ui import create_ui_server, make_mock_generate_fn as mk_ui

        ui_server = create_ui_server(
            host,
            ui_port,
            mk_ui(),
            api_url=f"http://{host}:{port}/v1/chat/completions",
        )
        threading.Thread(target=ui_server.serve_forever, daemon=True).start()
        if _RICH and _console:
            _console.print(
                f"  [aurelius.ok]✓[/aurelius.ok] Web UI      →  "
                f"[bright_cyan]http://{host}:{ui_port}[/bright_cyan]"
            )
        else:
            print(f"  {GREEN('OK')} Web UI      ->  http://{host}:{ui_port}")
    except Exception as exc:
        if _RICH and _console:
            _console.print(f"  [aurelius.error]✗  Web UI: {exc}[/aurelius.error]")
        else:
            print(RED(f"  FAIL  Web UI: {exc}"))

    ui_url = f"http://{host}:{ui_port}"
    if _RICH and _console:
        _console.print()
        _console.print("[dim]  Opening browser…  Press Ctrl+C to stop.[/dim]\n")
    else:
        print(f"\n  {DIM('Opening browser...')}")
        print(f"  {DIM('Press Ctrl+C to stop.')}\n")

    try:
        webbrowser.open(ui_url)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if _RICH and _console:
            _console.print("\n[dim]  Servers stopped.[/dim]")
        else:
            print(f"\n{DIM('  Servers stopped.')}")


# ── argument parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aurelius",
        description="Aurelius -- 1.395B AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
        """\
        examples:
          aurelius                         start interactive chat
          aurelius chat -s coding          use the 'coding' persona
          aurelius chat --model-path ckpt  load trained weights
          aurelius backend list            inspect backend adapters
          aurelius session list            inspect persistent sessions
          aurelius serve                   API server + web UI
          aurelius serve --port 8080       API on custom port
          aurelius train --config ...      launch training
          aurelius eval checkpoints/run    run evaluation harness
        """
        ),
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"aurelius {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # chat
    chat_p = sub.add_parser("chat", help="interactive chat (default)")
    chat_p.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM,
        metavar="PERSONA_OR_PROMPT",
        help="system prompt text or persona name (default/coding/security/...)",
    )
    chat_p.add_argument("--model-path", metavar="PATH", help="checkpoint directory or .pt file")
    chat_p.add_argument("--max-tokens", type=int, default=1024)
    chat_p.add_argument("--temperature", type=float, default=0.7)

    # serve
    serve_p = sub.add_parser("serve", help="start API server + web UI")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)
    serve_p.add_argument("--ui-port", type=int, default=7860)
    serve_p.add_argument("--model-path", metavar="PATH")

    # train
    train_p = sub.add_parser("train", help="launch training")
    train_p.add_argument("--config", default="configs/train_1b.yaml")
    train_p.add_argument("--resume", metavar="DIR")

    # eval
    eval_p = sub.add_parser("eval", help="run evaluation harness")
    eval_p.add_argument("checkpoint", nargs="?")
    eval_p.add_argument("--results-dir", default="results")

    from src.cli.family_commands import build_family_parser
    from src.cli.interface_commands import build_interface_parser
    from src.cli.backend_commands import build_backend_parser
    from src.cli.session_commands import build_session_parser

    build_family_parser(sub)
    build_interface_parser(sub)
    build_backend_parser(sub)
    build_session_parser(sub)

    return parser


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None or args.command == "chat":
        system = getattr(args, "system", DEFAULT_SYSTEM)
        # Resolve persona names
        try:
            from src.serving.system_prompts import SystemPromptLibrary

            lib = SystemPromptLibrary()
            if system in lib.list_personas():
                system = lib.get(system)
        except Exception:
            pass

        _run_chat(
            system_prompt=system,
            model_path=getattr(args, "model_path", None),
            max_tokens=getattr(args, "max_tokens", 1024),
            temperature=getattr(args, "temperature", 0.7),
        )

    elif args.command == "serve":
        _run_serve(
            host=args.host,
            port=args.port,
            ui_port=args.ui_port,
            model_path=getattr(args, "model_path", None),
        )

    elif args.command == "train":
        import subprocess

        cmd = [sys.executable, "-m", "src.training.trainer", "--config", args.config]
        if args.resume:
            cmd += ["--resume", args.resume]
        if _RICH and _console:
            _console.print(
                Panel(
                    f"[bold bright_red]Aurelius Training[/bold bright_red]\n\n"
                    f"[dim]{' '.join(cmd)}[/dim]",
                    border_style="bright_red",
                    box=box.HEAVY,
                    padding=(0, 2),
                )
            )
        else:
            print(CYAN(BOLD("\n  Aurelius Training\n")))
            print(DIM(f"  {' '.join(cmd)}\n"))
        return subprocess.call(cmd)

    elif args.command == "eval":
        import subprocess

        if not args.checkpoint:
            if _RICH and _console:
                _console.print(
                    "[aurelius.warn]  Usage: aurelius eval <checkpoint-dir>[/aurelius.warn]\n"
                )
            else:
                print(YELLOW("  Usage: aurelius eval <checkpoint-dir>\n"))
            return 1
        cmd = [
            sys.executable,
            "-m",
            "src.eval.harness",
            args.checkpoint,
            "--results-dir",
            args.results_dir,
        ]
        if _RICH and _console:
            _console.print(
                Panel(
                    f"[bold bright_red]Aurelius Evaluation[/bold bright_red]\n\n"
                    f"[dim]{' '.join(cmd)}[/dim]",
                    border_style="bright_red",
                    box=box.HEAVY,
                    padding=(0, 2),
                )
            )
        else:
            print(CYAN(BOLD("\n  Aurelius Evaluation\n")))
            print(DIM(f"  {' '.join(cmd)}\n"))
        return subprocess.call(cmd)

    elif args.command == "family":
        try:
            from src.cli.family_commands import dispatch_family_command

            return dispatch_family_command(args, sys.stdout)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    elif args.command == "backend":
        try:
            from src.cli.backend_commands import dispatch_backend_command

            return dispatch_backend_command(args, sys.stdout)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    elif args.command == "session":
        try:
            from src.cli.session_commands import dispatch_session_command

            return dispatch_session_command(args, sys.stdout)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    elif args.command == "interface":
        try:
            from src.cli.interface_commands import dispatch_interface_command

            return dispatch_interface_command(args, sys.stdout)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
