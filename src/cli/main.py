"""
src/cli/main.py

Aurelius CLI -- interactive AI assistant and model toolkit.

Usage:
    aurelius                    # launch interactive chat
    aurelius chat               # interactive chat (explicit)
    aurelius chat -s "..."      # chat with custom system prompt
    aurelius serve              # start API server + web UI
    aurelius serve --port 8080  # custom port
    aurelius train              # launch training
    aurelius eval <ckpt>        # run evaluation
    aurelius --version          # print version
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Optional

__version__ = "0.1.0"

# ── ANSI colours (disabled on non-TTY / NO_COLOR env) ────────────────────────

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

BOLD    = lambda t: _c("1",  t)
DIM     = lambda t: _c("2",  t)
GREEN   = lambda t: _c("32", t)
CYAN    = lambda t: _c("36", t)
YELLOW  = lambda t: _c("33", t)
BLUE    = lambda t: _c("34", t)
RED     = lambda t: _c("31", t)
MAGENTA = lambda t: _c("35", t)


# ── banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
   ___                _ _
  / _ \ _   _ _ __ ___| (_) _   _ ___
 / /_\/| | | | '__/ _ \ | || | | / __|
/ /_\\ | |_| | | |  __/ | || |_| \__ \
\____/ \__,_|_|  \___|_|_| \__,_|___/

  1.395B decoder-only language model
"""

def _print_banner() -> None:
    print(MAGENTA(BANNER))
    print(DIM(f"  version {__version__}  |  type /help for commands\n"))


# ── slash-command help ────────────────────────────────────────────────────────

CHAT_COMMANDS = {
    "/help":        "show this help message",
    "/quit, /exit": "end the session",
    "/reset":       "clear conversation history",
    "/history":     "show conversation history",
    "/system <p>":  "set a new system prompt",
    "/save <id>":   "save conversation to disk",
    "/load <id>":   "load a saved conversation",
    "/list":        "list saved conversations",
    "/model":       "show current model info",
    "/version":     "show version",
    "/clear":       "clear the screen",
}

def _print_help() -> None:
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
        state = torch.load(str(p / "pytorch_model.bin"), map_location="cpu", weights_only=True)
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
        new_ids = out_ids[0, input_ids.shape[1]:].tolist()
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
        "To load weights:  aurelius chat --model-path <checkpoint>\n"
        f'Your message was: "{last_user}"'
    )


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
            print(DIM(f"  Model loaded from {model_path}\n"))
        except Exception as exc:
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
    print(f"  System: {DIM(textwrap.shorten(system_prompt, 80))}\n")

    while True:
        try:
            user_input = input(f"{BLUE(BOLD('You'))} » ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM('  Goodbye.')}")
            break

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd  = parts[0].lower()
            arg  = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print(DIM("  Goodbye."))
                break

            elif cmd == "/help":
                _print_help()

            elif cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                _print_banner()

            elif cmd == "/reset":
                history.clear()
                current_conv_id = None
                print(DIM("  Conversation reset.\n"))

            elif cmd == "/history":
                if not history:
                    print(DIM("  No messages yet.\n"))
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
                    print(DIM("  System prompt updated. History cleared.\n"))
                else:
                    print(DIM(f"  Current: {system_prompt}\n"))

            elif cmd == "/save":
                if store:
                    cid = arg or current_conv_id or "default"
                    store.save(cid, history)
                    current_conv_id = cid
                    print(DIM(f"  Saved as '{cid}'.\n"))
                else:
                    print(YELLOW("  Conversation store unavailable.\n"))

            elif cmd == "/load":
                if store and arg:
                    msgs = store.load(arg)
                    if msgs:
                        history = msgs
                        current_conv_id = arg
                        print(DIM(f"  Loaded '{arg}' ({len(msgs)} messages).\n"))
                    else:
                        print(YELLOW(f"  No conversation found: '{arg}'\n"))
                else:
                    print(YELLOW("  Usage: /load <conversation-id>\n"))

            elif cmd == "/list":
                if store:
                    ids = store.list_conversations()
                    if ids:
                        print(f"\n  {CYAN('Saved conversations:')}")
                        for cid in sorted(ids):
                            mark = " *" if cid == current_conv_id else ""
                            print(f"    {cid}{DIM(mark)}")
                        print()
                    else:
                        print(DIM("  No saved conversations.\n"))
                else:
                    print(YELLOW("  Conversation store unavailable.\n"))

            elif cmd == "/model":
                mp = model_path or "none (mock mode)"
                print(f"\n  {CYAN('Model path:')} {mp}")
                print(f"  {CYAN('Version:')} {__version__}\n")

            elif cmd == "/version":
                print(f"  Aurelius {__version__}\n")

            else:
                print(YELLOW(f"  Unknown command: {cmd}  (type /help)\n"))

            continue

        # ── normal message ────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        prompt = _build_prompt()

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

def _run_serve(host: str = "127.0.0.1", port: int = 8080,
               ui_port: int = 7860, model_path: Optional[str] = None) -> None:
    import threading
    import time
    import webbrowser

    print(CYAN(BOLD("\n  Starting Aurelius servers...\n")))

    try:
        from src.serving.api_server import create_server, make_mock_generate_fn
        api_server = create_server(host, port, make_mock_generate_fn())
        threading.Thread(target=api_server.serve_forever, daemon=True).start()
        print(f"  {GREEN('OK')} API server  ->  http://{host}:{port}/v1/chat/completions")
    except Exception as exc:
        print(RED(f"  FAIL  API server: {exc}"))

    try:
        from src.serving.web_ui import create_ui_server, make_mock_generate_fn as mk_ui
        ui_server = create_ui_server(host, ui_port, mk_ui())
        threading.Thread(target=ui_server.serve_forever, daemon=True).start()
        print(f"  {GREEN('OK')} Web UI      ->  http://{host}:{ui_port}")
    except Exception as exc:
        print(RED(f"  FAIL  Web UI: {exc}"))

    ui_url = f"http://{host}:{ui_port}"
    print(f"\n  {DIM('Opening browser...')}")
    print(f"  {DIM('Press Ctrl+C to stop.')}\n")
    try:
        webbrowser.open(ui_url)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{DIM('  Servers stopped.')}")


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aurelius",
        description="Aurelius -- 1.395B AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          aurelius                         start interactive chat
          aurelius chat -s coding          use the 'coding' persona
          aurelius chat --model-path ckpt  load trained weights
          aurelius serve                   API server + web UI
          aurelius serve --port 8080       API on custom port
          aurelius train --config ...      launch training
          aurelius eval checkpoints/run    run evaluation harness
        """),
    )
    parser.add_argument("--version", "-V", action="version",
                        version=f"aurelius {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    # chat
    chat_p = sub.add_parser("chat", help="interactive chat (default)")
    chat_p.add_argument("-s", "--system", default=DEFAULT_SYSTEM,
                        metavar="PERSONA_OR_PROMPT",
                        help="system prompt text or persona name (default/coding/security/...)")
    chat_p.add_argument("--model-path", metavar="PATH",
                        help="checkpoint directory or .pt file")
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
        print(CYAN(BOLD("\n  Aurelius Training\n")))
        print(DIM(f"  {' '.join(cmd)}\n"))
        return subprocess.call(cmd)

    elif args.command == "eval":
        import subprocess
        if not args.checkpoint:
            print(YELLOW("  Usage: aurelius eval <checkpoint-dir>\n"))
            return 1
        cmd = [sys.executable, "-m", "src.eval.harness",
               args.checkpoint, "--results-dir", args.results_dir]
        print(CYAN(BOLD("\n  Aurelius Evaluation\n")))
        print(DIM(f"  {' '.join(cmd)}\n"))
        return subprocess.call(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
