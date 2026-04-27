"""
Interactive terminal chat REPL for Aurelius.

Run directly: python -m src.serving.terminal_chat
Or:           python -m src.serving.terminal_chat --system "You are a Python tutor"
"""

from __future__ import annotations

import sys
from collections.abc import Callable

# ChatML special tokens (must match tokenizer_config.json)
_SYSTEM_TOKEN = "<|system|>"  # noqa: S105
_USER_TOKEN = "<|user|>"  # noqa: S105
_ASSISTANT_TOKEN = "<|assistant|>"  # noqa: S105
_END_TOKEN = "<|end|>"  # noqa: S105

_DEFAULT_SYSTEM = (
    "You are Aurelius, a helpful, harmless, and honest AI assistant. "
    "Respond thoughtfully and accurately."
)

_ANSI = {
    "blue": "\033[34m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}

_MOCK_RESPONSE = "[model not loaded — mock response]"

_WELCOME_BANNER = (
    "Aurelius Terminal Chat\n"
    "Type your message and press Enter. Special commands:\n"
    "  /quit            — exit\n"
    "  /reset           — clear conversation history\n"
    "  /history         — show conversation history\n"
    "  /system <prompt> — replace the system prompt\n" + "-" * 50
)


def build_chatml_prompt(messages: list[dict], system_prompt: str) -> str:
    """Build a ChatML prompt string from a message list and a system prompt.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
        system_prompt: Text for the system turn prepended before all messages.

    Returns:
        Formatted ChatML string ready for tokenization.
    """
    parts: list[str] = []
    parts.append(f"{_SYSTEM_TOKEN}\n{system_prompt}{_END_TOKEN}\n")
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"{_USER_TOKEN}\n{content}{_END_TOKEN}\n")
        elif role == "assistant":
            parts.append(f"{_ASSISTANT_TOKEN}\n{content}{_END_TOKEN}\n")
    parts.append(_ASSISTANT_TOKEN + "\n")
    return "".join(parts)


class TerminalChat:
    """Interactive terminal REPL that wraps a generation callable.

    Args:
        model: Unused placeholder kept for API compatibility; may be None.
        tokenizer: Unused placeholder kept for API compatibility; may be None.
        gen_config: Optional generation config object (stored but not used here).
        system_prompt: Initial system prompt for the session.
        use_colors: Whether to emit ANSI color codes in output.
        generate_fn: Callable[[str], str] that maps a prompt to a response
            string. Defaults to a mock that returns a fixed placeholder string.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        gen_config=None,
        system_prompt: str | None = None,
        use_colors: bool = True,
        generate_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.system_prompt: str = system_prompt or _DEFAULT_SYSTEM
        self.use_colors: bool = use_colors
        self.generate_fn: Callable[[str], str] = (
            generate_fn if generate_fn is not None else lambda _prompt: _MOCK_RESPONSE
        )
        self.history: list[dict] = []

    def _colorize(self, text: str, color: str) -> str:
        """Wrap text in ANSI escape codes for the given color name.

        Args:
            text: Plain text to colorize.
            color: One of 'blue', 'green', 'yellow', 'reset'.

        Returns:
            ANSI-escaped string if use_colors is True, otherwise text unchanged.
        """
        if not self.use_colors:
            return text
        code = _ANSI.get(color, "")
        reset = _ANSI["reset"]
        return f"{code}{text}{reset}"

    def _format_user_turn(self, text: str) -> str:
        """Return user turn label + text, colorized in blue."""
        return self._colorize(f"\n[You]: {text}", "blue")

    def _format_assistant_label(self) -> str:
        """Return assistant label string, colorized in green."""
        return self._colorize("\n[Aurelius]: ", "green")

    def add_message(self, role: str, content: str) -> None:
        """Append a message dict to history.

        Args:
            role: 'user' or 'assistant'.
            content: Message text.
        """
        self.history.append({"role": role, "content": content})

    def build_prompt(self) -> str:
        """Build a full ChatML prompt from the current history and system prompt."""
        return build_chatml_prompt(self.history, self.system_prompt)

    def reset(self) -> None:
        """Clear conversation history, retaining the current system prompt."""
        self.history = []

    def chat(self, user_input: str) -> str:
        """Process one user turn and return the assistant response.

        Appends the user message to history, builds the prompt, calls
        generate_fn, appends the assistant response to history, and returns
        the response string.

        Args:
            user_input: The user's message text.

        Returns:
            The assistant's response string.
        """
        self.add_message("user", user_input)
        prompt = self.build_prompt()
        response = self.generate_fn(prompt)
        self.add_message("assistant", response)
        return response

    def run_loop(self) -> None:
        """Run the interactive REPL until the user exits.

        Handles special commands:
            /quit            — exit the loop
            /reset           — clear conversation history
            /history         — print all history entries
            /system <prompt> — replace the system prompt
        """
        print(self._colorize(_WELCOME_BANNER, "yellow"))

        while True:
            try:
                user_input = input(self._colorize("\n[You]: ", "blue")).strip()
            except (EOFError, KeyboardInterrupt):
                print(self._colorize("\n[Aurelius]: Goodbye!", "yellow"))
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print(self._colorize("[Goodbye!]", "yellow"))
                break

            if user_input == "/reset":
                self.reset()
                print(self._colorize("[Conversation history cleared.]", "yellow"))
                continue

            if user_input == "/history":
                if not self.history:
                    print(self._colorize("[No history yet.]", "yellow"))
                else:
                    for i, msg in enumerate(self.history):
                        role_label = f"[{msg['role']}]"
                        print(self._colorize(f"{i + 1}. {role_label}: {msg['content']}", "yellow"))
                continue

            if user_input.startswith("/system "):
                new_prompt = user_input[len("/system ") :].strip()
                if new_prompt:
                    self.system_prompt = new_prompt
                    self.reset()
                    print(self._colorize("[System prompt updated. History cleared.]", "yellow"))
                else:
                    print(self._colorize("[Usage: /system <new prompt text>]", "yellow"))
                continue

            print(self._format_assistant_label(), end="", flush=True)
            response = self.chat(user_input)
            print(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius terminal chat REPL")
    parser.add_argument(
        "--system",
        type=str,
        default=_DEFAULT_SYSTEM,
        help="System prompt for the session.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a model checkpoint directory.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per turn.",
    )
    args = parser.parse_args()

    generate_fn: Callable[[str], str] | None = None

    if args.model_path is not None:
        try:
            from src.serving.chat_session import (
                ChatSession,
                GenerationConfig,
                load_model_for_chat,
            )

            model, tokenizer = load_model_for_chat(args.model_path)
            gen_cfg = GenerationConfig(max_new_tokens=args.max_tokens)
            _session = ChatSession(model, tokenizer, system_prompt=args.system, gen_cfg=gen_cfg)

            def generate_fn(prompt: str) -> str:
                return _session.chat.__func__(_session, "")

        except Exception as exc:
            print(
                f"[Warning] Could not load model from {args.model_path!r}: {exc}", file=sys.stderr
            )
            print("[Falling back to mock generate_fn]", file=sys.stderr)

    tc = TerminalChat(
        system_prompt=args.system,
        generate_fn=generate_fn,
    )
    tc.run_loop()
