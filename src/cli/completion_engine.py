"""Completion engine — tab-completion hints for the Aurelius REPL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CompletionKind(str, Enum):
    COMMAND = "command"
    ARGUMENT = "argument"
    FILEPATH = "filepath"
    HISTORY = "history"
    PERSONA = "persona"


@dataclass
class Completion:
    text: str
    kind: CompletionKind
    description: str = ""


class CompletionEngine:
    """Provide tab-completion candidates for partial input."""

    COMMANDS: list[str] = [
        "chat",
        "eval",
        "train",
        "serve",
        "export",
        "version",
        "help",
        "quit",
    ]

    # Argument hints per command
    _COMMAND_ARGS: dict[str, list[str]] = {
        "chat": ["--model", "--persona", "--temperature", "--max-tokens"],
        "eval": ["--dataset", "--metric", "--output"],
        "train": ["--config", "--resume", "--epochs", "--lr"],
        "serve": ["--host", "--port", "--workers"],
        "export": ["--format", "--output", "--quantize"],
        "version": [],
        "help": [],
        "quit": [],
    }

    # Fake stub paths returned for file-path completion
    _STUB_PATHS: list[str] = ["./models/", "./configs/", "./data/"]

    def __init__(self) -> None:
        self._commands: list[str] = list(self.COMMANDS)
        self._command_descriptions: dict[str, str] = {}
        self._history: list[str] = []

    # ------------------------------------------------------------------
    # History injection (called by ReplSession integration)
    # ------------------------------------------------------------------

    def feed_history(self, entries: list[str]) -> None:
        """Replace the completion history with *entries*."""
        self._history = list(entries)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def complete(
        self,
        partial: str,
        context: Optional[dict] = None,  # noqa: ARG002
    ) -> list[Completion]:
        """Return a list of Completion objects for *partial* input."""

        # Filepath completion: "./" prefix or absolute path with sub-segments
        if partial.startswith("./") or (
            partial.startswith("/") and "/" in partial[1:]
        ):
            return [
                Completion(text=p, kind=CompletionKind.FILEPATH)
                for p in self._STUB_PATHS
            ]

        # Command completion: starts with "/" (single-segment, e.g. "/chat")
        if partial.startswith("/"):
            prefix = partial[1:]
            return [
                Completion(
                    text=f"/{cmd}",
                    kind=CompletionKind.COMMAND,
                    description=self._command_descriptions.get(cmd, ""),
                )
                for cmd in self._commands
                if cmd.startswith(prefix)
            ]

        # Exact history match
        if partial in self._history:
            return [
                Completion(text=partial, kind=CompletionKind.HISTORY)
            ]

        # History prefix match
        matches = [h for h in self._history if h.startswith(partial) and h != partial]
        return [
            Completion(text=m, kind=CompletionKind.HISTORY)
            for m in matches
        ]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_command(self, name: str, description: str = "") -> None:
        """Add *name* to the command list if not already present."""
        if name not in self._commands:
            self._commands.append(name)
        self._command_descriptions[name] = description

    # ------------------------------------------------------------------
    # Argument hints
    # ------------------------------------------------------------------

    def get_completions_for(self, command: str) -> list[str]:
        """Return argument hints for *command*."""
        return list(self._COMMAND_ARGS.get(command, []))
