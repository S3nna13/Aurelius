"""Tab completion engine for interactive CLI sessions."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompletionEngine:
    """Provides tab-completion candidates for CLI commands."""

    commands: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def register(self, command: str, subcommands: list[str]) -> None:
        self.commands[command] = subcommands

    def complete(self, line: str) -> list[str]:
        parts = line.strip().split()
        if not parts or len(parts) == 1:
            token = parts[0] if parts else ""
            return [c for c in self.commands if c.startswith(token)]
        cmd = parts[0]
        token = parts[-1] if len(parts) > 1 else ""
        subcommands = self.commands.get(cmd, [])
        return [s for s in subcommands if s.startswith(token)]

    def add_dynamic(self, command: str, completer):
        self.commands[command] = completer

    def registered_commands(self) -> list[str]:
        return list(self.commands.keys())


COMPLETION_ENGINE = CompletionEngine()