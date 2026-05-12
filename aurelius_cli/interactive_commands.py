"""Interactive CLI command dispatcher for agent shells."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CommandResult:
    success: bool
    output: str


@dataclass
class CLICommand:
    name: str
    handler: Callable[[list[str]], CommandResult]
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class InteractiveCLI:
    commands: dict[str, CLICommand] = field(default_factory=dict, repr=False)

    def register(
        self,
        name: str,
        handler: Callable,
        *,
        description: str = "",
        aliases: list[str] | None = None,
    ) -> None:
        cmd = CLICommand(name=name, handler=handler, description=description, aliases=aliases or [])
        for alias in [name] + (aliases or []):
            self.commands[alias] = cmd

    def execute(self, line: str) -> CommandResult:
        parts = line.strip().split()
        if not parts:
            return CommandResult(False, "no command")
        name = parts[0]
        if name not in self.commands:
            return CommandResult(False, f"unknown command: {name}")
        try:
            return self.commands[name].handler(parts[1:])
        except Exception as e:
            return CommandResult(False, str(e))

    def list_commands(self) -> list[CLICommand]:
        seen: set[int] = set()
        result: list[CLICommand] = []
        for cmd in self.commands.values():
            if id(cmd) not in seen:
                seen.add(id(cmd))
                result.append(cmd)
        return result

    def has_command(self, name: str) -> bool:
        return name in self.commands


INTERACTIVE_CLI = InteractiveCLI()
