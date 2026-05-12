"""CLI commands for agent interaction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class AgentCLICommand:
    """A single CLI command for agent interaction."""

    name: str
    description: str
    handler: Callable[[list[str]], str]
    aliases: list[str] = field(default_factory=list)


class AgentCLIDispatcher:
    """Dispatch CLI commands to agent handlers."""

    def __init__(self) -> None:
        self._commands: dict[str, AgentCLICommand] = {}

    def register(self, cmd: AgentCLICommand) -> None:
        for name in [cmd.name] + cmd.aliases:
            self._commands[name] = cmd

    def dispatch(self, raw: str) -> str:
        parts = raw.strip().split()
        if not parts:
            return "no command"
        name = parts[0]
        if name not in self._commands:
            return f"unknown command: {name}"
        return self._commands[name].handler(parts[1:])

    def list_commands(self) -> list[AgentCLICommand]:
        seen = set()
        result: list[AgentCLICommand] = []
        for cmd in self._commands.values():
            if id(cmd) not in seen:
                seen.add(id(cmd))
                result.append(cmd)
        return result


AGENT_CLI = AgentCLIDispatcher()
