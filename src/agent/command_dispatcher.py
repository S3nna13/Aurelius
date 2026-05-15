"""Command dispatcher for routing parsed agent commands to handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.agent.nl_command_parser import ParsedCommand
from src.agent.skill_executor import SkillContext

__all__ = [
    "CommandDispatchError",
    "DispatchResult",
    "CommandDispatcher",
    "DEFAULT_COMMAND_DISPATCHER",
    "COMMAND_DISPATCHER_REGISTRY",
]


class CommandDispatchError(Exception):
    """Raised when a command cannot be dispatched."""


@dataclass
class DispatchResult:
    """Outcome of a command dispatch."""

    success: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandDispatcher:
    """Routes :class:`ParsedCommand` instances to the appropriate backend."""

    skill_catalog: Any | None = None
    plugin_loader: Any | None = None
    skill_executor: Any | None = None
    trigger_engine: Any | None = None

    def __post_init__(self) -> None:
        self._handlers: dict[str, Callable[[ParsedCommand], DispatchResult]] = {}

    def register_handler(
        self,
        action: str,
        handler: Callable[[ParsedCommand], DispatchResult],
    ) -> None:
        """Register a custom *handler* for *action*."""
        self._handlers[action] = handler

    def dispatch(self, command: ParsedCommand) -> DispatchResult:
        """Route *command* to its handler and return a :class:`DispatchResult`."""
        handler = self._handlers.get(command.action)
        if handler is not None:
            try:
                return handler(command)
            except Exception as exc:  # noqa: BLE001
                return DispatchResult(success=False, output=str(exc))

        known_actions = {
            "run_skill",
            "list_skills",
            "activate_skill",
            "deactivate_skill",
            "load_plugin",
            "list_plugins",
            "agent_status",
            "list_agents",
            "run_task",
            "show_board",
            "chat",
        }
        if command.action not in known_actions:
            raise CommandDispatchError(f"Unknown action: {command.action}")

        try:
            return self._dispatch_default(command)
        except Exception as exc:  # noqa: BLE001
            return DispatchResult(success=False, output=str(exc))

    def _dispatch_default(self, command: ParsedCommand) -> DispatchResult:
        """Handle built-in actions."""
        action = command.action

        if action == "run_skill":
            if self.skill_catalog is None:
                return DispatchResult(
                    success=False,
                    output="run_skill requires skill_catalog which is not configured",
                )
            if self.skill_executor is None:
                return DispatchResult(
                    success=False,
                    output="run_skill requires skill_executor which is not configured",
                )
            target = command.target
            if not target:
                raise CommandDispatchError("run_skill requires a target")
            skill = self.skill_catalog.get(target)
            if skill is None:
                raise CommandDispatchError(f"Skill not found: {target}")
            instructions = skill.instructions
            context = SkillContext(variables=command.args)
            result = self.skill_executor.execute(target, instructions, context)
            return DispatchResult(success=True, output=result.output)

        if action == "list_skills":
            if self.skill_catalog is None:
                return DispatchResult(
                    success=False,
                    output="list_skills requires skill_catalog which is not configured",
                )
            skills = self.skill_catalog.list()
            ids = [getattr(s, "skill_id", str(s)) for s in skills]
            return DispatchResult(
                success=True,
                output=", ".join(ids) if ids else "No skills found",
            )

        if action == "activate_skill":
            if self.skill_catalog is None:
                return DispatchResult(
                    success=False,
                    output="activate_skill requires skill_catalog which is not configured",
                )
            target = command.target
            if not target:
                raise CommandDispatchError("activate_skill requires a target")
            self.skill_catalog.activate(target)
            return DispatchResult(success=True, output=f"Skill {target} activated")

        if action == "deactivate_skill":
            if self.skill_catalog is None:
                return DispatchResult(
                    success=False,
                    output="deactivate_skill requires skill_catalog which is not configured",
                )
            target = command.target
            if not target:
                raise CommandDispatchError("deactivate_skill requires a target")
            self.skill_catalog.deactivate(target)
            return DispatchResult(success=True, output=f"Skill {target} deactivated")

        if action == "load_plugin":
            if self.plugin_loader is None:
                return DispatchResult(
                    success=False,
                    output="load_plugin requires plugin_loader which is not configured",
                )
            target = command.target
            if not target:
                raise CommandDispatchError("load_plugin requires a target")
            entry_point = command.args.get("entry_point", target)
            self.plugin_loader.load(target, entry_point)
            return DispatchResult(success=True, output=f"Plugin {target} loaded")

        if action == "list_plugins":
            if self.plugin_loader is None:
                return DispatchResult(
                    success=False,
                    output="list_plugins requires plugin_loader which is not configured",
                )
            plugins = self.plugin_loader.list_loaded()
            return DispatchResult(
                success=True,
                output=", ".join(plugins) if plugins else "No plugins loaded",
            )

        if action == "agent_status":
            return DispatchResult(success=True, output="Agent status: operational")

        if action == "list_agents":
            return DispatchResult(success=True, output="Agents: default")

        if action == "run_task":
            target = command.target or "unnamed"
            return DispatchResult(success=True, output=f"Task started: {target}")

        if action == "show_board":
            return DispatchResult(success=True, output="Board view: no active items")

        if action == "chat":
            return DispatchResult(success=True, output=command.raw_text)

        # Defensive fallback — should never reach here because of the guard in dispatch().
        raise CommandDispatchError(f"Unhandled action: {action}")


DEFAULT_COMMAND_DISPATCHER = CommandDispatcher()

COMMAND_DISPATCHER_REGISTRY: dict[str, CommandDispatcher] = {
    "default": DEFAULT_COMMAND_DISPATCHER,
}
