"""Agent mode registry for Aurelius.

Modes are presets that change how the agent behaves — system prompt prefix,
allowed tools, and response style.  Inspired by Roo-Code's mode system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "AgentModeError",
    "AgentMode",
    "AgentModeRegistry",
    "DEFAULT_MODE_REGISTRY",
    "AGENT_MODE_REGISTRY",
]


class AgentModeError(Exception):
    """Raised when a mode operation is invalid."""


@dataclass
class AgentMode:
    """A single agent behavior preset."""

    mode_id: str
    name: str
    description: str
    system_prompt_prefix: str
    allowed_tools: list[str]
    response_style: str
    custom_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentModeRegistry:
    """Stores and queries :class:`AgentMode` presets."""

    _modes: dict[str, AgentMode] = field(default_factory=dict)

    def register(self, mode: AgentMode) -> None:
        """Add *mode* to the registry.

        Raises:
            AgentModeError: If ``mode_id`` is empty or already registered.
        """
        if not isinstance(mode.mode_id, str) or not mode.mode_id.strip():
            raise AgentModeError("mode_id must be a non-empty string")
        if mode.mode_id in self._modes:
            raise AgentModeError(f"mode_id '{mode.mode_id}' is already registered")
        self._modes[mode.mode_id] = mode

    def unregister(self, mode_id: str) -> None:
        """Remove the mode identified by *mode_id*.

        Raises:
            AgentModeError: If the mode is not found.
        """
        if mode_id not in self._modes:
            raise AgentModeError(f"mode_id '{mode_id}' not found")
        del self._modes[mode_id]

    def get(self, mode_id: str) -> AgentMode:
        """Return the :class:`AgentMode` for *mode_id*.

        Raises:
            AgentModeError: If the mode is not found.
        """
        if mode_id not in self._modes:
            raise AgentModeError(f"mode_id '{mode_id}' not found")
        return self._modes[mode_id]

    def list_modes(self) -> list[AgentMode]:
        """Return all registered modes."""
        return list(self._modes.values())

    def find_by_tool(self, tool_name: str) -> list[AgentMode]:
        """Return modes that allow *tool_name*.

        An empty ``allowed_tools`` list means all tools are permitted.
        """
        return [
            mode
            for mode in self._modes.values()
            if not mode.allowed_tools or tool_name in mode.allowed_tools
        ]

    def default_mode(self) -> AgentMode:
        """Return the ``code`` mode, or the first registered mode."""
        if "code" in self._modes:
            return self._modes["code"]
        if self._modes:
            return next(iter(self._modes.values()))
        raise AgentModeError("no modes registered")

    def switch_context(self, mode_id: str, current_context: dict) -> dict:
        """Merge the mode's ``system_prompt_prefix`` into *current_context*.

        The prefix is prepended to ``current_context["system_prompt"]``,
        separated by a newline.  If ``system_prompt`` is missing it is
        initialised to the prefix.
        """
        mode = self.get(mode_id)
        new_context = dict(current_context)
        existing = new_context.get("system_prompt", "")
        if existing:
            new_context["system_prompt"] = f"{mode.system_prompt_prefix}\n{existing}"
        else:
            new_context["system_prompt"] = mode.system_prompt_prefix
        return new_context

    def is_tool_allowed(self, mode_id: str, tool_name: str) -> bool:
        """Check whether *tool_name* is permitted in *mode_id*.

        An empty ``allowed_tools`` list means ALL tools are allowed.
        """
        mode = self.get(mode_id)
        if not mode.allowed_tools:
            return True
        return tool_name in mode.allowed_tools


# ---------------------------------------------------------------------------
# Pre-register five default modes at module import time.
# ---------------------------------------------------------------------------

DEFAULT_MODE_REGISTRY = AgentModeRegistry()

DEFAULT_MODE_REGISTRY.register(
    AgentMode(
        mode_id="code",
        name="Code",
        description="Focus on writing, editing, and refactoring code.",
        system_prompt_prefix=(
            "You are in code mode. Focus on writing, editing, and refactoring code. Be concise."
        ),
        allowed_tools=[],
        response_style="concise",
    )
)

DEFAULT_MODE_REGISTRY.register(
    AgentMode(
        mode_id="architect",
        name="Architect",
        description="Design systems, plan migrations, evaluate trade-offs.",
        system_prompt_prefix=(
            "You are in architect mode. Design systems, plan migrations, evaluate trade-offs. Think step by step."
        ),
        allowed_tools=["read", "write", "search", "analyze"],
        response_style="structured",
    )
)

DEFAULT_MODE_REGISTRY.register(
    AgentMode(
        mode_id="ask",
        name="Ask",
        description="Answer questions, explain concepts, and provide documentation.",
        system_prompt_prefix=(
            "You are in ask mode. Answer questions, explain concepts, and provide documentation. Be thorough."
        ),
        allowed_tools=["read", "search"],
        response_style="verbose",
    )
)

DEFAULT_MODE_REGISTRY.register(
    AgentMode(
        mode_id="debug",
        name="Debug",
        description="Trace issues, add logs, isolate root causes.",
        system_prompt_prefix=(
            "You are in debug mode. Trace issues, add logs, isolate root causes. Be methodical."
        ),
        allowed_tools=["read", "write", "run", "search"],
        response_style="methodical",
    )
)

DEFAULT_MODE_REGISTRY.register(
    AgentMode(
        mode_id="custom",
        name="Custom",
        description="Follow the user's specialized instructions.",
        system_prompt_prefix="You are in custom mode. Follow the user's specialized instructions.",
        allowed_tools=[],
        response_style="adaptive",
    )
)

AGENT_MODE_REGISTRY: dict[str, AgentModeRegistry] = {
    "default": DEFAULT_MODE_REGISTRY,
}
