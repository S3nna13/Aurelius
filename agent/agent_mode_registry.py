"""Agent mode registry for Aurelius.

BACKWARD-COMPATIBLE WRAPPER: delegates to UnifiedPersona / UnifiedPersonaRegistry.
Mode data is derived from the built-in agent-mode personas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.persona import (
    AURELIUS_ARCHITECT_MODE,
    AURELIUS_ASK_MODE,
    AURELIUS_CODE_MODE,
    AURELIUS_DEBUG_MODE,
    UnifiedPersonaRegistry,
)

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
    """Backward-compatible wrapper delegating to UnifiedPersonaRegistry."""

    _modes: dict[str, AgentMode] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._unified = UnifiedPersonaRegistry()

    def register(self, mode: AgentMode) -> None:
        if not isinstance(mode.mode_id, str) or not mode.mode_id.strip():
            raise AgentModeError("mode_id must be a non-empty string")
        if mode.mode_id in self._modes:
            raise AgentModeError(f"mode_id '{mode.mode_id}' is already registered")
        self._modes[mode.mode_id] = mode

    def unregister(self, mode_id: str) -> None:
        if mode_id not in self._modes:
            raise AgentModeError(f"mode_id '{mode_id}' not found")
        del self._modes[mode_id]

    def get(self, mode_id: str) -> AgentMode:
        if mode_id not in self._modes:
            raise AgentModeError(f"mode_id '{mode_id}' not found")
        return self._modes[mode_id]

    def list_modes(self) -> list[AgentMode]:
        return list(self._modes.values())

    def find_by_tool(self, tool_name: str) -> list[AgentMode]:
        return [
            mode
            for mode in self._modes.values()
            if not mode.allowed_tools or tool_name in mode.allowed_tools
        ]

    def default_mode(self) -> AgentMode:
        if "code" in self._modes:
            return self._modes["code"]
        if self._modes:
            return next(iter(self._modes.values()))
        raise AgentModeError("no modes registered")

    def switch_context(self, mode_id: str, current_context: dict) -> dict:
        mode = self.get(mode_id)
        new_context = dict(current_context)
        existing = new_context.get("system_prompt", "")
        if existing:
            new_context["system_prompt"] = f"{mode.system_prompt_prefix}\n{existing}"
        else:
            new_context["system_prompt"] = mode.system_prompt_prefix
        return new_context

    def is_tool_allowed(self, mode_id: str, tool_name: str) -> bool:
        mode = self.get(mode_id)
        if not mode.allowed_tools:
            return True
        return tool_name in mode.allowed_tools


_MODE_PERSONAS = {
    "code": AURELIUS_CODE_MODE,
    "architect": AURELIUS_ARCHITECT_MODE,
    "ask": AURELIUS_ASK_MODE,
    "debug": AURELIUS_DEBUG_MODE,
}

DEFAULT_MODE_REGISTRY = AgentModeRegistry()

for mode_id, persona in _MODE_PERSONAS.items():
    DEFAULT_MODE_REGISTRY.register(
        AgentMode(
            mode_id=mode_id,
            name=persona.name.replace("Aurelius-", ""),
            description=persona.description,
            system_prompt_prefix=persona.system_prompt,
            allowed_tools=list(persona.allowed_tools),
            response_style=persona.response_style.value,
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
