"""Tool registry — versioned tools with sandboxing, authorization, and profiles."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Built-in profiles (OpenClaw/Hermes-inspired)
# ---------------------------------------------------------------------------
# Each profile is an allowlist of tool groups.  The agent selects one profile
# at session start; only tools in groups declared by that profile are callable.
# ---------------------------------------------------------------------------

TOOL_PROFILES: dict[str, list[str]] = {
    "full": [
        "runtime",
        "fs",
        "web",
        "sessions",
        "memory",
        "ui",
        "automation",
        "messaging",
        "nodes",
        "agents",
        "media",
    ],
    "coding": [
        "runtime",
        "fs",
        "web",
        "sessions",
        "memory",
        "automation",
        "agents",
    ],
    "messaging": [
        "messaging",
        "sessions",
        "memory",
    ],
    "minimal": [
        "sessions",
    ],
}

TOOL_GROUP_REGISTRY: dict[str, str] = {}
"""Maps tool-id → group-name.  Populated by :meth:`ToolRegistry.set_group`."""


@dataclass
class ToolSpec:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_auth: bool = False
    sandbox_level: str = "read"  # read, write, admin
    enabled: bool = True


class ToolRegistry:
    """Registry for tools with versioning, authorization, sandboxing, and profiles."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable] = {}
        self._execution_history: list[dict[str, Any]] = []
        self._profile: str = "full"

    @property
    def active_profile(self) -> str:
        return self._profile

    def set_profile(self, profile: str) -> None:
        if profile not in TOOL_PROFILES:
            raise ValueError(
                f"Unknown tool profile {profile!r}; available: {', '.join(sorted(TOOL_PROFILES))}"
            )
        self._profile = profile

    def register(self, spec: ToolSpec, handler: Callable | None = None) -> None:
        self._tools[spec.id] = spec
        if handler:
            self._handlers[spec.id] = handler

    def set_group(self, tool_id: str, group: str) -> None:
        TOOL_GROUP_REGISTRY[tool_id] = group

    def callable_ids(self) -> list[str]:
        """Return tool ids allowed by the currently active profile."""
        allowed_groups = set(TOOL_PROFILES.get(self._profile, []))
        if not allowed_groups:
            return []
        return [
            tid
            for tid, spec in self._tools.items()
            if spec.enabled
            and (
                tid not in TOOL_GROUP_REGISTRY or TOOL_GROUP_REGISTRY.get(tid, "") in allowed_groups
            )
        ]

    def get(self, tool_id: str) -> ToolSpec | None:
        return self._tools.get(tool_id)

    def find_by_name(self, name: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if name.lower() in t.name.lower()]

    def execute(
        self,
        tool_id: str,
        *,
        authenticated: bool = False,
        requested_level: str = "read",
        **kwargs,
    ) -> dict[str, Any]:
        spec = self._tools.get(tool_id)
        if not spec or not spec.enabled:
            return {"error": f"Tool {tool_id} not found or disabled"}
        if tool_id not in self.callable_ids():
            return {"error": f"Tool {tool_id} not allowed by profile {self._profile!r}"}
        if spec.requires_auth and not authenticated:
            return {"error": f"Tool {tool_id} requires authentication"}
        if not self.sandbox_check(tool_id, requested_level):
            return {"error": f"Tool {tool_id} denied for sandbox level {requested_level}"}
        handler = self._handlers.get(tool_id)
        if not handler:
            return {"error": f"No handler for tool {tool_id}"}
        result = handler(**kwargs)
        self._execution_history.append(
            {
                "tool_id": tool_id,
                "input": kwargs,
                "output": result,
                "authenticated": authenticated,
                "requested_level": requested_level,
            }
        )
        return result

    def history(self, n: int = 10) -> list[dict[str, Any]]:
        return self._execution_history[-n:]

    def sandbox_check(self, tool_id: str, requested_level: str) -> bool:
        spec = self._tools.get(tool_id)
        if not spec:
            return False
        levels = {"read": 0, "write": 1, "admin": 2}
        requested = levels.get(requested_level.lower())
        tool_level = levels.get(spec.sandbox_level.lower())
        if requested is None or tool_level is None:
            return False
        return requested <= tool_level
