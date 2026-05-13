"""Aurelius agent state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AureliusState:
    """Central state container for an Aurelius agent instance."""

    agent_id: str = ""
    model_name: str = ""
    context_length: int = 4096
    max_tool_calls: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)
    running: bool = False


_state: AureliusState | None = None


def get_state() -> AureliusState:
    """Get or create the singleton agent state."""
    global _state
    if _state is None:
        _state = AureliusState()
    return _state
