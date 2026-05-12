"""Agent registry shim for the aurelius namespace.

The top-level agent/ package owns the real registries (AGENT_LOOP_REGISTRY,
TOOL_REGISTRY, etc.). This module re-exports them under the legacy names that
aurelius/__init__.py expects.
"""

from __future__ import annotations

from typing import Any

from agent import AGENT_LOOP_REGISTRY

# Legacy names — maps to the real registries
AGENT_REGISTRY: dict[str, Any] = AGENT_LOOP_REGISTRY

# Agent category constants (legacy compat)
ALL_AGENTS: list[str] = list(AGENT_LOOP_REGISTRY.keys())

RESEARCH_AGENT: str = "research"
DEVOPS_AGENT: str = "devops"
CREATIVE_AGENT: str = "creative"
TUTOR_AGENT: str = "tutor"

AGENTS_BY_CATEGORY: dict[str, list[str]] = {
    "research": [RESEARCH_AGENT],
    "devops": [DEVOPS_AGENT],
    "creative": [CREATIVE_AGENT],
    "tutor": [TUTOR_AGENT],
}

__all__ = [
    "AGENT_REGISTRY",
    "AGENTS_BY_CATEGORY",
    "ALL_AGENTS",
    "CREATIVE_AGENT",
    "DEVOPS_AGENT",
    "RESEARCH_AGENT",
    "TUTOR_AGENT",
]
