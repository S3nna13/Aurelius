"""Skills registry shim for the aurelius namespace."""

from __future__ import annotations

from typing import Any

from src.agent.skills_registry import SKILLS_REGISTRY, Skill

SKILL_REGISTRY = SKILLS_REGISTRY

ALL_SKILLS: list[Any] = list(SKILLS_REGISTRY.values())

# Legacy compat — populated by agent/__init__.py at runtime
SKILL_REGISTRY: dict[str, Any] = {}
ALL_SKILLS: list[Any] = []
SKILLS_BY_CATEGORY: dict[str, list[Any]] = {}

__all__ = ["Skill", "SKILL_REGISTRY", "ALL_SKILLS", "SKILLS_BY_CATEGORY"]
