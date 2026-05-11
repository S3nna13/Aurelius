"""Skills registry shim for the aurelius namespace."""

from __future__ import annotations

from typing import Any

from agent.skill_library import Skill

# Legacy compat — populated by agent/__init__.py at runtime
SKILL_REGISTRY: dict[str, Any] = {}
ALL_SKILLS: list[Any] = []
SKILLS_BY_CATEGORY: dict[str, list[Any]] = {}

__all__ = ["Skill", "SKILL_REGISTRY", "ALL_SKILLS", "SKILLS_BY_CATEGORY"]
