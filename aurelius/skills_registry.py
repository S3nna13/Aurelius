from typing import Any

from src.agent.skills_registry import Skill, SKILLS_REGISTRY

SKILL_REGISTRY = SKILLS_REGISTRY

ALL_SKILLS: list[Any] = list(SKILLS_REGISTRY.values())

SKILLS_BY_CATEGORY: dict[str, list[Any]] = {}
for _skill in ALL_SKILLS:
    SKILLS_BY_CATEGORY.setdefault(_skill.category, []).append(_skill)

__all__ = ["Skill", "SKILLS_REGISTRY", "SKILL_REGISTRY", "ALL_SKILLS", "SKILLS_BY_CATEGORY"]
