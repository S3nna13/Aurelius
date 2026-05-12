"""Legacy Aurelius skills registry backed by the canonical registry snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from registry_snapshot import (
    SKILL_CATALOG,
)
from registry_snapshot import (
    SKILL_CATEGORIES as SNAPSHOT_SKILL_CATEGORIES,
)


@dataclass(frozen=True)
class SkillRecord:
    id: str
    name: str
    description: str
    category: str
    agent_types: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "agent_types": list(self.agent_types),
            "tags": list(self.tags),
        }


Skill = SkillRecord


def _build_skill(record: dict[str, Any]) -> SkillRecord:
    return SkillRecord(
        id=str(record["id"]),
        name=str(record["name"]),
        description=str(record["description"]),
        category=str(record["category"]),
        agent_types=list(record.get("agent_types", [])),
        tags=list(record.get("tags", [])),
    )


SKILL_CATEGORIES = list(SNAPSHOT_SKILL_CATEGORIES)

ALL_SKILLS: list[SkillRecord] = [_build_skill(skill) for skill in SKILL_CATALOG]
SKILL_REGISTRY: dict[str, SkillRecord] = {skill.id: skill for skill in ALL_SKILLS}
SKILLS_BY_CATEGORY: dict[str, list[SkillRecord]] = {}
for skill in ALL_SKILLS:
    SKILLS_BY_CATEGORY.setdefault(skill.category, []).append(skill)


def skill_to_dict(skill: SkillRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(skill, SkillRecord):
        return skill.to_dict()
    return {
        "id": str(skill["id"]),
        "name": str(skill["name"]),
        "description": str(skill["description"]),
        "category": str(skill["category"]),
        "agent_types": list(skill.get("agent_types", [])),
        "tags": list(skill.get("tags", [])),
    }


__all__ = [
    "Skill",
    "SkillRecord",
    "SKILL_REGISTRY",
    "SKILL_CATEGORIES",
    "ALL_SKILLS",
    "SKILLS_BY_CATEGORY",
    "skill_to_dict",
]
