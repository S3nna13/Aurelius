"""Skill management — registry, composition, execution, versioning."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillSpec:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    prompt_template: str = ""
    tools: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    enabled: bool = True


class SkillRegistry:
    """Central registry for skills with versioning and dependencies."""

    def __init__(self):
        self._skills: dict[str, SkillSpec] = {}
        self._compositions: dict[str, list[str]] = {}

    def register(self, skill: SkillSpec) -> None:
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> SkillSpec | None:
        return self._skills.get(skill_id)

    def find_by_name(self, name: str) -> list[SkillSpec]:
        return [s for s in self._skills.values() if name.lower() in s.name.lower()]

    def compose(self, composition_id: str, skill_ids: list[str]) -> None:
        self._compositions[composition_id] = list(skill_ids)

    def get_composition(self, composition_id: str) -> list[str] | None:
        composition = self._compositions.get(composition_id)
        if composition is None:
            return None
        return list(composition)

    def list_compositions(self) -> dict[str, list[str]]:
        return {key: list(skill_ids) for key, skill_ids in self._compositions.items()}

    def resolve_dependencies(self, skill_id: str) -> list[str]:
        resolved: list[str] = []
        visited: set[str] = set()

        def dfs(sid: str) -> None:
            if sid in visited:
                return
            visited.add(sid)
            skill = self._skills.get(sid)
            if skill:
                for dep in skill.dependencies:
                    dfs(dep)
                resolved.append(sid)

        dfs(skill_id)
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "version": skill.version,
                    "description": skill.description,
                    "prompt_template": skill.prompt_template,
                    "tools": list(skill.tools),
                    "dependencies": list(skill.dependencies),
                    "enabled": skill.enabled,
                }
                for skill in self._skills.values()
            ],
            "compositions": self.list_compositions(),
        }

    def update_from_dict(self, data: dict[str, Any]) -> None:
        self._skills.clear()
        self._compositions.clear()

        for raw_skill in data.get("skills", []):
            skill = SkillSpec(
                id=raw_skill.get("id", uuid.uuid4().hex[:12]),
                name=raw_skill.get("name", ""),
                version=raw_skill.get("version", "1.0.0"),
                description=raw_skill.get("description", ""),
                prompt_template=raw_skill.get("prompt_template", ""),
                tools=list(raw_skill.get("tools", [])),
                dependencies=list(raw_skill.get("dependencies", [])),
                enabled=raw_skill.get("enabled", True),
            )
            self.register(skill)

        compositions = data.get("compositions", {})
        if isinstance(compositions, dict):
            for composition_id, skill_ids in compositions.items():
                self.compose(composition_id, list(skill_ids))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillRegistry:
        registry = cls()
        registry.update_from_dict(data)
        return registry

    @property
    def count(self) -> int:
        return len(self._skills)

    def enable(self, skill_id: str) -> None:
        skill = self._skills.get(skill_id)
        if skill:
            skill.enabled = True

    def disable(self, skill_id: str) -> None:
        skill = self._skills.get(skill_id)
        if skill:
            skill.enabled = False
