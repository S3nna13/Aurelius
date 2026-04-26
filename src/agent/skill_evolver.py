"""Self-evolving skill tree inspired by GenericAgent.

Crystallizes execution traces into reusable skills that accumulate over time,
forming a hierarchical skill tree.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.memory.layered_memory import LayeredMemory


class SkillEvolutionError(Exception):
    """Raised for invalid skill evolution operations."""


@dataclass
class CrystallizedSkill:
    """A reusable skill extracted from a completed task."""

    skill_id: str
    name: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    last_used: datetime.datetime | None = None
    parent_skill_id: str | None = None


class SkillEvolver:
    """Crystallize execution traces into reusable skills."""

    def __init__(self, memory: LayeredMemory | None = None) -> None:
        self._skills: dict[str, CrystallizedSkill] = {}
        self._memory = memory

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def crystallize(self, task_record: dict[str, Any]) -> CrystallizedSkill:
        """Convert a task record into a reusable skill."""
        name = str(task_record.get("task_name", "unnamed_task"))
        raw_steps = task_record.get("steps", [])
        parameters = dict(task_record.get("parameters", {}))
        context = task_record.get("context", {})

        # Extract successful steps
        steps: list[str] = []
        for step in raw_steps:
            if isinstance(step, dict) and step.get("success") is True:
                action = str(step.get("action", ""))
                if action:
                    steps.append(action)

        # Detect parameter templates in steps
        templated_steps = list(steps)
        for param_key, param_value in parameters.items():
            placeholder = f"{{{param_key}}}"
            for i, step in enumerate(templated_steps):
                if str(param_value) in step:
                    templated_steps[i] = step.replace(str(param_value), placeholder)

        # Preconditions from required context keys
        preconditions = sorted(context.keys()) if isinstance(context, dict) else []

        # Link to parent skill if applicable
        parent_skill_id: str | None = None
        parent_task = task_record.get("parent_task")
        if parent_task:
            for sid, skill in self._skills.items():
                if skill.name == parent_task:
                    parent_skill_id = sid
                    break

        skill = CrystallizedSkill(
            skill_id=uuid.uuid4().hex[:8],
            name=name,
            description=task_record.get("description", f"Skill derived from {name}"),
            preconditions=preconditions,
            steps=templated_steps,
            parameters=parameters,
            parent_skill_id=parent_skill_id,
        )
        self._skills[skill.skill_id] = skill

        if self._memory is not None:
            self._memory.store(
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                    "preconditions": skill.preconditions,
                    "steps": skill.steps,
                    "parameters": skill.parameters,
                    "parent_skill_id": skill.parent_skill_id,
                },
                layer_name="L3 Task Skills",
            )

        return skill

    def match(self, query: str) -> list[CrystallizedSkill]:
        """Find skills matching *query* by name, description, or keywords."""
        q = query.lower()
        results: list[CrystallizedSkill] = []
        for skill in self._skills.values():
            if (
                q in skill.name.lower()
                or q in skill.description.lower()
                or any(q in s.lower() for s in skill.steps)
                or any(q in p.lower() for p in skill.preconditions)
            ):
                results.append(skill)
        return results

    def refine(self, skill_id: str, feedback: dict[str, Any]) -> CrystallizedSkill:
        """Update a skill based on feedback."""
        if skill_id not in self._skills:
            raise SkillEvolutionError(f"Skill {skill_id!r} not found.")
        skill = self._skills[skill_id]

        if "add_preconditions" in feedback:
            for p in feedback["add_preconditions"]:
                if p not in skill.preconditions:
                    skill.preconditions.append(p)
            skill.preconditions.sort()

        if "replace_steps" in feedback:
            skill.steps = list(feedback["replace_steps"])

        if "add_steps" in feedback:
            for s in feedback["add_steps"]:
                if s not in skill.steps:
                    skill.steps.append(s)

        if "update_parameters" in feedback:
            skill.parameters.update(feedback["update_parameters"])

        if "increment_success" in feedback:
            skill.success_count += int(feedback["increment_success"])

        if "increment_failure" in feedback:
            skill.failure_count += int(feedback["increment_failure"])

        skill.last_used = datetime.datetime.now(datetime.UTC)
        return skill

    def get_skill_tree(self, root_id: str | None = None) -> dict[str, Any]:
        """Return hierarchical tree of skills."""

        def _build_node(skill_id: str) -> dict[str, Any]:
            skill = self._skills[skill_id]
            children = [
                _build_node(cid)
                for cid, cskill in self._skills.items()
                if cskill.parent_skill_id == skill_id
            ]
            return {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "children": children,
            }

        if root_id is not None:
            if root_id not in self._skills:
                raise SkillEvolutionError(f"Skill {root_id!r} not found.")
            return _build_node(root_id)

        # Return forest: all roots (skills with no parent)
        roots = [sid for sid, s in self._skills.items() if s.parent_skill_id is None]
        return {"forest": [_build_node(rid) for rid in roots]}

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the skill tree."""
        total = len(self._skills)
        if total == 0:
            return {
                "total_skills": 0,
                "avg_success_rate": 0.0,
                "deepest_tree_depth": 0,
            }

        total_attempts = sum(
            s.success_count + s.failure_count for s in self._skills.values()
        )
        avg_success = (
            sum(s.success_count for s in self._skills.values()) / total_attempts
            if total_attempts
            else 0.0
        )

        def _depth(skill_id: str) -> int:
            skill = self._skills[skill_id]
            if skill.parent_skill_id is None:
                return 1
            return 1 + _depth(skill.parent_skill_id)

        deepest = max((_depth(sid) for sid in self._skills), default=0)

        return {
            "total_skills": total,
            "avg_success_rate": avg_success,
            "deepest_tree_depth": deepest,
        }

    def forget(self, skill_id: str) -> bool:
        """Remove a skill and orphan its children."""
        if skill_id not in self._skills:
            return False
        del self._skills[skill_id]
        # Orphan children
        for skill in self._skills.values():
            if skill.parent_skill_id == skill_id:
                skill.parent_skill_id = None
        return True

    def list_skills(self) -> list[CrystallizedSkill]:
        """Return all crystallized skills."""
        return list(self._skills.values())


# ---------------------------------------------------------------------------
# Singleton & registry
# ---------------------------------------------------------------------------

DEFAULT_SKILL_EVOLVER = SkillEvolver()

SKILL_EVOLVER_REGISTRY: dict[str, SkillEvolver] = {
    "default": DEFAULT_SKILL_EVOLVER,
}
