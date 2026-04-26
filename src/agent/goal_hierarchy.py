"""
Hierarchical goal decomposition.

Nested goal tree with status tracking. Extends the flat step-list model of
``autonomous_loop.py`` to support arbitrary-depth parent/child relationships.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GoalStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Goal:
    description: str
    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 0
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalHierarchy:
    """In-memory tree of ``Goal`` objects indexed by ``goal_id``."""

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}

    # --------------------------------------------------------------- mutation
    def add_goal(
        self,
        description: str,
        parent_id: str | None = None,
        priority: int = 0,
    ) -> Goal:
        if parent_id is not None and parent_id not in self._goals:
            raise KeyError(f"unknown parent goal_id: {parent_id}")
        goal = Goal(description=description, parent_id=parent_id, priority=priority)
        self._goals[goal.goal_id] = goal
        if parent_id is not None:
            self._goals[parent_id].children.append(goal.goal_id)
        return goal

    def set_status(self, goal_id: str, status: GoalStatus) -> None:
        if goal_id not in self._goals:
            raise KeyError(f"unknown goal_id: {goal_id}")
        self._goals[goal_id].status = status

    # ---------------------------------------------------------------- queries
    def get_goal(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def children_of(self, goal_id: str) -> list[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return []
        return [self._goals[cid] for cid in goal.children if cid in self._goals]

    def root_goals(self) -> list[Goal]:
        return [g for g in self._goals.values() if g.parent_id is None]

    def leaf_goals(self) -> list[Goal]:
        return [g for g in self._goals.values() if not g.children]

    def active_path(self) -> list[Goal]:
        actives = [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

        def depth(g: Goal) -> int:
            d = 0
            cur: Goal | None = g
            while cur is not None and cur.parent_id is not None:
                cur = self._goals.get(cur.parent_id)
                d += 1
            return d

        actives.sort(key=lambda g: -depth(g))
        return actives

    def completion_ratio(self) -> float:
        if not self._goals:
            return 0.0
        completed = sum(1 for g in self._goals.values() if g.status == GoalStatus.COMPLETED)
        return completed / len(self._goals)

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        def serialize(goal: Goal) -> dict[str, Any]:
            return {
                "goal_id": goal.goal_id,
                "description": goal.description,
                "parent_id": goal.parent_id,
                "status": goal.status.value,
                "priority": goal.priority,
                "metadata": dict(goal.metadata),
                "children": [serialize(self._goals[c]) for c in goal.children if c in self._goals],
            }

        return {
            "roots": [serialize(g) for g in self.root_goals()],
            "count": len(self._goals),
            "completion_ratio": self.completion_ratio(),
        }


GOAL_HIERARCHY_REGISTRY = {"default": GoalHierarchy}
