"""
task_planner.py – Hierarchical task planner that breaks goals into executable subtasks.
Stdlib-only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubTask:
    task_id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""


@dataclass
class TaskPlan:
    goal: str
    subtasks: list[SubTask]
    created_at: float


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TaskPlanner:
    """Rule-based hierarchical task planner."""

    def __init__(self, max_depth: int = 5, max_subtasks: int = 20) -> None:
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks

    # --- keyword → subtask-name mapping -----------------------------------
    _KEYWORD_STEPS: dict[tuple[str, ...], list[str]] = {
        ("research", "find"):        ["fetch_sources", "summarize", "validate"],
        ("write", "draft"):          ["outline", "draft", "review", "finalize"],
        ("fix", "debug"):            ["reproduce", "diagnose", "patch", "verify"],
        ("analyze", "evaluate"):     ["collect_data", "process", "interpret", "report"],
    }
    _DEFAULT_STEPS: list[str] = ["plan", "execute", "verify"]

    def _steps_for(self, goal: str) -> list[str]:
        lower = goal.lower()
        for keywords, steps in self._KEYWORD_STEPS.items():
            if any(kw in lower for kw in keywords):
                return steps
        return self._DEFAULT_STEPS

    def plan(self, goal: str, context: Optional[dict] = None) -> TaskPlan:
        steps = self._steps_for(goal)[: self.max_subtasks]
        subtasks: list[SubTask] = []
        for idx, name in enumerate(steps):
            task_id = f"t{idx + 1}"
            depends_on = [f"t{idx}"] if idx > 0 else []
            subtasks.append(SubTask(task_id=task_id, description=name, depends_on=depends_on))
        return TaskPlan(goal=goal, subtasks=subtasks, created_at=time.monotonic())

    # --- plan mutation helpers --------------------------------------------

    def mark_complete(self, plan: TaskPlan, task_id: str, result: str = "") -> None:
        for st in plan.subtasks:
            if st.task_id == task_id:
                st.status = "completed"
                st.result = result
                return

    def next_ready(self, plan: TaskPlan) -> list[SubTask]:
        completed_ids = {st.task_id for st in plan.subtasks if st.status == "completed"}
        return [
            st for st in plan.subtasks
            if st.status == "pending" and all(dep in completed_ids for dep in st.depends_on)
        ]

    def is_done(self, plan: TaskPlan) -> bool:
        return all(st.status == "completed" for st in plan.subtasks)

    def progress(self, plan: TaskPlan) -> dict:
        total = len(plan.subtasks)
        completed = sum(1 for st in plan.subtasks if st.status == "completed")
        pct = (completed / total * 100.0) if total else 0.0
        return {"total": total, "completed": completed, "pct": pct}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_PLANNER_REGISTRY: dict[str, type] = {"default": TaskPlanner}
