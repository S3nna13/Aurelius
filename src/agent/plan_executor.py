"""Plan executor: sequential and parallel step execution with dependency tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str = ""


class PlanExecutor:
    """Execute a plan composed of interdependent steps."""

    def __init__(self, steps: Optional[list[PlanStep]] = None) -> None:
        self._steps: dict[str, PlanStep] = {}
        for step in (steps or []):
            self._steps[step.id] = step

    def add_step(self, step: PlanStep) -> None:
        self._steps[step.id] = step

    def ready_steps(self) -> list[PlanStep]:
        """Return steps that are PENDING and have all dependencies DONE."""
        ready = []
        for step in self._steps.values():
            if step.status != StepStatus.PENDING:
                continue
            if all(
                self._steps.get(dep_id, PlanStep("_", "_", status=StepStatus.PENDING)).status == StepStatus.DONE
                for dep_id in step.depends_on
            ):
                ready.append(step)
        return ready

    def mark_done(self, step_id: str, result: str = "") -> bool:
        if step_id not in self._steps:
            return False
        self._steps[step_id].status = StepStatus.DONE
        self._steps[step_id].result = result
        return True

    def mark_failed(self, step_id: str, result: str = "") -> bool:
        if step_id not in self._steps:
            return False
        self._steps[step_id].status = StepStatus.FAILED
        self._steps[step_id].result = result
        return True

    def skip_dependents(self, step_id: str) -> list[str]:
        """Mark all transitive dependents of step_id as SKIPPED. Return their ids."""
        skipped: list[str] = []
        # BFS / iterative expansion
        frontier = {step_id}
        visited: set[str] = set()
        while frontier:
            current = frontier.pop()
            visited.add(current)
            for step in self._steps.values():
                if step.id in visited:
                    continue
                if current in step.depends_on:
                    step.status = StepStatus.SKIPPED
                    skipped.append(step.id)
                    frontier.add(step.id)
        return skipped

    def is_complete(self) -> bool:
        """True when every step is DONE, FAILED, or SKIPPED."""
        terminal = {StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal for s in self._steps.values())

    def summary(self) -> dict:
        counts: dict[str, int] = {"total": 0, "done": 0, "failed": 0, "skipped": 0, "pending": 0}
        for step in self._steps.values():
            counts["total"] += 1
            if step.status == StepStatus.DONE:
                counts["done"] += 1
            elif step.status == StepStatus.FAILED:
                counts["failed"] += 1
            elif step.status == StepStatus.SKIPPED:
                counts["skipped"] += 1
            else:
                counts["pending"] += 1
        return counts


PLAN_EXECUTOR_REGISTRY: dict[str, type] = {"default": PlanExecutor}
