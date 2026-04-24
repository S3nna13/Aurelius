"""
Plan / Act / Observe / Reflect loop.

Pure-Python, no-async, callable-driven adaptation of the reference
``run_autonomous_loop``. Callers inject three callables: ``planner`` (produces
numbered step text from a task), ``actor`` (executes one step, returns a string
observation), and ``reflector`` (summarizes the run).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class LoopStep:
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    observation: str = ""
    error: str = ""


@dataclass(frozen=True)
class LoopResult:
    task: str
    steps: List[LoopStep]
    reflection: str
    succeeded: bool
    total_steps: int


def _parse_steps(plan_text: str, max_steps: int) -> List[str]:
    """Extract ``"1. ..."`` or ``"1) ..."`` lines; fall back to non-empty lines.

    Direct port of the reference ``_parse_numbered_steps``.
    """
    lines: List[str] = []
    for raw in plan_text.splitlines():
        m = re.match(r"^\s*\d+[\.)]\s*(.+)$", raw.strip())
        if m:
            lines.append(m.group(1).strip())
        elif raw.strip() and not lines:
            continue
    if not lines:
        for raw in plan_text.splitlines():
            t = raw.strip()
            if t and not t.startswith("#"):
                lines.append(t)
    if max_steps is not None and max_steps >= 0:
        lines = lines[:max_steps]
    return lines


class PlanObserveReflect:
    """Callable-driven PLAN -> ACT -> OBSERVE -> REFLECT loop."""

    def plan(self, task: str, planner: Callable[[str], str]) -> List[str]:
        plan_text = planner(task)
        steps = _parse_steps(plan_text or "", max_steps=999)
        if not steps:
            steps = [task]
        return steps

    def act(self, step: str, actor: Callable[[str], str]) -> str:
        obs = actor(step)
        return obs if obs is not None else ""

    def reflect(
        self,
        task: str,
        steps: List[LoopStep],
        reflector: Callable[[str], str],
    ) -> str:
        summary_lines = [f"Task: {task}"]
        for s in steps:
            summary_lines.append(
                f"Step {s.index} [{s.status.value}]: {s.description} -> {s.observation or s.error}"
            )
        summary = "\n".join(summary_lines)
        out = reflector(summary)
        return out if out is not None else ""

    def run(
        self,
        task: str,
        planner: Callable[[str], str],
        actor: Callable[[str], str],
        reflector: Callable[[str], str],
        max_steps: int = 5,
    ) -> LoopResult:
        plan_text = planner(task) or ""
        step_descriptions = _parse_steps(plan_text, max_steps)
        if not step_descriptions:
            step_descriptions = [task]
        step_descriptions = step_descriptions[:max_steps]

        executed: List[LoopStep] = []
        for i, desc in enumerate(step_descriptions, start=1):
            try:
                observation = actor(desc)
                executed.append(
                    LoopStep(
                        index=i,
                        description=desc,
                        status=StepStatus.COMPLETED,
                        observation=observation if observation is not None else "",
                        error="",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - we want to capture any failure
                executed.append(
                    LoopStep(
                        index=i,
                        description=desc,
                        status=StepStatus.FAILED,
                        observation="",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        reflection = self.reflect(task, executed, reflector)
        succeeded = all(s.status == StepStatus.COMPLETED for s in executed)
        return LoopResult(
            task=task,
            steps=list(executed),
            reflection=reflection,
            succeeded=succeeded,
            total_steps=len(executed),
        )


PLAN_OBSERVE_REFLECT_REGISTRY = {"default": PlanObserveReflect}
