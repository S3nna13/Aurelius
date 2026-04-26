"""
task_decomposer.py
Decomposes high-level computer-use tasks into atomic steps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtomicStep:
    step_id: int
    description: str
    action_type: str
    target: str = ""
    estimated_ms: int = 100


@dataclass(frozen=True)
class DecomposedTask:
    task_description: str
    steps: list[AtomicStep]
    total_estimated_ms: int


class TaskDecomposer:
    def __init__(self) -> None:
        # Rule-based decomposition map: keyword -> step factory
        self._rules: list[tuple[list[str], list[dict]]] = [
            (
                ["open", "launch"],
                [
                    {
                        "description": "Navigate to application",
                        "action_type": "NAVIGATE",
                        "target": "app",
                        "estimated_ms": 300,
                    },
                    {
                        "description": "Take screenshot of result",
                        "action_type": "SCREENSHOT",
                        "target": "",
                        "estimated_ms": 100,
                    },
                ],
            ),
            (
                ["type", "enter"],
                [
                    {
                        "description": "Click on target field",
                        "action_type": "CLICK",
                        "target": "field",
                        "estimated_ms": 100,
                    },
                    {
                        "description": "Type the specified text",
                        "action_type": "TYPE",
                        "target": "field",
                        "estimated_ms": 200,
                    },
                    {
                        "description": "Take screenshot of result",
                        "action_type": "SCREENSHOT",
                        "target": "",
                        "estimated_ms": 100,
                    },
                ],
            ),
            (
                ["click"],
                [
                    {
                        "description": "Take screenshot to locate target",
                        "action_type": "SCREENSHOT",
                        "target": "",
                        "estimated_ms": 100,
                    },
                    {
                        "description": "Click the specified target",
                        "action_type": "CLICK",
                        "target": "target",
                        "estimated_ms": 100,
                    },
                ],
            ),
            (
                ["scroll"],
                [
                    {
                        "description": "Scroll in the specified direction",
                        "action_type": "SCROLL",
                        "target": "direction",
                        "estimated_ms": 200,
                    },
                    {
                        "description": "Take screenshot of result",
                        "action_type": "SCREENSHOT",
                        "target": "",
                        "estimated_ms": 100,
                    },
                ],
            ),
        ]
        self._default_steps: list[dict] = [
            {
                "description": "Take screenshot to assess state",
                "action_type": "SCREENSHOT",
                "target": "",
                "estimated_ms": 100,
            },
            {
                "description": "Wait for task completion",
                "action_type": "WAIT",
                "target": "",
                "estimated_ms": 500,
            },
        ]

    def decompose(self, task: str) -> DecomposedTask:
        task_lower = task.lower()
        matched_steps: list[dict] | None = None

        for keywords, step_templates in self._rules:
            if any(kw in task_lower for kw in keywords):
                matched_steps = step_templates
                break

        if matched_steps is None:
            matched_steps = self._default_steps

        atomic_steps: list[AtomicStep] = [
            AtomicStep(
                step_id=idx,
                description=tmpl["description"],
                action_type=tmpl["action_type"],
                target=tmpl.get("target", ""),
                estimated_ms=tmpl.get("estimated_ms", 100),
            )
            for idx, tmpl in enumerate(matched_steps)
        ]

        total_estimated_ms = sum(s.estimated_ms for s in atomic_steps)

        return DecomposedTask(
            task_description=task,
            steps=atomic_steps,
            total_estimated_ms=total_estimated_ms,
        )

    def to_actions(self, task: DecomposedTask) -> list[dict]:
        return [
            {
                "step_id": step.step_id,
                "action_type": step.action_type,
                "target": step.target,
            }
            for step in task.steps
        ]


TASK_DECOMPOSER_REGISTRY: dict = {"default": TaskDecomposer}
