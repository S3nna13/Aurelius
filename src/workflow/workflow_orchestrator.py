"""Workflow orchestrator with dependency-aware step execution."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable


class WorkflowOrchestrator:
    """Orchestrate workflow steps in dependency order."""

    def __init__(self, max_steps: int = 100) -> None:
        self.max_steps = max_steps
        self._steps: dict[str, Callable] = {}
        self._deps: dict[str, list[str]] = {}
        self._results: dict[str, object] = {}

    def register_step(self, name: str, func: Callable, depends_on: list[str] | None = None) -> None:
        if name in self._steps:
            raise ValueError(f"duplicate step name: {name}")
        if len(self._steps) >= self.max_steps:
            raise ValueError(f"step count exceeds max_steps: {self.max_steps}")
        self._steps[name] = func
        self._deps[name] = list(depends_on or [])

    def get_execution_order(self) -> list[str]:
        self._validate()
        return self._topological_sort()

    def execute(self, inputs: dict | None = None) -> dict[str, object]:
        inputs = inputs or {}
        self._validate()
        order = self._topological_sort()

        self._results = {}
        for name in order:
            func = self._steps[name]
            deps = self._deps[name]
            if not deps:
                if inputs:
                    self._results[name] = func(**inputs)
                else:
                    self._results[name] = func()
            else:
                kwargs = {dep: self._results[dep] for dep in deps}
                self._results[name] = func(**kwargs)

        return dict(self._results)

    def reset(self) -> None:
        self._results = {}

    def _validate(self) -> None:
        for name, deps in self._deps.items():
            for dep in deps:
                if dep not in self._steps:
                    raise ValueError(f"missing dependency: {dep} for step {name}")
        self._topological_sort()

    def _topological_sort(self) -> list[str]:
        indegree: dict[str, int] = {name: 0 for name in self._steps}
        adj: dict[str, list[str]] = {name: [] for name in self._steps}
        for name, deps in self._deps.items():
            for dep in deps:
                adj[dep].append(name)
                indegree[name] += 1

        queue: deque[str] = deque([name for name, d in indegree.items() if d == 0])
        order: list[str] = []
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for nxt in adj[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self._steps):
            raise ValueError("circular dependency detected")
        return order


WORKFLOW_ORCHESTRATOR_REGISTRY: dict[str, type[WorkflowOrchestrator]] = {
    "default": WorkflowOrchestrator,
}
