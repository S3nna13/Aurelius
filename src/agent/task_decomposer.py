"""Task decomposer for the Aurelius agent surface.

Given a high-level task string, this module uses a caller-provided
``generate_fn`` to produce a directed acyclic graph (DAG) of sub-tasks
with dependencies, validates the DAG (cycle detection via topological
sort), and provides run-order traversal + parallelizable-task grouping.

Pure stdlib: ``json`` and ``collections`` only. No silent fallbacks.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


class TaskDecompositionError(ValueError):
    """Raised on malformed generator output or invalid DAGs."""


@dataclass
class SubTask:
    """A single decomposed sub-task with explicit dependencies."""

    id: str
    description: str
    depends_on: list[str]
    estimated_complexity: str = "medium"
    tool_hints: list[str] = field(default_factory=list)


@dataclass
class TaskDAG:
    """A directed acyclic graph of :class:`SubTask` nodes."""

    tasks: list[SubTask]
    roots: list[str]
    leaves: list[str]


_VALID_COMPLEXITY = {"low", "medium", "high"}


class TaskDecomposer:
    """Decompose a high-level task into a validated sub-task DAG.

    ``generate_fn`` is expected to return a JSON array of sub-task dicts
    with keys ``id``, ``description``, ``depends_on`` and optional
    ``estimated_complexity`` / ``tool_hints``.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        max_depth: int = 4,
        max_tasks: int = 32,
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_tasks < 1:
            raise ValueError("max_tasks must be >= 1")
        self.generate_fn = generate_fn
        self.max_depth = max_depth
        self.max_tasks = max_tasks

    # ------------------------------------------------------------------ core

    def decompose(self, task: str) -> TaskDAG:
        """Call ``generate_fn`` and build a validated :class:`TaskDAG`."""
        if not isinstance(task, str) or not task.strip():
            raise TaskDecompositionError("task must be a non-empty string")

        raw = self.generate_fn(task)
        if not isinstance(raw, str):
            raise TaskDecompositionError(
                "generate_fn must return a string (JSON array)"
            )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TaskDecompositionError(
                f"generate_fn returned malformed JSON: {exc}"
            ) from exc

        if not isinstance(parsed, list):
            raise TaskDecompositionError(
                "generate_fn must return a JSON array at the top level"
            )

        tasks: list[SubTask] = []
        for i, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                raise TaskDecompositionError(
                    f"sub-task #{i} is not a JSON object"
                )
            tasks.append(self._coerce_subtask(entry, i))

        dag = self._build_dag(tasks)
        self.validate_dag(dag)
        return dag

    # ---------------------------------------------------------------- helpers

    def _coerce_subtask(self, entry: dict, index: int) -> SubTask:
        missing = [k for k in ("id", "description", "depends_on") if k not in entry]
        if missing:
            raise TaskDecompositionError(
                f"sub-task #{index} missing required keys: {missing}"
            )
        tid = entry["id"]
        desc = entry["description"]
        deps = entry["depends_on"]
        complexity = entry.get("estimated_complexity", "medium")
        hints = entry.get("tool_hints", [])

        if not isinstance(tid, str) or not tid:
            raise TaskDecompositionError(
                f"sub-task #{index} has non-string/empty id"
            )
        if not isinstance(desc, str) or not desc:
            raise TaskDecompositionError(
                f"sub-task {tid!r} has non-string/empty description"
            )
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise TaskDecompositionError(
                f"sub-task {tid!r} has non-list-of-strings depends_on"
            )
        if complexity not in _VALID_COMPLEXITY:
            raise TaskDecompositionError(
                f"sub-task {tid!r} has invalid estimated_complexity {complexity!r}"
            )
        if not isinstance(hints, list) or not all(isinstance(h, str) for h in hints):
            raise TaskDecompositionError(
                f"sub-task {tid!r} has non-list-of-strings tool_hints"
            )

        return SubTask(
            id=tid,
            description=desc,
            depends_on=list(deps),
            estimated_complexity=complexity,
            tool_hints=list(hints),
        )

    def _build_dag(self, tasks: list[SubTask]) -> TaskDAG:
        ids = {t.id for t in tasks}
        roots = [t.id for t in tasks if not t.depends_on]
        has_dependents: set[str] = set()
        for t in tasks:
            for d in t.depends_on:
                has_dependents.add(d)
        leaves = [t.id for t in tasks if t.id not in has_dependents]
        # preserve insertion order in roots/leaves; ids is only used later
        _ = ids
        return TaskDAG(tasks=tasks, roots=roots, leaves=leaves)

    # ------------------------------------------------------------- validation

    def validate_dag(self, dag: TaskDAG) -> None:
        """Raise :class:`TaskDecompositionError` if the DAG is invalid."""
        tasks = dag.tasks
        if len(tasks) > self.max_tasks:
            raise TaskDecompositionError(
                f"DAG has {len(tasks)} tasks, exceeds max_tasks={self.max_tasks}"
            )

        ids: list[str] = [t.id for t in tasks]
        seen: set[str] = set()
        for tid in ids:
            if tid in seen:
                raise TaskDecompositionError(f"duplicate sub-task id: {tid!r}")
            seen.add(tid)

        id_set = set(ids)
        for t in tasks:
            for d in t.depends_on:
                if d == t.id:
                    raise TaskDecompositionError(
                        f"self-loop detected on sub-task {t.id!r}"
                    )
                if d not in id_set:
                    raise TaskDecompositionError(
                        f"sub-task {t.id!r} depends on unknown id {d!r}"
                    )

        # Cycle detection via topological sort (also enforces max_depth).
        order = self._topo_sort_raw(tasks)
        depth = self._max_depth(tasks, order)
        if depth > self.max_depth:
            raise TaskDecompositionError(
                f"DAG depth {depth} exceeds max_depth={self.max_depth}"
            )

    # --------------------------------------------------------- traversal API

    def topological_sort(self, dag: TaskDAG) -> list[str]:
        """Return a valid topological ordering of task ids."""
        return self._topo_sort_raw(dag.tasks)

    def parallelizable_groups(self, dag: TaskDAG) -> list[list[str]]:
        """Return layers of task ids; tasks in each layer may run in parallel."""
        tasks = dag.tasks
        by_id = {t.id: t for t in tasks}
        indegree: dict[str, int] = {t.id: len(t.depends_on) for t in tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in tasks}
        for t in tasks:
            for d in t.depends_on:
                dependents[d].append(t.id)

        # validate first so we never silently loop forever
        remaining = len(tasks)
        groups: list[list[str]] = []
        ready = [t.id for t in tasks if indegree[t.id] == 0]
        while ready:
            # stable order: follow original task order
            layer = sorted(ready, key=lambda tid: ids_index(tasks, tid))
            groups.append(layer)
            next_ready: list[str] = []
            for tid in layer:
                remaining -= 1
                for dep in dependents[tid]:
                    indegree[dep] -= 1
                    if indegree[dep] == 0:
                        next_ready.append(dep)
            ready = next_ready
        if remaining != 0:
            raise TaskDecompositionError("cycle detected in DAG")
        _ = by_id
        return groups

    # ------------------------------------------------------------- internals

    def _topo_sort_raw(self, tasks: list[SubTask]) -> list[str]:
        indegree: dict[str, int] = {t.id: len(t.depends_on) for t in tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in tasks}
        for t in tasks:
            for d in t.depends_on:
                if d not in indegree:
                    raise TaskDecompositionError(
                        f"sub-task {t.id!r} depends on unknown id {d!r}"
                    )
                dependents[d].append(t.id)

        order: list[str] = []
        queue: deque[str] = deque(
            t.id for t in tasks if indegree[t.id] == 0
        )
        while queue:
            tid = queue.popleft()
            order.append(tid)
            for dep in dependents[tid]:
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)
        if len(order) != len(tasks):
            raise TaskDecompositionError("cycle detected in DAG")
        return order

    def _max_depth(self, tasks: list[SubTask], order: list[str]) -> int:
        by_id = {t.id: t for t in tasks}
        depth: dict[str, int] = {}
        for tid in order:
            deps = by_id[tid].depends_on
            depth[tid] = 1 if not deps else 1 + max(depth[d] for d in deps)
        return max(depth.values()) if depth else 0


def ids_index(tasks: list[SubTask], tid: str) -> int:
    """Return the original insertion index of ``tid`` in ``tasks``."""
    for i, t in enumerate(tasks):
        if t.id == tid:
            return i
    return len(tasks)


__all__ = [
    "SubTask",
    "TaskDAG",
    "TaskDecomposer",
    "TaskDecompositionError",
]
