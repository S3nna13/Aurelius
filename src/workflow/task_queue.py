"""Task queue for dependency-based task scheduling."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    id: str
    deps: list[str] | None = None
    data: Any = None


@dataclass
class TaskQueue:
    """Queue tasks respecting dependency ordering."""

    _pending: dict[str, Task] = field(default_factory=dict, repr=False)
    _completed: set[str] = field(default_factory=set, repr=False)

    def add(self, task: Task) -> None:
        self._pending[task.id] = task

    def ready(self) -> list[Task]:
        return [t for t in self._pending.values()
                if not t.deps or all(d in self._completed for d in t.deps)]

    def complete(self, task_id: str) -> None:
        self._pending.pop(task_id, None)
        self._completed.add(task_id)

    def pending_count(self) -> int:
        return len(self._pending)

    def clear_completed(self) -> None:
        self._completed.clear()


TASK_QUEUE = TaskQueue()