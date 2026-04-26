"""Task route table: static (TaskType × ComplexityTier) dispatch table for inference backends."""

from __future__ import annotations

from dataclasses import dataclass

from src.inference.request_classifier import ComplexityTier, TaskType


@dataclass
class RouteEntry:
    task_type: TaskType
    complexity: ComplexityTier
    backend: str
    priority: int = 5
    timeout_s: float = 30.0


def _default_routes() -> list[RouteEntry]:
    entries: list[RouteEntry] = []
    for task_type in TaskType:
        entries.append(
            RouteEntry(task_type=task_type, complexity=ComplexityTier.LOW, backend="cached")
        )
        entries.append(
            RouteEntry(task_type=task_type, complexity=ComplexityTier.MEDIUM, backend="local")
        )
        entries.append(
            RouteEntry(task_type=task_type, complexity=ComplexityTier.HIGH, backend="api")
        )
    return entries


class TaskRouteTable:
    def __init__(self, routes: list[RouteEntry] | None = None) -> None:
        self._table: dict[tuple[TaskType, ComplexityTier], RouteEntry] = {}
        for entry in _default_routes():
            self._table[(entry.task_type, entry.complexity)] = entry
        if routes:
            for entry in routes:
                self.register(entry)

    def lookup(self, task_type: TaskType, complexity: ComplexityTier) -> RouteEntry:
        key = (task_type, complexity)
        if key in self._table:
            return self._table[key]
        fallback_key = (TaskType.UNKNOWN, complexity)
        return self._table[fallback_key]

    def register(self, entry: RouteEntry) -> None:
        self._table[(entry.task_type, entry.complexity)] = entry

    def list_routes(self) -> list[RouteEntry]:
        return list(self._table.values())


TASK_ROUTE_TABLE = TaskRouteTable()
