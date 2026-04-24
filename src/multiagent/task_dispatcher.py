"""Dispatches tasks to available agents."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class DispatchStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class DispatchedTask:
    task_id: str
    description: str
    assigned_to: str = ""
    status: DispatchStatus = DispatchStatus.PENDING
    result: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.monotonic()


class TaskDispatcher:
    def __init__(self, agents: list[str]) -> None:
        self._agents: set[str] = set(agents)
        self._tasks: dict[str, DispatchedTask] = {}

    def submit(self, description: str) -> DispatchedTask:
        task_id = uuid.uuid4().hex[:8]
        task = DispatchedTask(task_id=task_id, description=description)
        self._tasks[task_id] = task
        return task

    def assign(self, task_id: str, agent_id: str) -> bool:
        if task_id not in self._tasks:
            return False
        if agent_id not in self._agents:
            return False
        task = self._tasks[task_id]
        task.assigned_to = agent_id
        task.status = DispatchStatus.ASSIGNED
        return True

    def complete(self, task_id: str, result: str = "") -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        task.status = DispatchStatus.COMPLETED
        task.result = result
        return True

    def fail(self, task_id: str, reason: str = "") -> bool:
        if task_id not in self._tasks:
            return False
        task = self._tasks[task_id]
        task.status = DispatchStatus.FAILED
        task.result = reason
        return True

    def pending_tasks(self) -> list[DispatchedTask]:
        return [t for t in self._tasks.values() if t.status == DispatchStatus.PENDING]

    def agent_load(self) -> dict[str, int]:
        counts: dict[str, int] = {agent_id: 0 for agent_id in self._agents}
        for task in self._tasks.values():
            if task.status == DispatchStatus.ASSIGNED and task.assigned_to in counts:
                counts[task.assigned_to] += 1
        return counts

    def add_agent(self, agent_id: str) -> None:
        self._agents.add(agent_id)

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            self._agents.discard(agent_id)
            return True
        return False


TASK_DISPATCHER_REGISTRY: dict[str, type] = {"default": TaskDispatcher}
