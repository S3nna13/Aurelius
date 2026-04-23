"""Multi-agent orchestrator: task dispatch, result collection, retry logic."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class AgentSpec:
    name: str
    capabilities: list[str] = field(default_factory=list)
    max_concurrent: int = 1
    priority: int = 0
    agent_id: str = field(default_factory=_short_id)


@dataclass
class TaskAssignment:
    agent_id: str
    task: str
    status: str = "pending"
    result: str = ""
    attempts: int = 0
    task_id: str = field(default_factory=_short_id)


class Orchestrator:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self._agents: dict[str, AgentSpec] = {}
        self._tasks: dict[str, TaskAssignment] = {}

    def register_agent(self, spec: AgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    def agents(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def assign(self, task: str, agent_id: str) -> TaskAssignment:
        if agent_id not in self._agents:
            raise ValueError(f"Agent '{agent_id}' is not registered.")
        assignment = TaskAssignment(agent_id=agent_id, task=task)
        self._tasks[assignment.task_id] = assignment
        return assignment

    def complete(self, task_id: str, result: str) -> bool:
        if task_id not in self._tasks:
            return False
        assignment = self._tasks[task_id]
        assignment.status = "done"
        assignment.result = result
        return True

    def fail(self, task_id: str, error: str = "") -> bool:
        if task_id not in self._tasks:
            return False
        assignment = self._tasks[task_id]
        assignment.status = "failed"
        assignment.result = error
        assignment.attempts += 1
        return True

    def retry_eligible(self, assignment: TaskAssignment) -> bool:
        return assignment.status == "failed" and assignment.attempts <= self.max_retries

    def pending_tasks(self) -> list[TaskAssignment]:
        result = []
        for assignment in self._tasks.values():
            if assignment.status == "pending":
                result.append(assignment)
            elif self.retry_eligible(assignment):
                result.append(assignment)
        return result

    def summary(self) -> dict:
        total = len(self._tasks)
        pending = sum(1 for a in self._tasks.values() if a.status == "pending")
        done = sum(1 for a in self._tasks.values() if a.status == "done")
        failed = sum(1 for a in self._tasks.values() if a.status == "failed")
        return {"total": total, "pending": pending, "done": done, "failed": failed}


ORCHESTRATOR = Orchestrator()
