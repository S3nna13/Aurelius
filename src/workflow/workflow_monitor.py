from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(Enum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowEvent:
    event_type: EventType
    workflow_id: str
    message: str = ""
    timestamp: float = field(default_factory=time.time)


def create_event(event_type: EventType, workflow_id: str, message: str = "") -> WorkflowEvent:
    return WorkflowEvent(event_type=event_type, workflow_id=workflow_id, message=message)


class WorkflowMonitor:
    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowStatus] = {}
        self._events: dict[str, list[WorkflowEvent]] = {}
        self._names: dict[str, str] = {}

    def start_workflow(self, workflow_id: str, name: str = "") -> None:
        self._workflows[workflow_id] = WorkflowStatus.RUNNING
        self._names[workflow_id] = name
        self._events.setdefault(workflow_id, []).append(create_event(EventType.STARTED, workflow_id, f"Started: {name}"))

    def complete_workflow(self, workflow_id: str, message: str = "") -> None:
        self._workflows[workflow_id] = WorkflowStatus.COMPLETED
        self._events.setdefault(workflow_id, []).append(create_event(EventType.COMPLETED, workflow_id, message))

    def fail_workflow(self, workflow_id: str, message: str = "") -> None:
        self._workflows[workflow_id] = WorkflowStatus.FAILED
        self._events.setdefault(workflow_id, []).append(create_event(EventType.FAILED, workflow_id, message))

    def log_event(self, workflow_id: str, event_type: EventType, message: str = "") -> None:
        self._events.setdefault(workflow_id, []).append(create_event(event_type, workflow_id, message))

    def get_status(self, workflow_id: str) -> WorkflowStatus | None:
        return self._workflows.get(workflow_id)

    def get_events(self, workflow_id: str) -> list[WorkflowEvent]:
        return self._events.get(workflow_id, [])

    def active_count(self) -> int:
        return sum(1 for s in self._workflows.values() if s == WorkflowStatus.RUNNING)

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self._workflows),
            "running": sum(1 for s in self._workflows.values() if s == WorkflowStatus.RUNNING),
            "completed": sum(1 for s in self._workflows.values() if s == WorkflowStatus.COMPLETED),
            "failed": sum(1 for s in self._workflows.values() if s == WorkflowStatus.FAILED),
        }


WORKFLOW_MONITOR = WorkflowMonitor()
