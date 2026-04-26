from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum


class EventType(StrEnum):
    WORKFLOW_STARTED = "workflow_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    CHECKPOINT = "checkpoint"


@dataclass
class WorkflowEvent:
    event_id: str
    workflow_id: str
    event_type: EventType
    step_id: str | None
    payload: dict
    timestamp: float


class WorkflowEventStore:
    """Append-only in-memory event store for workflow event sourcing."""

    def __init__(self) -> None:
        self._events: dict[str, list[WorkflowEvent]] = {}

    def append(
        self,
        workflow_id: str,
        event_type: EventType,
        step_id: str | None = None,
        payload: dict | None = None,
    ) -> WorkflowEvent:
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            event_type=event_type,
            step_id=step_id,
            payload=payload or {},
            timestamp=time.time(),
        )
        self._events.setdefault(workflow_id, []).append(event)
        return event

    def get_history(self, workflow_id: str) -> list[WorkflowEvent]:
        return list(self._events.get(workflow_id, []))

    def replay(self, workflow_id: str) -> dict:
        state: dict = {
            "status": "unknown",
            "completed_steps": [],
            "failed_steps": [],
            "current_step": None,
        }
        for event in self._events.get(workflow_id, []):
            et = event.event_type
            if et == EventType.WORKFLOW_STARTED:
                state["status"] = "running"
            elif et == EventType.STEP_STARTED:
                state["current_step"] = event.step_id
            elif et == EventType.STEP_COMPLETED:
                if event.step_id not in state["completed_steps"]:
                    state["completed_steps"].append(event.step_id)
                state["current_step"] = None
            elif et == EventType.STEP_FAILED:
                if event.step_id not in state["failed_steps"]:
                    state["failed_steps"].append(event.step_id)
                state["current_step"] = None
            elif et == EventType.WORKFLOW_COMPLETED:
                state["status"] = "completed"
                state["current_step"] = None
            elif et == EventType.WORKFLOW_FAILED:
                state["status"] = "failed"
                state["current_step"] = None
        return state

    def list_workflows(self) -> list[str]:
        return list(self._events.keys())

    def prune(self, workflow_id: str) -> int:
        events = self._events.pop(workflow_id, [])
        return len(events)

    def export_json(self, workflow_id: str) -> str:
        history = self.get_history(workflow_id)
        return json.dumps(
            [
                {
                    "event_id": e.event_id,
                    "workflow_id": e.workflow_id,
                    "event_type": e.event_type.value,
                    "step_id": e.step_id,
                    "payload": e.payload,
                    "timestamp": e.timestamp,
                }
                for e in history
            ]
        )


WORKFLOW_EVENT_STORE = WorkflowEventStore()
