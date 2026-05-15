from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum


class ExecutionEventType(StrEnum):
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CHECKPOINT = "checkpoint"


@dataclass
class ExecutionEvent:
    event_id: str
    session_id: str
    event_type: ExecutionEventType
    step: int
    content: dict
    timestamp: float
    duration_ms: float | None = None


class ExecutionTracer:
    """Full agent execution tracer: think/act/observe cycle logging."""

    def __init__(self, max_events_per_session: int = 10_000) -> None:
        self._max = max_events_per_session
        self._sessions: dict[str, list[ExecutionEvent]] = {}
        self._steps: dict[str, int] = {}
        self._agent_ids: dict[str, str] = {}

    def start_session(self, agent_id: str) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        self._steps[session_id] = 0
        self._agent_ids[session_id] = agent_id
        self.log(session_id, ExecutionEventType.AGENT_START, {"agent_id": agent_id})
        return session_id

    def log(
        self,
        session_id: str,
        event_type: ExecutionEventType,
        content: dict,
        duration_ms: float | None = None,
    ) -> ExecutionEvent:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        events = self._sessions[session_id]
        if len(events) >= self._max:
            raise OverflowError(f"Session {session_id} reached max events ({self._max})")
        step = self._steps[session_id]
        self._steps[session_id] = step + 1
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            step=step,
            content=content,
            timestamp=time.time(),
            duration_ms=duration_ms,
        )
        events.append(event)
        return event

    def stop_session(self, session_id: str) -> None:
        self.log(session_id, ExecutionEventType.AGENT_STOP, {})

    def get_session(self, session_id: str) -> list[ExecutionEvent]:
        return list(self._sessions.get(session_id, []))

    def filter_events(
        self, session_id: str, event_type: ExecutionEventType
    ) -> list[ExecutionEvent]:
        return [e for e in self._sessions.get(session_id, []) if e.event_type == event_type]

    def session_stats(self, session_id: str) -> dict:
        events = self._sessions.get(session_id, [])
        durations = [e.duration_ms for e in events if e.duration_ms is not None]
        return {
            "n_events": len(events),
            "n_think": sum(1 for e in events if e.event_type == ExecutionEventType.THINK),
            "n_act": sum(1 for e in events if e.event_type == ExecutionEventType.ACT),
            "n_tool_calls": sum(1 for e in events if e.event_type == ExecutionEventType.TOOL_CALL),
            "n_errors": sum(1 for e in events if e.event_type == ExecutionEventType.ERROR),
            "total_duration_ms": sum(durations),
            "step_count": self._steps.get(session_id, 0),
        }

    def export_jsonl(self, session_id: str) -> str:
        lines = []
        for event in self._sessions.get(session_id, []):
            lines.append(
                json.dumps(
                    {
                        "event_id": event.event_id,
                        "session_id": event.session_id,
                        "event_type": event.event_type.value,
                        "step": event.step,
                        "content": event.content,
                        "timestamp": event.timestamp,
                        "duration_ms": event.duration_ms,
                    }
                )
            )
        return "\n".join(lines)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def prune_session(self, session_id: str) -> int:
        events = self._sessions.pop(session_id, [])
        self._steps.pop(session_id, None)
        self._agent_ids.pop(session_id, None)
        return len(events)
