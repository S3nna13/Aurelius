"""Execution tracing for agent reasoning events (think/act/tool/error)."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ExecutionEventType(StrEnum):
    THINK = "think"
    ACT = "act"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    OBSERVE = "observe"


@dataclass
class ExecutionEvent:
    session_id: str
    event_type: ExecutionEventType
    timestamp: float
    payload: dict = field(default_factory=dict)
    event_id: str | None = None

    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())[:8]


@dataclass
class ExecutionTracer:
    max_sessions: int = 100
    max_events_per_session: int = 10000
    _sessions: dict[str, list[ExecutionEvent]] = field(default_factory=dict)

    def start_session(self) -> str:
        sid = str(uuid.uuid4())[:8]
        self._sessions[sid] = []
        if len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions)
            del self._sessions[oldest]
        return sid

    def stop_session(self, session_id: str) -> None:
        pass

    def get_session(self, session_id: str) -> list[ExecutionEvent]:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id!r} not found")
        return self._sessions[session_id]

    def log(
        self,
        session_id: str,
        event_type: ExecutionEventType,
        payload: dict | None = None,
    ) -> ExecutionEvent:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id!r} not found")
        events = self._sessions[session_id]
        if len(events) >= self.max_events_per_session:
            raise OverflowError(
                f"Max events ({self.max_events_per_session}) "
                f"reached for session {session_id!r}"
            )
        event = ExecutionEvent(
            session_id=session_id,
            event_type=event_type,
            timestamp=time.time(),
            payload=payload or {},
        )
        events.append(event)
        return event

    def filter_events(
        self,
        session_id: str,
        event_type: ExecutionEventType | None = None,
    ) -> list[ExecutionEvent]:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id!r} not found")
        events = self._sessions[session_id]
        if event_type is None:
            return list(events)
        return [e for e in events if e.event_type == event_type]

    def session_stats(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id!r} not found")
        events = self._sessions[session_id]
        return {
            "n_think": sum(1 for e in events if e.event_type == ExecutionEventType.THINK),
            "n_act": sum(1 for e in events if e.event_type == ExecutionEventType.ACT),
            "n_tool_calls": sum(1 for e in events if e.event_type == ExecutionEventType.TOOL_CALL),
            "n_errors": sum(1 for e in events if e.event_type == ExecutionEventType.ERROR),
            "n_total": len(events),
        }

    def export_jsonl(self, session_id: str) -> list[str]:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id!r} not found")
        lines = []
        for e in self._sessions[session_id]:
            lines.append(json.dumps({
                "event_type": e.event_type.value,
                "timestamp": e.timestamp,
                "payload": e.payload,
            }))
        return lines

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def prune_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]


EXECUTION_TRACER_REGISTRY: dict[str, object] = {"default": ExecutionTracer()}