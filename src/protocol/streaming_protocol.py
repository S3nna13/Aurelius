"""Streaming protocol: SSE event framing, delta accumulation, done signaling."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    DONE = "done"
    ERROR = "error"
    PING = "ping"


def _hex8() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class StreamEvent:
    event_type: StreamEventType
    id: str = field(default_factory=_hex8)
    data: str = ""
    sequence: int = 0


class StreamingProtocol:
    def __init__(self) -> None:
        self._log: list[StreamEvent] = []
        self._sequence: int = 0

    def emit(self, event_type: StreamEventType, data: str = "") -> StreamEvent:
        event = StreamEvent(event_type=event_type, data=data, sequence=self._sequence)
        self._sequence += 1
        self._log.append(event)
        return event

    def to_sse(self, event: StreamEvent) -> str:
        return f"id: {event.id}\nevent: {event.event_type.value}\ndata: {event.data}\n\n"

    def accumulate_text(self, events: list[StreamEvent]) -> str:
        return "".join(
            e.data for e in events if e.event_type == StreamEventType.TEXT_DELTA
        )

    def is_done(self, events: list[StreamEvent]) -> bool:
        return any(e.event_type == StreamEventType.DONE for e in events)

    def event_log(self) -> list[StreamEvent]:
        return list(self._log)

    def reset(self) -> None:
        self._log = []
        self._sequence = 0


STREAMING_PROTOCOL = StreamingProtocol()
