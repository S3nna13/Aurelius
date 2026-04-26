from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceEvent:
    name: str
    start_s: float
    end_s: float
    category: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000.0


class TraceCollector:
    def __init__(self):
        self._events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        start_s: float,
        end_s: float,
        category: str = "",
        **metadata,
    ) -> TraceEvent:
        evt = TraceEvent(
            name=name,
            start_s=start_s,
            end_s=end_s,
            category=category,
            metadata=dict(metadata),
        )
        self._events.append(evt)
        return evt

    @contextmanager
    def trace(self, name: str, category: str = ""):
        start = time.perf_counter()
        container: dict = {}
        try:
            yield container
        finally:
            end = time.perf_counter()
            evt = self.record(name, start, end, category=category)
            container["event"] = evt

    def events(self, category: str | None = None) -> list[TraceEvent]:
        if category is None:
            return list(self._events)
        return [e for e in self._events if e.category == category]

    def to_chrome_trace(self, events: list[TraceEvent] | None = None) -> str:
        evs = events if events is not None else self._events
        out = []
        for e in evs:
            ts_us = e.start_s * 1_000_000.0
            dur_us = (e.end_s - e.start_s) * 1_000_000.0
            out.append(
                {
                    "name": e.name,
                    "ph": "X",
                    "ts": ts_us,
                    "dur": dur_us,
                    "cat": e.category,
                    "args": dict(e.metadata),
                }
            )
        return json.dumps(out)

    def summary(self, events: list[TraceEvent] | None = None) -> dict:
        evs = events if events is not None else self._events
        total_duration_ms = sum(e.duration_ms for e in evs)
        by_category: dict[str, dict] = {}
        for e in evs:
            bucket = by_category.setdefault(e.category, {"total_ms": 0.0, "count": 0})
            bucket["total_ms"] += e.duration_ms
            bucket["count"] += 1
        return {
            "total_events": len(evs),
            "total_duration_ms": total_duration_ms,
            "by_category": by_category,
        }

    def clear(self) -> None:
        self._events.clear()


TRACE_COLLECTOR_REGISTRY: dict[str, type[TraceCollector]] = {
    "default": TraceCollector,
}
