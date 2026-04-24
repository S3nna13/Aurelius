"""Distributed tracing collector: spans, traces, Jaeger-compatible export."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Import the monitoring registry lazily to avoid circular imports.
# The registry is populated at module level below.


class SpanKind(str, Enum):
    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


@dataclass
class Span:
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    name: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


class TracingCollector:
    """Collects and manages distributed trace spans."""

    def __init__(self) -> None:
        # Maps trace_id -> list of completed Span
        self._completed: dict[str, list[Span]] = defaultdict(list)

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Span:
        """Create and return a new in-flight span."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        return Span(
            trace_id=trace_id,
            name=name,
            kind=kind,
            start_time=time.monotonic(),
        )

    def end_span(self, span: Span) -> None:
        """Set end_time on a span and move it to completed storage."""
        span.end_time = time.monotonic()
        self._completed[span.trace_id].append(span)

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: Optional[dict] = None,
    ) -> None:
        """Attach a timestamped event to a span."""
        event: dict = {"name": name, "timestamp": time.monotonic()}
        if attributes:
            event["attributes"] = attributes
        span.events.append(event)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Return all completed spans belonging to *trace_id*."""
        return list(self._completed.get(trace_id, []))

    def export_jaeger_format(self, trace_id: str) -> dict:
        """Return a minimal Jaeger-compatible dict for the given trace."""
        spans = self.get_trace(trace_id)
        jaeger_spans = []
        for span in spans:
            duration_us = (
                int((span.end_time - span.start_time) * 1_000_000)
                if span.end_time is not None
                else 0
            )
            jaeger_spans.append(
                {
                    "traceID": trace_id,
                    "spanID": span.span_id,
                    "operationName": span.name,
                    "startTime": int(span.start_time * 1_000_000),
                    "duration": duration_us,
                    "tags": [
                        {"key": k, "value": v}
                        for k, v in span.attributes.items()
                    ],
                    "logs": span.events,
                    "spanKind": span.kind.value,
                }
            )
        return {"traceID": trace_id, "spans": jaeger_spans}


# Module-level singleton – also registered in MONITORING_REGISTRY below.
_TRACING_COLLECTOR = TracingCollector()

# Lazily extend the existing MONITORING_REGISTRY dict from __init__ so that
# each sub-module can register itself without importing the others.
try:
    from src.monitoring import MONITORING_REGISTRY as _REG  # type: ignore[import]
    _REG["tracing"] = _TRACING_COLLECTOR
except Exception:
    pass

MONITORING_REGISTRY: dict[str, object] = {"tracing": _TRACING_COLLECTOR}
