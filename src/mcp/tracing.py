"""MCP request/response metadata and tracing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class MCPTracingContext:
    """Trace metadata for an MCP request/response roundtrip."""

    trace_id: str = ""
    parent_span_id: str = ""
    span_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = uuid4().hex[:16]
        if not self.span_id:
            self.span_id = uuid4().hex[:12]
        self.start_time = time.monotonic()

    def finish(self) -> None:
        self.end_time = time.monotonic()

    def duration_ms(self) -> float:
        if self.end_time == 0.0:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_headers(self) -> dict[str, str]:
        return {
            "x-trace-id": self.trace_id,
            "x-span-id": self.span_id,
            "x-parent-span-id": self.parent_span_id,
        }


@dataclass
class MCPTracer:
    """Manages MCP tracing contexts."""

    _contexts: dict[str, MCPTracingContext] = field(default_factory=dict, repr=False)

    def start(self, trace_id: str = "", parent_span: str = "") -> str:
        ctx = MCPTracingContext(
            trace_id=trace_id or uuid4().hex[:16],
            parent_span_id=parent_span,
        )
        self._contexts[ctx.span_id] = ctx
        return ctx.span_id

    def finish(self, span_id: str) -> MCPTracingContext | None:
        ctx = self._contexts.get(span_id)
        if ctx:
            ctx.finish()
        return ctx


MCP_TRACER = MCPTracer()