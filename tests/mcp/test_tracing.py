"""Tests for MCP tracing."""
from __future__ import annotations

import pytest

from src.mcp.tracing import MCPTracer


class TestMCPTracer:
    def test_start_and_finish(self):
        tracer = MCPTracer()
        span_id = tracer.start()
        assert span_id

        ctx = tracer.finish(span_id)
        assert ctx is not None
        assert ctx.duration_ms() >= 0.0

    def test_finish_unknown_returns_none(self):
        tracer = MCPTracer()
        assert tracer.finish("nonexistent") is None

    def test_trace_id_propagation(self):
        tracer = MCPTracer()
        span1 = tracer.start(trace_id="trace-abc")
        span2 = tracer.start(trace_id="trace-abc", parent_span=span1)
        ctx2 = tracer.finish(span2)
        assert ctx2 is not None
        assert ctx2.parent_span_id == span1
        assert ctx2.trace_id == "trace-abc"

    def test_headers(self):
        from src.mcp.tracing import MCPTracingContext
        ctx = MCPTracingContext(trace_id="t1", parent_span_id="p1")
        headers = ctx.to_headers()
        assert headers["x-trace-id"] == "t1"
        assert headers["x-parent-span-id"] == "p1"