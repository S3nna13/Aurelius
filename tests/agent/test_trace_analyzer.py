"""Tests for src/agent/trace_analyzer.py."""

from __future__ import annotations

from src.agent.execution_tracer import ExecutionEvent, ExecutionEventType
from src.agent.trace_analyzer import TraceAnalyzer


def _evt(
    event_type: ExecutionEventType,
    step: int,
    content: dict | None = None,
    duration_ms: float | None = None,
    event_id: str = "e1",
) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        session_id="s1",
        event_type=event_type,
        step=step,
        content=content or {},
        timestamp=0.0,
        duration_ms=duration_ms,
    )


def test_empty_trace():
    ta = TraceAnalyzer()
    result = ta.analyze([])
    assert result.total_events == 0
    assert result.total_steps == 0
    assert result.error_count == 0
    assert result.recommendations == []


def test_counts_errors_and_tools():
    ta = TraceAnalyzer()
    events = [
        _evt(ExecutionEventType.AGENT_START, 0),
        _evt(ExecutionEventType.THINK, 1, duration_ms=10.0),
        _evt(
            ExecutionEventType.TOOL_CALL, 2, {"tool_name": "read"}, duration_ms=50.0, event_id="tc1"
        ),
        _evt(ExecutionEventType.TOOL_RESULT, 3, {"tool_call_id": "tc1"}, event_id="tr1"),
        _evt(
            ExecutionEventType.TOOL_CALL, 4, {"tool_name": "read"}, duration_ms=60.0, event_id="tc2"
        ),
        _evt(
            ExecutionEventType.TOOL_RESULT,
            5,
            {"tool_call_id": "tc2", "error": True},
            event_id="tr2",
        ),
        _evt(ExecutionEventType.ERROR, 6, {"message": "oops"}),
        _evt(ExecutionEventType.AGENT_STOP, 7),
    ]
    result = ta.analyze(events)
    assert result.total_events == 8
    assert result.error_count == 1
    assert len(result.tool_summaries) == 1
    ts = result.tool_summaries[0]
    assert ts.name == "read"
    assert ts.calls == 2
    assert ts.errors == 1


def test_detects_rapid_retry_loop():
    ta = TraceAnalyzer()
    events = [
        _evt(ExecutionEventType.TOOL_CALL, 0, {"tool_name": "fetch"}),
        _evt(ExecutionEventType.TOOL_RESULT, 1, {"tool_call_id": "e1", "error": True}),
        _evt(ExecutionEventType.TOOL_CALL, 2, {"tool_name": "fetch"}),
        _evt(ExecutionEventType.TOOL_RESULT, 3, {"tool_call_id": "e2", "error": True}),
        _evt(ExecutionEventType.TOOL_CALL, 4, {"tool_name": "fetch"}),
        _evt(ExecutionEventType.TOOL_RESULT, 5, {"tool_call_id": "e3", "error": True}),
    ]
    result = ta.analyze(events)
    assert any("rapid_retry_loop" in p for p in result.failure_patterns)
    assert any("backoff" in r.lower() for r in result.recommendations)


def test_detects_tool_error_chain():
    ta = TraceAnalyzer()
    events = [
        _evt(ExecutionEventType.TOOL_CALL, 0, {"tool_name": "run"}),
        _evt(ExecutionEventType.ERROR, 1, {"message": "crash"}),
    ]
    result = ta.analyze(events)
    assert any("tool_error_chain" in p for p in result.failure_patterns)


def test_detects_think_heavy():
    ta = TraceAnalyzer()
    events = [_evt(ExecutionEventType.THINK, i) for i in range(10)] + [
        _evt(ExecutionEventType.ACT, 10),
    ]
    result = ta.analyze(events)
    assert any("think_heavy" in p for p in result.failure_patterns)
    assert any("thinking steps" in r.lower() for r in result.recommendations)


def test_slowest_event_reported():
    ta = TraceAnalyzer(slow_threshold_ms=5.0)
    events = [
        _evt(ExecutionEventType.TOOL_CALL, 0, duration_ms=1.0),
        _evt(ExecutionEventType.TOOL_CALL, 1, duration_ms=100.0),
    ]
    result = ta.analyze(events)
    assert result.slowest_event is not None
    assert result.slowest_event["duration_ms"] == 100.0
    assert any("100.0 ms" in r for r in result.recommendations)


def test_unique_errors_normalizes():
    ta = TraceAnalyzer()
    events = [
        _evt(ExecutionEventType.ERROR, 0, {"message": "file /tmp/123.txt not found"}),
        _evt(ExecutionEventType.ERROR, 1, {"message": "file /tmp/456.txt not found"}),
    ]
    result = ta.analyze(events)
    assert len(result.unique_errors) == 1
    assert "<PATH>" in result.unique_errors[0]
