"""Tests for AgentTelemetry."""

from __future__ import annotations

import threading

import pytest

from src.observability.agent_telemetry import AgentTelemetry, TelemetryResult
from src.observability.event_bus import Event
from src.observability.trace_context import TraceContext


class TestTelemetryResult:
    def test_fields(self) -> None:
        result = TelemetryResult(
            success=True,
            duration_ms=1.5,
            trace_id="t1",
            metadata={"key": "value"},
        )
        assert result.success is True
        assert result.duration_ms == 1.5
        assert result.trace_id == "t1"
        assert result.metadata == {"key": "value"}

    def test_default_metadata(self) -> None:
        result = TelemetryResult(success=True, duration_ms=0.0)
        assert result.metadata == {}


class TestAgentTelemetryInit:
    def test_creates_defaults(self) -> None:
        t = AgentTelemetry()
        assert t.audit is not None
        assert t.events is not None
        assert t.metrics is not None

    def test_accepts_custom_instances(self) -> None:
        from src.observability.audit_logger import AuditLogger
        from src.observability.event_bus import EventBus
        from src.observability.metrics_collector import MetricsCollector

        audit = AuditLogger()
        events = EventBus()
        metrics = MetricsCollector()
        t = AgentTelemetry(audit_logger=audit, event_bus=events, metrics=metrics)
        assert t.audit is audit
        assert t.events is events
        assert t.metrics is metrics


class TestRecordAction:
    def test_returns_audit_entry(self) -> None:
        t = AgentTelemetry()
        entry = t.record_action("alice", "read", "doc/1", "success")
        assert entry.actor == "alice"
        assert entry.action == "read"
        assert entry.resource == "doc/1"
        assert entry.status == "success"

    def test_passes_metadata(self) -> None:
        t = AgentTelemetry()
        entry = t.record_action("alice", "read", "doc/1", "success", metadata={"ip": "127.0.0.1"})
        assert entry.metadata == {"ip": "127.0.0.1"}

    def test_passes_trace_id(self) -> None:
        t = AgentTelemetry()
        entry = t.record_action("alice", "read", "doc/1", "success", trace_id="t1")
        assert entry.trace_id == "t1"

    def test_publishes_event(self) -> None:
        t = AgentTelemetry()
        received: list[Event] = []

        def handler(ev: Event) -> None:
            received.append(ev)

        t.events.subscribe("audit.read", handler)
        t.record_action("alice", "read", "doc/1", "success")
        assert len(received) == 1
        assert received[0].payload["actor"] == "alice"


class TestCount:
    def test_increment_counter(self) -> None:
        t = AgentTelemetry()
        assert t.count("requests") == 1.0
        assert t.count("requests") == 2.0

    def test_increment_by_value(self) -> None:
        t = AgentTelemetry()
        assert t.count("bytes", 1024) == 1024.0


class TestGauge:
    def test_set_gauge(self) -> None:
        t = AgentTelemetry()
        assert t.gauge("temperature", 36.6) == 36.6

    def test_gauge_stores_value(self) -> None:
        t = AgentTelemetry()
        t.gauge("temperature", 36.6)
        assert t.metrics.gauge_value("temperature") == 36.6


class TestRecordHistogram:
    def test_record_histogram(self) -> None:
        t = AgentTelemetry()
        t.record_histogram("latency_ms", 42.5)
        assert 42.5 in t.metrics.histogram_values("latency_ms")


class TestTraceContextManager:
    def test_trace_records_success(self) -> None:
        t = AgentTelemetry()
        with t.trace("my_op", actor="bob", resource="res/1"):
            pass
        entries = t.audit.get_entries(action="my_op")
        assert len(entries) == 1
        assert entries[0].status == "success"
        assert entries[0].actor == "bob"
        assert entries[0].resource == "res/1"

    def test_trace_records_failure(self) -> None:
        t = AgentTelemetry()
        with pytest.raises(RuntimeError):
            with t.trace("failing_op"):
                raise RuntimeError("boom")
        entries = t.audit.get_entries(action="failing_op")
        assert len(entries) == 1
        assert entries[0].status == "failure"
        assert entries[0].metadata.get("error_type") == "Exception"

    def test_trace_records_duration(self) -> None:
        t = AgentTelemetry()
        with t.trace("timed_op"):
            pass
        entries = t.audit.get_entries(action="timed_op")
        assert entries[0].metadata["duration_ms"] >= 0

    def test_trace_yields_context(self) -> None:
        t = AgentTelemetry()
        with t.trace("op") as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id is not None

    def test_trace_sets_current_context(self) -> None:
        t = AgentTelemetry()
        TraceContext.clear_current()
        with t.trace("op") as ctx:
            assert TraceContext.current() is ctx
        assert TraceContext.current() is None

    def test_trace_restores_previous_context(self) -> None:
        t = AgentTelemetry()
        TraceContext.clear_current()
        previous = TraceContext.new()
        TraceContext.set_current(previous)
        with t.trace("op"):
            pass
        assert TraceContext.current() is previous
        TraceContext.clear_current()

    def test_trace_records_metrics(self) -> None:
        t = AgentTelemetry()
        with t.trace("measured_op"):
            pass
        assert t.metrics.counter_value("op.measured_op.success") == 1.0

    def test_trace_publishes_event(self) -> None:
        t = AgentTelemetry()
        received: list[Event] = []

        def handler(ev: Event) -> None:
            received.append(ev)

        t.events.subscribe("op.my_event.success", handler)
        with t.trace("my_event"):
            pass
        assert len(received) == 1
        assert received[0].payload["actor"] == "agent"

    def test_trace_inherits_parent_context(self) -> None:
        t = AgentTelemetry()
        parent = TraceContext.new()
        with t.trace("child_op", parent_ctx=parent) as ctx:
            assert ctx.trace_id == parent.trace_id
            assert ctx.parent_span_id == parent.span_id


class TestTraceResult:
    def test_returns_telemetry_result(self) -> None:
        t = AgentTelemetry()
        result = t.trace_result("marker_op")
        assert isinstance(result, TelemetryResult)
        assert result.success is True
        assert result.duration_ms >= 0

    def test_trace_result_contains_trace_id(self) -> None:
        t = AgentTelemetry()
        result = t.trace_result("marker_op")
        assert result.trace_id is not None

    def test_trace_result_contains_audit_entry(self) -> None:
        t = AgentTelemetry()
        result = t.trace_result("marker_op")
        assert result.audit_entry is not None
        assert result.audit_entry.action == "marker_op"

    def test_trace_result_records_metrics(self) -> None:
        t = AgentTelemetry()
        t.trace_result("counted_op")
        assert t.metrics.counter_value("op.counted_op.success") == 1.0

    def test_trace_result_accepts_metadata(self) -> None:
        t = AgentTelemetry()
        result = t.trace_result("meta_op", metadata={"key": "val"})
        assert result.metadata["key"] == "val"


class TestAgentTelemetryThreading:
    def test_concurrent_trace_calls(self) -> None:
        t = AgentTelemetry()
        errors: list[Exception] = []

        def worker(name: str) -> None:
            try:
                for _ in range(50):
                    with t.trace(f"op_{name}"):
                        pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        assert t.metrics.counter_value("op.op_0.success") == 50.0
