"""Tests for src/agent/execution_tracer.py"""
from __future__ import annotations

import pytest

from src.agent.execution_tracer import (
    EXECUTION_TRACER_REGISTRY,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionTracer,
)


@pytest.fixture
def tracer():
    return ExecutionTracer(max_sessions=10, max_events_per_session=100)


class TestStartSession:
    def test_returns_id(self, tracer):
        sid = tracer.start_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_session_listed(self, tracer):
        sid = tracer.start_session()
        assert sid in tracer.list_sessions()


class TestLog:
    def test_log_appends_event(self, tracer):
        sid = tracer.start_session()
        event = tracer.log(sid, ExecutionEventType.THINK, {"text": "hello"})
        assert isinstance(event, ExecutionEvent)
        assert event.event_type == ExecutionEventType.THINK
        events = tracer.get_session(sid)
        assert len(events) == 1

    def test_log_multiple_events(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK)
        tracer.log(sid, ExecutionEventType.ACT)
        events = tracer.get_session(sid)
        assert len(events) == 2


class TestStopSession:
    def test_stop_session_no_error(self, tracer):
        sid = tracer.start_session()
        tracer.stop_session(sid)


class TestGetSession:
    def test_returns_events(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK)
        events = tracer.get_session(sid)
        assert len(events) == 1

    def test_keyerror_unknown_session(self, tracer):
        with pytest.raises(KeyError):
            tracer.get_session("nonexistent")


class TestFilterEvents:
    def test_filter_by_type(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK)
        tracer.log(sid, ExecutionEventType.ACT)
        tracer.log(sid, ExecutionEventType.THINK)
        think_events = tracer.filter_events(sid, ExecutionEventType.THINK)
        assert len(think_events) == 2

    def test_filter_no_type_returns_all(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK)
        tracer.log(sid, ExecutionEventType.ACT)
        all_events = tracer.filter_events(sid)
        assert len(all_events) == 2


class TestSessionStats:
    def test_counts_think(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK)
        tracer.log(sid, ExecutionEventType.THINK)
        stats = tracer.session_stats(sid)
        assert stats["n_think"] == 2

    def test_counts_act(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.ACT)
        stats = tracer.session_stats(sid)
        assert stats["n_act"] == 1

    def test_counts_tool_calls(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.TOOL_CALL)
        stats = tracer.session_stats(sid)
        assert stats["n_tool_calls"] == 1

    def test_counts_errors(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.ERROR)
        stats = tracer.session_stats(sid)
        assert stats["n_errors"] == 1


class TestExportJsonl:
    def test_export_jsonl(self, tracer):
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK, {"text": "test"})
        lines = tracer.export_jsonl(sid)
        assert len(lines) == 1
        import json
        obj = json.loads(lines[0])
        assert obj["event_type"] == "think"


class TestListSessions:
    def test_list_sessions(self, tracer):
        tracer.start_session()
        tracer.start_session()
        sessions = tracer.list_sessions()
        assert len(sessions) == 2


class TestPruneSession:
    def test_prune_removes_session(self, tracer):
        sid = tracer.start_session()
        tracer.prune_session(sid)
        assert sid not in tracer.list_sessions()


class TestOverflowError:
    def test_max_events_raises_overflow(self, tracer):
        tracer = ExecutionTracer(max_events_per_session=3)
        sid = tracer.start_session()
        tracer.log(sid, ExecutionEventType.THINK)
        tracer.log(sid, ExecutionEventType.THINK)
        tracer.log(sid, ExecutionEventType.THINK)
        with pytest.raises(OverflowError):
            tracer.log(sid, ExecutionEventType.THINK)


class TestKeyError:
    def test_keyerror_on_unknown_session(self, tracer):
        with pytest.raises(KeyError):
            tracer.get_session("does_not_exist")


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in EXECUTION_TRACER_REGISTRY