"""Tests for src/agent/tool_use_tracer.py"""
from __future__ import annotations

import json

import pytest

from src.agent.tool_use_tracer import (
    TOOL_USE_TRACER_REGISTRY,
    ToolCall,
    ToolCallStatus,
    ToolTrace,
    ToolUseTracer,
)


@pytest.fixture
def tracer():
    return ToolUseTracer(max_traces=10, max_calls_per_trace=100)


class TestStartTrace:
    def test_returns_trace_id(self, tracer):
        tid = tracer.start_trace()
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_trace_retrievable(self, tracer):
        tid = tracer.start_trace()
        trace = tracer.get_trace(tid)
        assert isinstance(trace, ToolTrace)


class TestRecordCall:
    def test_record_call_success(self, tracer):
        tid = tracer.start_trace()
        call = tracer.record_call(tid, "search", success=True)
        assert isinstance(call, ToolCall)
        assert call.status == ToolCallStatus.SUCCESS
        assert call.tool_name == "search"

    def test_record_call_failure(self, tracer):
        tid = tracer.start_trace()
        call = tracer.record_call(tid, "fetch", success=False, error="timeout")
        assert call.status == ToolCallStatus.FAILURE
        assert call.error == "timeout"


class TestStartCompleteFail:
    def test_start_call(self, tracer):
        tid = tracer.start_trace()
        cid = tracer.start_call(tid, "search")
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_complete_call(self, tracer):
        tid = tracer.start_trace()
        cid = tracer.start_call(tid, "search")
        tracer.complete_call(cid)
        assert cid not in tracer.active_calls

    def test_fail_call(self, tracer):
        tid = tracer.start_trace()
        cid = tracer.start_call(tid, "search")
        tracer.fail_call(cid, "error")
        assert cid not in tracer.active_calls
        trace = tracer.get_trace(tid)
        assert len(trace.calls) == 1
        assert trace.calls[0].error == "error"


class TestGetTrace:
    def test_get_trace_success_rate(self, tracer):
        tid = tracer.start_trace()
        tracer.record_call(tid, "search", success=True)
        tracer.record_call(tid, "search", success=True)
        tracer.record_call(tid, "search", success=False)
        trace = tracer.get_trace(tid)
        assert len(trace.calls) == 3
        successes = sum(1 for c in trace.calls if c.status == ToolCallStatus.SUCCESS)
        assert successes == 2

    def test_get_trace_mean_latency(self, tracer):
        tid = tracer.start_trace()
        tracer.record_call(tid, "search", success=True)
        trace = tracer.get_trace(tid)
        assert len(trace.calls) >= 1


class TestExportTrace:
    def test_export_trace_valid_json(self, tracer):
        tid = tracer.start_trace()
        tracer.record_call(tid, "search", success=True)
        exported = tracer.export_trace(tid)
        obj = json.loads(exported)
        assert "trace_id" in obj
        assert "calls" in obj
        assert len(obj["calls"]) == 1


class TestActiveCalls:
    def test_active_calls(self, tracer):
        tid = tracer.start_trace()
        cid = tracer.start_call(tid, "search")
        active = tracer.active_calls
        assert len(active) == 1
        assert active[0].call_id == cid


class TestKeyErrors:
    def test_keyerror_unknown_trace(self, tracer):
        with pytest.raises(KeyError):
            tracer.get_trace("nonexistent")

    def test_keyerror_unknown_call(self, tracer):
        tracer.start_trace()
        with pytest.raises(KeyError):
            tracer.complete_call("fake_call_id")

    def test_keyerror_start_call_unknown_trace(self, tracer):
        with pytest.raises(KeyError):
            tracer.start_call("unknown_trace", "search")


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in TOOL_USE_TRACER_REGISTRY