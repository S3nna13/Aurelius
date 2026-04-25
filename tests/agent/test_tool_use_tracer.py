"""Tests for tool-use tracer."""
from __future__ import annotations

import json

from src.agent.tool_use_tracer import (
    ToolCall,
    ToolCallStatus,
    ToolTrace,
    ToolUseTracer,
)


def test_start_trace():
    t = ToolUseTracer()
    tid = t.start_trace()
    if not isinstance(tid, str) or len(tid) == 0:
        raise ValueError(f"Expected non-empty trace id, got {tid}")


def test_record_call_success():
    t = ToolUseTracer()
    tid = t.start_trace()
    call = t.record_call(tid, "read_file", success=True)
    if not isinstance(call, ToolCall):
        raise ValueError(f"Expected ToolCall, got {type(call)}")
    if call.status != ToolCallStatus.SUCCESS:
        raise ValueError(f"Expected SUCCESS status, got {call.status}")
    if call.tool_name != "read_file":
        raise ValueError(f"Expected tool_name=read_file, got {call.tool_name}")


def test_record_call_failure():
    t = ToolUseTracer()
    tid = t.start_trace()
    call = t.record_call(tid, "write_file", success=False, error="permission denied")
    if call.status != ToolCallStatus.FAILURE:
        raise ValueError(f"Expected FAILURE status, got {call.status}")
    if call.error != "permission denied":
        raise ValueError(f"Expected error='permission denied', got {call.error}")


def test_start_and_complete():
    t = ToolUseTracer()
    tid = t.start_trace()
    cid = t.start_call(tid, "bash")
    t.complete_call(cid)
    active = t.active_calls
    if any(c.call_id == cid for c in active):
        raise ValueError(f"Call {cid} should not be in active_calls after complete")


def test_start_and_fail():
    t = ToolUseTracer()
    tid = t.start_trace()
    cid = t.start_call(tid, "dangerous")
    t.fail_call(cid, "denied")
    active = t.active_calls
    if any(c.call_id == cid for c in active):
        raise ValueError(f"Call {cid} should not be in active_calls after fail")


def test_complete_unknown():
    t = ToolUseTracer()
    tid = t.start_trace()
    t.start_call(tid, "bash")
    caught = False
    try:
        t.complete_call("unknown_call_id")
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown call_id")


def test_fail_unknown():
    t = ToolUseTracer()
    t.start_trace()
    caught = False
    try:
        t.fail_call("unknown", "error")
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown call")


def test_get_trace():
    t = ToolUseTracer()
    tid = t.start_trace()
    t.record_call(tid, "read", success=True)
    t.record_call(tid, "write", success=False, error="fail")
    trace = t.get_trace(tid)
    if not isinstance(trace, ToolTrace):
        raise ValueError(f"Expected ToolTrace, got {type(trace)}")
    if trace.trace_id != tid:
        raise ValueError(f"Expected trace_id={tid}, got {trace.trace_id}")
    if len(trace.calls) != 2:
        raise ValueError(f"Expected 2 calls, got {len(trace.calls)}")


def test_get_trace_unknown():
    t = ToolUseTracer()
    caught = False
    try:
        t.get_trace("unknown")
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown trace")


def test_export_trace_valid_json():
    t = ToolUseTracer()
    tid = t.start_trace()
    t.record_call(tid, "read", success=True)
    t.record_call(tid, "write", success=False, error="fail")
    output = t.export_trace(tid)
    obj = json.loads(output)
    if "trace_id" not in obj:
        raise ValueError("Expected 'trace_id' in exported JSON")
    if "calls" not in obj:
        raise ValueError("Expected 'calls' in exported JSON")
    if len(obj["calls"]) != 2:
        raise ValueError(f"Expected 2 calls in export, got {len(obj['calls'])}")


def test_export_trace_unknown():
    t = ToolUseTracer()
    caught = False
    try:
        t.export_trace("unknown")
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown trace")


def test_start_call_unknown_trace():
    t = ToolUseTracer()
    caught = False
    try:
        t.start_call("unknown_trace", "tool")
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown trace")


def test_active_calls_property():
    t = ToolUseTracer()
    tid = t.start_trace()
    t.start_call(tid, "long_running")
    active = t.active_calls
    if len(active) != 1:
        raise ValueError(f"Expected 1 active call, got {len(active)}")
    if active[0].tool_name != "long_running":
        raise ValueError(f"Expected tool_name=long_running, got {active[0].tool_name}")