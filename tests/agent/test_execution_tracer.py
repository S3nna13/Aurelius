"""Tests for execution tracer."""
from __future__ import annotations

import json

from src.agent.execution_tracer import (
    EXECUTION_TRACER_REGISTRY,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionTracer,
)


def test_start_session():
    t = ExecutionTracer()
    sid = t.start_session()
    if not isinstance(sid, str) or len(sid) == 0:
        raise ValueError(f"Expected non-empty session id, got {sid}")
    sessions = t.list_sessions()
    if sid not in sessions:
        raise ValueError(f"Session {sid} not in list_sessions()")


def test_log_appends_event():
    t = ExecutionTracer()
    sid = t.start_session()
    ev = t.log(sid, ExecutionEventType.THINK, {"text": "hello"})
    if not isinstance(ev, ExecutionEvent):
        raise ValueError(f"Expected ExecutionEvent, got {type(ev)}")
    if ev.event_type != ExecutionEventType.THINK:
        raise ValueError(f"Expected THINK event, got {ev.event_type}")
    events = t.get_session(sid)
    if len(events) != 1:
        raise ValueError(f"Expected 1 event, got {len(events)}")


def test_log_unknown_session():
    t = ExecutionTracer()
    caught = False
    try:
        t.log("unknown", ExecutionEventType.THINK)
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown session")


def test_filter_events():
    t = ExecutionTracer()
    sid = t.start_session()
    t.log(sid, ExecutionEventType.THINK)
    t.log(sid, ExecutionEventType.ACT)
    t.log(sid, ExecutionEventType.THINK)
    thinks = t.filter_events(sid, ExecutionEventType.THINK)
    if len(thinks) != 2:
        raise ValueError(f"Expected 2 THINK events, got {len(thinks)}")
    all_events = t.filter_events(sid, None)
    if len(all_events) != 3:
        raise ValueError(f"Expected 3 total events, got {len(all_events)}")


def test_session_stats():
    t = ExecutionTracer()
    sid = t.start_session()
    t.log(sid, ExecutionEventType.THINK)
    t.log(sid, ExecutionEventType.ACT)
    t.log(sid, ExecutionEventType.TOOL_CALL)
    t.log(sid, ExecutionEventType.ERROR)
    stats = t.session_stats(sid)
    if stats["n_think"] != 1:
        raise ValueError(f"Expected n_think=1, got {stats['n_think']}")
    if stats["n_act"] != 1:
        raise ValueError(f"Expected n_act=1, got {stats['n_act']}")
    if stats["n_tool_calls"] != 1:
        raise ValueError(f"Expected n_tool_calls=1, got {stats['n_tool_calls']}")
    if stats["n_errors"] != 1:
        raise ValueError(f"Expected n_errors=1, got {stats['n_errors']}")
    if stats["n_total"] != 4:
        raise ValueError(f"Expected n_total=4, got {stats['n_total']}")


def test_export_jsonl():
    t = ExecutionTracer()
    sid = t.start_session()
    t.log(sid, ExecutionEventType.THINK, {"text": "test"})
    lines = t.export_jsonl(sid)
    if len(lines) != 1:
        raise ValueError(f"Expected 1 JSON line, got {len(lines)}")
    obj = json.loads(lines[0])
    if obj["event_type"] != "think":
        raise ValueError(f"Expected event_type=think, got {obj['event_type']}")


def test_export_jsonl_unknown():
    t = ExecutionTracer()
    caught = False
    try:
        t.export_jsonl("unknown")
    except KeyError:
        caught = True
    if not caught:
        raise ValueError("Expected KeyError for unknown session")


def test_prune_session():
    t = ExecutionTracer()
    sid = t.start_session()
    t.log(sid, ExecutionEventType.THINK)
    t.prune_session(sid)
    if sid in t.list_sessions():
        raise ValueError(f"Session {sid} should be pruned")


def test_prune_unknown():
    t = ExecutionTracer()
    t.prune_session("unknown")


def test_overflow_max_events():
    t = ExecutionTracer(max_events_per_session=2)
    sid = t.start_session()
    t.log(sid, ExecutionEventType.THINK)
    t.log(sid, ExecutionEventType.ACT)
    caught = False
    try:
        t.log(sid, ExecutionEventType.ERROR)
    except OverflowError:
        caught = True
    if not caught:
        raise ValueError("Expected OverflowError at max events")


def test_registry():
    if "default" not in EXECUTION_TRACER_REGISTRY:
        raise ValueError("default not in EXECUTION_TRACER_REGISTRY")
    inst = EXECUTION_TRACER_REGISTRY["default"]
    if not isinstance(inst, ExecutionTracer):
        raise ValueError(f"Expected ExecutionTracer instance, got {type(inst)}")