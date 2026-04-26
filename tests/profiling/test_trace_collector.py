import json
import time

from src.profiling.trace_collector import (
    TRACE_COLLECTOR_REGISTRY,
    TraceCollector,
    TraceEvent,
)


def test_trace_event_duration_ms():
    e = TraceEvent("x", 1.0, 1.5)
    assert abs(e.duration_ms - 500.0) < 1e-6


def test_trace_event_frozen():
    e = TraceEvent("x", 0.0, 1.0)
    try:
        e.name = "y"  # type: ignore
    except Exception:
        return
    assert False, "expected frozen"


def test_trace_event_default_category():
    e = TraceEvent("x", 0.0, 1.0)
    assert e.category == ""


def test_trace_event_default_metadata():
    e = TraceEvent("x", 0.0, 1.0)
    assert e.metadata == {}


def test_record_appends_event():
    c = TraceCollector()
    c.record("op", 0.0, 1.0)
    assert len(c.events()) == 1


def test_record_returns_trace_event():
    c = TraceCollector()
    e = c.record("op", 0.0, 1.0, category="io", foo="bar")
    assert isinstance(e, TraceEvent)
    assert e.category == "io"
    assert e.metadata == {"foo": "bar"}


def test_trace_contextmanager_records():
    c = TraceCollector()
    with c.trace("block"):
        time.sleep(0.001)
    evs = c.events()
    assert len(evs) == 1
    assert evs[0].name == "block"
    assert evs[0].duration_ms > 0


def test_trace_contextmanager_category():
    c = TraceCollector()
    with c.trace("block", category="compute"):
        pass
    assert c.events()[0].category == "compute"


def test_trace_contextmanager_yields_container():
    c = TraceCollector()
    with c.trace("b") as ctx:
        assert isinstance(ctx, dict)
    # After exit the event is populated.
    assert "event" in ctx
    assert ctx["event"].name == "b"


def test_events_filter_by_category():
    c = TraceCollector()
    c.record("a", 0.0, 1.0, category="io")
    c.record("b", 0.0, 1.0, category="compute")
    c.record("c", 0.0, 1.0, category="io")
    out = c.events("io")
    assert len(out) == 2
    assert all(e.category == "io" for e in out)


def test_events_no_filter_returns_all():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    c.record("b", 0.0, 1.0)
    assert len(c.events()) == 2


def test_events_filter_no_match():
    c = TraceCollector()
    c.record("a", 0.0, 1.0, category="io")
    assert c.events("nope") == []


def test_events_returns_copy():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    ev_list = c.events()
    ev_list.clear()
    assert len(c.events()) == 1


def test_to_chrome_trace_valid_json():
    c = TraceCollector()
    c.record("a", 0.0, 0.001)
    out = c.to_chrome_trace()
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


def test_to_chrome_trace_fields():
    c = TraceCollector()
    c.record("op", 1.0, 1.5, category="cat", k="v")
    parsed = json.loads(c.to_chrome_trace())
    entry = parsed[0]
    assert entry["name"] == "op"
    assert entry["ph"] == "X"
    assert entry["cat"] == "cat"
    assert entry["args"] == {"k": "v"}


def test_to_chrome_trace_microseconds():
    c = TraceCollector()
    c.record("op", 1.0, 1.5)
    parsed = json.loads(c.to_chrome_trace())
    entry = parsed[0]
    assert abs(entry["ts"] - 1_000_000.0) < 1e-3
    assert abs(entry["dur"] - 500_000.0) < 1e-3


def test_to_chrome_trace_explicit_events():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    custom = [TraceEvent("only", 0.0, 0.1)]
    parsed = json.loads(c.to_chrome_trace(custom))
    assert len(parsed) == 1
    assert parsed[0]["name"] == "only"


def test_to_chrome_trace_empty():
    c = TraceCollector()
    parsed = json.loads(c.to_chrome_trace())
    assert parsed == []


def test_summary_counts_events():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    c.record("b", 0.0, 2.0)
    s = c.summary()
    assert s["total_events"] == 2


def test_summary_total_duration_ms():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)  # 1000ms
    c.record("b", 0.0, 0.5)  # 500ms
    s = c.summary()
    assert abs(s["total_duration_ms"] - 1500.0) < 1e-6


def test_summary_by_category():
    c = TraceCollector()
    c.record("a", 0.0, 1.0, category="io")
    c.record("b", 0.0, 0.5, category="io")
    c.record("c", 0.0, 1.0, category="compute")
    s = c.summary()
    cats = s["by_category"]
    assert cats["io"]["count"] == 2
    assert abs(cats["io"]["total_ms"] - 1500.0) < 1e-6
    assert cats["compute"]["count"] == 1


def test_summary_empty():
    c = TraceCollector()
    s = c.summary()
    assert s["total_events"] == 0
    assert s["total_duration_ms"] == 0.0
    assert s["by_category"] == {}


def test_summary_explicit_events():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    s = c.summary([TraceEvent("other", 0.0, 2.0)])
    assert s["total_events"] == 1
    assert abs(s["total_duration_ms"] - 2000.0) < 1e-6


def test_clear_empties_events():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    c.record("b", 0.0, 1.0)
    c.clear()
    assert c.events() == []


def test_clear_allows_new_records():
    c = TraceCollector()
    c.record("a", 0.0, 1.0)
    c.clear()
    c.record("b", 0.0, 1.0)
    assert len(c.events()) == 1


def test_registry_key():
    assert "default" in TRACE_COLLECTOR_REGISTRY
    assert TRACE_COLLECTOR_REGISTRY["default"] is TraceCollector


def test_trace_contextmanager_exception_still_records():
    c = TraceCollector()
    try:
        with c.trace("boom"):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert len(c.events()) == 1
    assert c.events()[0].name == "boom"
