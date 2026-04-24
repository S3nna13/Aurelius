"""Tests for src/monitoring/tracing_collector.py"""
import time

import pytest

from src.monitoring.tracing_collector import (
    Span,
    SpanKind,
    TracingCollector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_collector() -> TracingCollector:
    return TracingCollector()


# ---------------------------------------------------------------------------
# SpanKind enum
# ---------------------------------------------------------------------------

def test_span_kind_values():
    assert SpanKind.INTERNAL == "INTERNAL"
    assert SpanKind.SERVER == "SERVER"
    assert SpanKind.CLIENT == "CLIENT"
    assert SpanKind.PRODUCER == "PRODUCER"
    assert SpanKind.CONSUMER == "CONSUMER"


# ---------------------------------------------------------------------------
# Span dataclass
# ---------------------------------------------------------------------------

def test_span_defaults():
    s = Span(trace_id="t1", name="op")
    assert s.span_id  # auto-generated uuid
    assert s.end_time is None
    assert s.attributes == {}
    assert s.events == []


def test_span_unique_ids():
    s1 = Span(trace_id="t1", name="a")
    s2 = Span(trace_id="t1", name="b")
    assert s1.span_id != s2.span_id


# ---------------------------------------------------------------------------
# start_span / end_span
# ---------------------------------------------------------------------------

def test_start_span_creates_trace_id():
    tc = make_collector()
    span = tc.start_span("my-op")
    assert span.trace_id  # auto-generated
    assert span.name == "my-op"
    assert span.kind == SpanKind.INTERNAL
    assert span.end_time is None


def test_start_span_reuses_trace_id():
    tc = make_collector()
    span = tc.start_span("op", trace_id="fixed-trace")
    assert span.trace_id == "fixed-trace"


def test_start_span_kind():
    tc = make_collector()
    span = tc.start_span("rpc", kind=SpanKind.CLIENT)
    assert span.kind == SpanKind.CLIENT


def test_end_span_sets_end_time():
    tc = make_collector()
    span = tc.start_span("op", trace_id="tr1")
    tc.end_span(span)
    assert span.end_time is not None
    assert span.end_time >= span.start_time


def test_end_span_stores_in_completed():
    tc = make_collector()
    span = tc.start_span("op", trace_id="tr1")
    tc.end_span(span)
    assert tc.get_trace("tr1") == [span]


# ---------------------------------------------------------------------------
# add_event
# ---------------------------------------------------------------------------

def test_add_event_basic():
    tc = make_collector()
    span = tc.start_span("op")
    tc.add_event(span, "cache-hit")
    assert len(span.events) == 1
    assert span.events[0]["name"] == "cache-hit"


def test_add_event_with_attributes():
    tc = make_collector()
    span = tc.start_span("op")
    tc.add_event(span, "retry", attributes={"attempt": 2})
    assert span.events[0]["attributes"]["attempt"] == 2


def test_add_multiple_events():
    tc = make_collector()
    span = tc.start_span("op")
    tc.add_event(span, "start")
    tc.add_event(span, "end")
    assert len(span.events) == 2


# ---------------------------------------------------------------------------
# get_trace
# ---------------------------------------------------------------------------

def test_get_trace_empty():
    tc = make_collector()
    assert tc.get_trace("nonexistent") == []


def test_get_trace_multiple_spans():
    tc = make_collector()
    s1 = tc.start_span("a", trace_id="t99")
    s2 = tc.start_span("b", trace_id="t99")
    tc.end_span(s1)
    tc.end_span(s2)
    trace = tc.get_trace("t99")
    assert len(trace) == 2
    assert s1 in trace
    assert s2 in trace


def test_get_trace_isolation():
    """Spans from different traces don't bleed into each other."""
    tc = make_collector()
    s1 = tc.start_span("a", trace_id="tA")
    s2 = tc.start_span("b", trace_id="tB")
    tc.end_span(s1)
    tc.end_span(s2)
    assert tc.get_trace("tA") == [s1]
    assert tc.get_trace("tB") == [s2]


# ---------------------------------------------------------------------------
# export_jaeger_format
# ---------------------------------------------------------------------------

def test_export_jaeger_format_structure():
    tc = make_collector()
    span = tc.start_span("query", trace_id="jaeger-trace", kind=SpanKind.SERVER)
    span.attributes["db"] = "postgres"
    tc.end_span(span)
    result = tc.export_jaeger_format("jaeger-trace")
    assert result["traceID"] == "jaeger-trace"
    assert len(result["spans"]) == 1
    js = result["spans"][0]
    assert js["operationName"] == "query"
    assert js["spanKind"] == "SERVER"
    assert isinstance(js["duration"], int)
    assert js["duration"] >= 0


def test_export_jaeger_format_empty_trace():
    tc = make_collector()
    result = tc.export_jaeger_format("ghost")
    assert result["traceID"] == "ghost"
    assert result["spans"] == []
