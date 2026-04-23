"""Tests for src.deployment.otel_instrumentation.

Production deployment patterns from Aurelius production_readiness_floor,
SLSA supply-chain attestation spec, Apache-2.0.
"""

from __future__ import annotations

import pytest

from src.deployment.otel_instrumentation import (
    DEFAULT_METRICS,
    DEFAULT_TRACER,
    OTEL_TRACER_REGISTRY,
    PrometheusMetrics,
    Span,
    SpanKind,
    Tracer,
)


# ---------------------------------------------------------------------------
# Tracer — start_span
# ---------------------------------------------------------------------------


def test_start_span_returns_span() -> None:
    t = Tracer()
    span = t.start_span("test-op")
    assert isinstance(span, Span)


def test_start_span_non_empty_span_id() -> None:
    t = Tracer()
    span = t.start_span("test-op")
    assert span.span_id != ""


def test_start_span_non_empty_trace_id() -> None:
    t = Tracer()
    span = t.start_span("test-op")
    assert span.trace_id != ""


def test_start_span_name_preserved() -> None:
    t = Tracer()
    span = t.start_span("my-operation")
    assert span.name == "my-operation"


def test_start_span_default_kind_internal() -> None:
    t = Tracer()
    span = t.start_span("op")
    assert span.kind == SpanKind.INTERNAL


def test_start_span_custom_kind() -> None:
    t = Tracer()
    span = t.start_span("op", kind=SpanKind.SERVER)
    assert span.kind == SpanKind.SERVER


def test_start_span_attributes_stored() -> None:
    t = Tracer()
    span = t.start_span("op", attributes={"http.method": "GET"})
    assert span.attributes["http.method"] == "GET"


# ---------------------------------------------------------------------------
# Tracer — end_span
# ---------------------------------------------------------------------------


def test_end_span_sets_end_ns() -> None:
    t = Tracer()
    span = t.start_span("op")
    assert span.end_ns is None
    t.end_span(span)
    assert span.end_ns is not None


def test_end_span_end_ns_greater_than_start_ns() -> None:
    t = Tracer()
    span = t.start_span("op")
    t.end_span(span)
    assert span.end_ns >= span.start_ns  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Tracer — get_spans
# ---------------------------------------------------------------------------


def test_get_spans_returns_list() -> None:
    t = Tracer()
    assert isinstance(t.get_spans(), list)


def test_get_spans_grows_with_each_span() -> None:
    t = Tracer()
    assert len(t.get_spans()) == 0
    t.start_span("a")
    assert len(t.get_spans()) == 1
    t.start_span("b")
    assert len(t.get_spans()) == 2


def test_get_spans_returns_copy() -> None:
    t = Tracer()
    t.start_span("a")
    spans = t.get_spans()
    spans.clear()
    assert len(t.get_spans()) == 1


# ---------------------------------------------------------------------------
# Tracer — context_manager
# ---------------------------------------------------------------------------


def test_context_manager_yields_span() -> None:
    t = Tracer()
    with t.context_manager("op") as span:
        assert isinstance(span, Span)


def test_context_manager_span_ended_after_block() -> None:
    t = Tracer()
    with t.context_manager("op") as span:
        pass
    assert span.end_ns is not None


def test_context_manager_span_ended_on_exception() -> None:
    t = Tracer()
    span_ref: Span | None = None
    try:
        with t.context_manager("op") as span:
            span_ref = span
            raise ValueError("oops")
    except ValueError:
        pass
    assert span_ref is not None
    assert span_ref.end_ns is not None


# ---------------------------------------------------------------------------
# PrometheusMetrics — counters
# ---------------------------------------------------------------------------


def test_inc_counter_default_increment() -> None:
    m = PrometheusMetrics()
    m.inc_counter("requests_total")
    assert m._counters["requests_total"] == 1.0


def test_inc_counter_custom_value() -> None:
    m = PrometheusMetrics()
    m.inc_counter("requests_total", 5.0)
    assert m._counters["requests_total"] == 5.0


def test_inc_counter_accumulates() -> None:
    m = PrometheusMetrics()
    m.inc_counter("requests_total")
    m.inc_counter("requests_total")
    assert m._counters["requests_total"] == 2.0


# ---------------------------------------------------------------------------
# PrometheusMetrics — gauges
# ---------------------------------------------------------------------------


def test_set_gauge_sets_value() -> None:
    m = PrometheusMetrics()
    m.set_gauge("memory_bytes", 1024.0)
    assert m._gauges["memory_bytes"] == 1024.0


def test_set_gauge_overwrites() -> None:
    m = PrometheusMetrics()
    m.set_gauge("memory_bytes", 100.0)
    m.set_gauge("memory_bytes", 200.0)
    assert m._gauges["memory_bytes"] == 200.0


# ---------------------------------------------------------------------------
# PrometheusMetrics — histograms
# ---------------------------------------------------------------------------


def test_observe_histogram_records_value() -> None:
    m = PrometheusMetrics()
    m.observe_histogram("latency_seconds", 0.05)
    assert 0.05 in m._histograms["latency_seconds"]


def test_observe_histogram_multiple_values() -> None:
    m = PrometheusMetrics()
    m.observe_histogram("latency_seconds", 0.1)
    m.observe_histogram("latency_seconds", 0.2)
    assert len(m._histograms["latency_seconds"]) == 2


# ---------------------------------------------------------------------------
# PrometheusMetrics — render_text
# ---------------------------------------------------------------------------


def test_render_text_contains_counter_name() -> None:
    m = PrometheusMetrics()
    m.inc_counter("http_requests_total")
    out = m.render_text()
    assert "http_requests_total" in out


def test_render_text_contains_gauge_name() -> None:
    m = PrometheusMetrics()
    m.set_gauge("active_connections", 42.0)
    out = m.render_text()
    assert "active_connections" in out


def test_render_text_counter_type_annotation() -> None:
    m = PrometheusMetrics()
    m.inc_counter("errors_total")
    out = m.render_text()
    assert "counter" in out


def test_render_text_gauge_type_annotation() -> None:
    m = PrometheusMetrics()
    m.set_gauge("cpu_usage", 0.75)
    out = m.render_text()
    assert "gauge" in out


# ---------------------------------------------------------------------------
# wsgi_metrics_app
# ---------------------------------------------------------------------------


def _make_environ(path: str = "/metrics", method: str = "GET") -> dict:
    return {"PATH_INFO": path, "REQUEST_METHOD": method}


def test_wsgi_metrics_app_get_metrics_200() -> None:
    m = PrometheusMetrics()
    statuses: list[str] = []
    m.inc_counter("test_counter")

    def start_response(status: str, headers: list) -> None:
        statuses.append(status)

    m.wsgi_metrics_app(_make_environ("/metrics", "GET"), start_response)
    assert statuses[0].startswith("200")


def test_wsgi_metrics_app_returns_bytes() -> None:
    m = PrometheusMetrics()

    def start_response(status: str, headers: list) -> None:
        pass

    result = m.wsgi_metrics_app(_make_environ("/metrics"), start_response)
    assert isinstance(result, list)
    assert all(isinstance(b, bytes) for b in result)


def test_wsgi_metrics_app_404_on_unknown_path() -> None:
    m = PrometheusMetrics()
    statuses: list[str] = []

    def start_response(status: str, headers: list) -> None:
        statuses.append(status)

    m.wsgi_metrics_app(_make_environ("/unknown"), start_response)
    assert statuses[0].startswith("404")


# ---------------------------------------------------------------------------
# Singletons and registries
# ---------------------------------------------------------------------------


def test_default_tracer_is_tracer_instance() -> None:
    assert isinstance(DEFAULT_TRACER, Tracer)


def test_default_metrics_is_prometheus_metrics_instance() -> None:
    assert isinstance(DEFAULT_METRICS, PrometheusMetrics)


def test_otel_tracer_registry_is_dict() -> None:
    assert isinstance(OTEL_TRACER_REGISTRY, dict)


def test_otel_tracer_registry_has_default() -> None:
    assert "default" in OTEL_TRACER_REGISTRY
