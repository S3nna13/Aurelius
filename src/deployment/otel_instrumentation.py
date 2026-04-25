"""OpenTelemetry and Prometheus stub instrumentation for Aurelius.

Inspired by Helm (Apache-2.0, helm.sh), OpenTelemetry SDK (Apache-2.0, opentelemetry.io),
Prometheus text format (Apache-2.0), clean-room Aurelius implementation.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


# ---------------------------------------------------------------------------
# SpanKind
# ---------------------------------------------------------------------------


class SpanKind(Enum):
    """Kind of a tracing span."""

    CLIENT = "CLIENT"
    SERVER = "SERVER"
    INTERNAL = "INTERNAL"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single distributed-tracing span."""

    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    start_ns: int
    end_ns: int | None = None
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class Tracer:
    """Minimal in-process tracer — no external SDK required."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict | None = None,
    ) -> Span:
        """Create and record a new Span.

        Args:
            name: Human-readable name for the operation.
            kind: SpanKind (default INTERNAL).
            attributes: Optional initial attributes dict.

        Returns:
            A new Span with uuid-based IDs and a nanosecond start timestamp.
        """
        span = Span(
            span_id=uuid.uuid4().hex,
            trace_id=uuid.uuid4().hex,
            name=name,
            kind=kind,
            start_ns=time.time_ns(),
            attributes=dict(attributes) if attributes else {},
        )
        self._spans.append(span)
        return span

    def end_span(self, span: Span) -> None:
        """Mark a span as ended by setting end_ns to current time.

        Args:
            span: The Span to end.
        """
        span.end_ns = time.time_ns()

    def get_spans(self) -> list[Span]:
        """Return a shallow copy of all recorded spans."""
        return list(self._spans)

    @contextlib.contextmanager
    def context_manager(self, name: str, **attrs: object) -> Iterator[Span]:
        """Context manager that starts a span and ends it on exit.

        The span is ended even if an exception is raised inside the block.

        Args:
            name: Span name.
            **attrs: Additional attributes to attach to the span.

        Yields:
            The active Span.
        """
        span = self.start_span(name, attributes=dict(attrs))
        try:
            yield span
        finally:
            self.end_span(span)


# ---------------------------------------------------------------------------
# PrometheusMetrics
# ---------------------------------------------------------------------------


class PrometheusMetrics:
    """Minimal Prometheus text-format metrics collector — no prometheus_client required."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(name: str, labels: dict | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inc_counter(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        """Increment a counter by value (default 1.0).

        Labels are folded into the metric name key for simplicity.
        """
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set a gauge to an exact value."""
        key = self._key(name, labels)
        self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: dict | None = None) -> None:
        """Record an observation into a histogram bucket list."""
        key = self._key(name, labels)
        self._histograms.setdefault(key, []).append(value)

    def render_text(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []

        for key, value in self._counters.items():
            # Extract base name (strip label portion for TYPE line)
            base = key.split("{")[0]
            lines.append(f"# TYPE {base} counter")
            lines.append(f"{key} {value}")

        for key, value in self._gauges.items():
            base = key.split("{")[0]
            lines.append(f"# TYPE {base} gauge")
            lines.append(f"{key} {value}")

        for key, observations in self._histograms.items():
            base = key.split("{")[0]
            lines.append(f"# TYPE {base} histogram")
            for obs in observations:
                lines.append(f"{key} {obs}")

        return "\n".join(lines) + "\n" if lines else ""

    def wsgi_metrics_app(self, environ: dict, start_response) -> list[bytes]:  # type: ignore[type-arg]
        """WSGI application that serves GET /metrics in Prometheus text format.

        GET /metrics → 200 OK with render_text() body.
        Any other path → 404 Not Found.
        Any non-GET method → 405 Method Not Allowed.
        """
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if path != "/metrics":
            body = b"Not Found\n"
            start_response("404 Not Found", [
                ("Content-Type", "text/plain"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if method != "GET":
            body = b"Method Not Allowed\n"
            start_response("405 Method Not Allowed", [
                ("Content-Type", "text/plain"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        body = self.render_text().encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "text/plain; version=0.0.4; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ])
        return [body]


# ---------------------------------------------------------------------------
# Singletons and registries
# ---------------------------------------------------------------------------

OTEL_TRACER_REGISTRY: dict[str, Tracer] = {}

DEFAULT_TRACER: Tracer = Tracer()
DEFAULT_METRICS: PrometheusMetrics = PrometheusMetrics()

# Register default tracer in registry
OTEL_TRACER_REGISTRY["default"] = DEFAULT_TRACER

# ---------------------------------------------------------------------------
# Register into DEPLOY_TARGET_REGISTRY
# ---------------------------------------------------------------------------

from src.deployment.healthz import DEPLOY_TARGET_REGISTRY  # noqa: E402

DEPLOY_TARGET_REGISTRY["otel"] = {
    "type": "otel",
    "description": "OpenTelemetry + Prometheus stub instrumentation",
    "metrics_path": "/metrics",
    "tracer": "DEFAULT_TRACER",
}
