"""High-level telemetry wrapper combining audit logging, metrics, and tracing."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.observability.audit_logger import AuditLogEntry, AuditLogger
from src.observability.event_bus import EventBus
from src.observability.metrics_collector import MetricsCollector
from src.observability.trace_context import TraceContext


@dataclass
class TelemetryResult:
    """Result of a telemetry-wrapped operation."""

    success: bool
    duration_ms: float
    trace_id: str | None = None
    audit_entry: AuditLogEntry | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTelemetry:
    """High-level telemetry facade for agent operations.

    Combines AuditLogger, EventBus, MetricsCollector, and TraceContext
    into a single, thread-safe, non-blocking interface.
    """

    def __init__(
        self,
        *,
        audit_logger: AuditLogger | None = None,
        event_bus: EventBus | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self.audit = AuditLogger() if audit_logger is None else audit_logger
        self.events = EventBus() if event_bus is None else event_bus
        self.metrics = MetricsCollector() if metrics is None else metrics

    # ------------------------------------------------------------------ #
    # Simple helpers
    # ------------------------------------------------------------------ #

    def record_action(
        self,
        actor: str,
        action: str,
        resource: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AuditLogEntry:
        """Record an audit log entry and emit a matching event."""
        entry = self.audit.log(
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            metadata=metadata,
            trace_id=trace_id,
        )
        self.events.publish_typed(
            event_type=f"audit.{action}",
            payload={
                "actor": actor,
                "resource": resource,
                "status": status,
                "metadata": entry.metadata,
            },
            trace_id=trace_id,
        )
        return entry

    def count(self, name: str, value: float = 1.0) -> float:
        """Increment a metric counter."""
        return self.metrics.increment(name, value)

    def gauge(self, name: str, value: float) -> float:
        """Set a gauge."""
        return self.metrics.gauge(name, value)

    def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram value."""
        self.metrics.record(name, value)

    # ------------------------------------------------------------------ #
    # Traced operation wrapper
    # ------------------------------------------------------------------ #

    @contextmanager
    def trace(
        self,
        operation: str,
        *,
        actor: str = "agent",
        resource: str = "",
        metadata: dict[str, Any] | None = None,
        parent_ctx: TraceContext | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Context manager that creates a trace span around a block of code.

        Automatically records:
          - an audit log entry on completion
          - operation duration histogram
          - success/failure counters
          - an event on the bus
        """
        ctx = TraceContext.new(parent=parent_ctx)
        previous = TraceContext.current()
        TraceContext.set_current(ctx)
        start = time.perf_counter()
        status = "started"
        try:
            yield ctx
            status = "success"
        except Exception:
            status = "failure"
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            TraceContext.set_current(previous)
            meta = dict(metadata) if metadata else {}
            meta["duration_ms"] = duration_ms
            meta["status"] = status
            if status == "failure":
                meta["error_type"] = "Exception"
            self.audit.log(
                actor=actor,
                action=operation,
                resource=resource or operation,
                status=status,
                metadata=meta,
                trace_id=ctx.trace_id,
            )
            self.metrics.record(f"op.{operation}.duration_ms", duration_ms)
            self.metrics.increment(f"op.{operation}.{status}")
            self.events.publish_typed(
                event_type=f"op.{operation}.{status}",
                payload={
                    "actor": actor,
                    "resource": resource or operation,
                    "duration_ms": duration_ms,
                    "metadata": meta,
                },
                trace_id=ctx.trace_id,
            )

    def trace_result(
        self,
        operation: str,
        *,
        actor: str = "agent",
        resource: str = "",
        metadata: dict[str, Any] | None = None,
        parent_ctx: TraceContext | None = None,
    ) -> TelemetryResult:
        """Convenience: run trace() and return a TelemetryResult.

        This is useful when you want to capture the result without using
        a context manager.
        """
        ctx = TraceContext.new(parent=parent_ctx)
        start = time.perf_counter()
        status = "success"
        try:
            # No operation body; caller uses this as a marker
            duration_ms = (time.perf_counter() - start) * 1000.0
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            status = "failure"
            raise
        finally:
            # handled below
            pass  # pragma: no cover

        duration_ms = (time.perf_counter() - start) * 1000.0
        meta = dict(metadata) if metadata else {}
        meta["duration_ms"] = duration_ms
        meta["status"] = status
        entry = self.audit.log(
            actor=actor,
            action=operation,
            resource=resource or operation,
            status=status,
            metadata=meta,
            trace_id=ctx.trace_id,
        )
        self.metrics.record(f"op.{operation}.duration_ms", duration_ms)
        self.metrics.increment(f"op.{operation}.{status}")
        self.events.publish_typed(
            event_type=f"op.{operation}.{status}",
            payload={
                "actor": actor,
                "resource": resource or operation,
                "duration_ms": duration_ms,
                "metadata": meta,
            },
            trace_id=ctx.trace_id,
        )
        return TelemetryResult(
            success=status == "success",
            duration_ms=duration_ms,
            trace_id=ctx.trace_id,
            audit_entry=entry,
            metadata=meta,
        )
