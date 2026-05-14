"""Distributed tracing context for agent calls."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# Module-level ContextVar so it works correctly with frozen/slotted dataclass
_trace_ctx_var: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)
_trace_local = threading.local()


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Immutable distributed tracing context.

    Uses a ContextVar for automatic propagation across async boundaries,
    plus a thread-local fallback for synchronous code.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def new(cls, parent: TraceContext | None = None) -> TraceContext:
        """Create a new trace context.

        If *parent* is provided, inherits trace_id and sets parent_span_id.
        """
        trace_id = parent.trace_id if parent else _gen_id()
        span_id = _gen_id()
        parent_span_id = parent.span_id if parent else None
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )

    @classmethod
    def current(cls) -> TraceContext | None:
        """Return the current trace context for this task/thread."""
        ctx = _trace_ctx_var.get(None)
        if ctx is not None:
            return ctx
        return getattr(_trace_local, "current", None)

    @classmethod
    def set_current(cls, ctx: TraceContext | None) -> None:
        """Set the current trace context."""
        _trace_ctx_var.set(ctx)
        _trace_local.current = ctx

    @classmethod
    def clear_current(cls) -> None:
        """Remove the current trace context."""
        _trace_ctx_var.set(None)
        _trace_local.current = None

    # ------------------------------------------------------------------ #
    # Context manager / helper
    # ------------------------------------------------------------------ #

    @classmethod
    @contextmanager
    def scope(cls, ctx: TraceContext | None = None) -> Generator[TraceContext, None, None]:
        """Context manager that sets *ctx* (or a new one) as current."""
        if ctx is None:
            ctx = cls.new()
        previous = cls.current()
        cls.set_current(ctx)
        try:
            yield ctx
        finally:
            cls.set_current(previous)

    def child(self) -> TraceContext:
        """Create a child span from this context."""
        return self.new(parent=self)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]
