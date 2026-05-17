"""Structured JSON audit logger with thread-safe buffering and output."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditLogEntry:
    """Immutable structured audit log entry."""

    timestamp: float
    actor: str
    action: str
    resource: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "status": self.status,
            "metadata": dict(self.metadata),
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, separators=(",", ":"))


class AuditLogger:
    """Thread-safe structured audit logger.

    Supports in-memory buffering and optional custom output sinks.
    All operations are non-blocking and lock-protected.
    """

    def __init__(
        self,
        *,
        buffer_size: int = 10_000,
        sinks: list[Callable[[AuditLogEntry], None]] | None = None,
    ) -> None:
        self._buffer_size = max(1, buffer_size)
        self._sinks: list[Callable[[AuditLogEntry], None]] = list(sinks) if sinks else []
        self._buffer: list[AuditLogEntry] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def log(
        self,
        actor: str,
        action: str,
        resource: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AuditLogEntry:
        """Record an audit entry. Thread-safe and non-blocking."""
        entry = AuditLogEntry(
            timestamp=time.time(),
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            metadata=deepcopy(metadata) if metadata else {},
            trace_id=trace_id,
        )
        with self._lock:
            if len(self._buffer) >= self._buffer_size:
                self._buffer.pop(0)
            self._buffer.append(entry)
        for sink in self._sinks:
            try:
                sink(entry)
            except Exception:
                pass  # sinks must not raise into caller
        return entry

    def get_entries(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        status: str | None = None,
        trace_id: str | None = None,
        limit: int | None = None,
    ) -> list[AuditLogEntry]:
        """Filter buffered entries. Returns newest-first."""
        with self._lock:
            entries = list(self._buffer)
        # iterate newest-first
        results: list[AuditLogEntry] = []
        for entry in reversed(entries):
            if actor is not None and entry.actor != actor:
                continue
            if action is not None and entry.action != action:
                continue
            if resource is not None and entry.resource != resource:
                continue
            if status is not None and entry.status != status:
                continue
            if trace_id is not None and entry.trace_id != trace_id:
                continue
            results.append(entry)
            if limit is not None and len(results) >= limit:
                break
        return results

    def add_sink(self, sink: Callable[[AuditLogEntry], None]) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[AuditLogEntry], None]) -> None:
        with self._lock:
            try:
                self._sinks.remove(sink)
            except ValueError:
                pass

    def flush(self) -> list[AuditLogEntry]:
        """Return and clear the in-memory buffer."""
        with self._lock:
            snapshot = list(self._buffer)
            self._buffer.clear()
        return snapshot

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
