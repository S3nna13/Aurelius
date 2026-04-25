"""Heartbeat monitor for protocol connection health."""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable


class HeartbeatError(ValueError):
    """Raised on heartbeat monitor misuse."""


@dataclass(frozen=True)
class Connection:
    conn_id: str
    endpoint: str
    last_heartbeat: float
    status: str
    missed_beats: int


class HeartbeatMonitor:
    """Track connection health via heartbeats."""

    STATUSES = ("healthy", "unhealthy", "disconnected")

    def __init__(self) -> None:
        self._connections: dict[str, Connection] = {}
        self._lock = threading.Lock()
        self._on_state_change: Callable[[str, str, str], None] | None = None

    def set_on_state_change(self, callback: Callable[[str, str, str], None] | None) -> None:
        """Set or clear the state-change callback."""
        with self._lock:
            self._on_state_change = callback

    def _validate_conn_id(self, conn_id: str) -> None:
        if not isinstance(conn_id, str) or not re.fullmatch(r"[A-Za-z0-9\-]{1,128}", conn_id):
            raise HeartbeatError("conn_id must be alphanumeric/hyphen, 1-128 chars")

    def _validate_endpoint(self, endpoint: str) -> None:
        if not isinstance(endpoint, str) or not endpoint:
            raise HeartbeatError("endpoint must be a non-empty string")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in ("http", "https", "ws", "wss", "tcp"):
            raise HeartbeatError(f"endpoint scheme not allowed: {parsed.scheme}")
        if not parsed.netloc and not parsed.path:
            raise HeartbeatError("endpoint must have host or path")
        if parsed.scheme in ("http", "https", "ws", "wss", "tcp") and not parsed.netloc:
            raise HeartbeatError("endpoint must have a netloc for network schemes")

    def _update_status(self, conn_id: str, new_status: str) -> None:
        old_status = self._connections[conn_id].status
        if old_status == new_status:
            return
        self._connections[conn_id] = Connection(
            conn_id=conn_id,
            endpoint=self._connections[conn_id].endpoint,
            last_heartbeat=self._connections[conn_id].last_heartbeat,
            status=new_status,
            missed_beats=self._connections[conn_id].missed_beats,
        )
        cb = self._on_state_change
        if cb is not None:
            try:
                cb(conn_id, old_status, new_status)
            except Exception:
                # Callback errors must not break monitor state  # noqa: S110
                pass

    def register(self, conn_id: str, endpoint: str) -> None:
        """Start tracking a connection."""
        self._validate_conn_id(conn_id)
        self._validate_endpoint(endpoint)
        with self._lock:
            self._connections[conn_id] = Connection(
                conn_id=conn_id,
                endpoint=endpoint,
                last_heartbeat=time.time(),
                status="healthy",
                missed_beats=0,
            )

    def heartbeat(self, conn_id: str) -> None:
        """Record a heartbeat for a connection."""
        self._validate_conn_id(conn_id)
        with self._lock:
            if conn_id not in self._connections:
                raise HeartbeatError(f"connection not found: {conn_id}")
            conn = self._connections[conn_id]
            self._connections[conn_id] = Connection(
                conn_id=conn_id,
                endpoint=conn.endpoint,
                last_heartbeat=time.time(),
                status="healthy",
                missed_beats=0,
            )
            self._update_status(conn_id, "healthy")

    def check_health(self, timeout_seconds: float = 30.0, max_missed: int = 3) -> list[Connection]:
        """Check all connections and return unhealthy/disconnected ones."""
        timeout_seconds = max(1.0, timeout_seconds)
        max_missed = max(1, max_missed)
        now = time.time()
        with self._lock:
            for conn_id in list(self._connections):
                conn = self._connections[conn_id]
                elapsed = now - conn.last_heartbeat
                if elapsed > timeout_seconds:
                    new_missed = conn.missed_beats + 1
                    if new_missed >= max_missed:
                        new_status = "disconnected"
                    else:
                        new_status = "unhealthy"
                    self._connections[conn_id] = Connection(
                        conn_id=conn_id,
                        endpoint=conn.endpoint,
                        last_heartbeat=conn.last_heartbeat,
                        status=new_status,
                        missed_beats=new_missed,
                    )
                    self._update_status(conn_id, new_status)
            return [
                c for c in self._connections.values() if c.status in ("unhealthy", "disconnected")
            ]

    def get_status(self, conn_id: str) -> Connection:
        """Get the current status of a connection."""
        self._validate_conn_id(conn_id)
        with self._lock:
            if conn_id not in self._connections:
                raise HeartbeatError(f"connection not found: {conn_id}")
            return self._connections[conn_id]

    def get_unhealthy(self) -> list[Connection]:
        """Return all unhealthy or disconnected connections."""
        with self._lock:
            return [
                c for c in self._connections.values() if c.status in ("unhealthy", "disconnected")
            ]

    def remove(self, conn_id: str) -> None:
        """Stop tracking a connection."""
        self._validate_conn_id(conn_id)
        with self._lock:
            if conn_id not in self._connections:
                raise HeartbeatError(f"connection not found: {conn_id}")
            del self._connections[conn_id]
