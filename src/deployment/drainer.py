"""Graceful deployment drainer with connection tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ActiveConnection:
    id: str
    started: float
    finished: float = 0.0

    @property
    def duration(self) -> float:
        return (self.finished or time.monotonic()) - self.started


@dataclass
class DeploymentDrainer:
    """Track active connections and drain before shutdown."""

    drain_timeout: float = 30.0
    _connections: dict[str, ActiveConnection] = field(default_factory=dict, repr=False)

    def track(self, conn_id: str) -> None:
        self._connections[conn_id] = ActiveConnection(id=conn_id, started=time.monotonic())

    def finish(self, conn_id: str) -> None:
        conn = self._connections.get(conn_id)
        if conn:
            conn.finished = time.monotonic()

    def active_count(self) -> int:
        self._prune()
        return len([c for c in self._connections.values() if c.finished == 0.0])

    def drain(self) -> bool:
        deadline = time.monotonic() + self.drain_timeout
        while time.monotonic() < deadline:
            if self.active_count() == 0:
                return True
            time.sleep(0.1)
        return self.active_count() == 0

    def _prune(self) -> None:
        now = time.monotonic()
        self._connections = {
            k: v for k, v in self._connections.items()
            if v.finished == 0.0 or (v.finished > 0 and now - v.finished < 60)
        }


DEPLOYMENT_DRAINER = DeploymentDrainer()