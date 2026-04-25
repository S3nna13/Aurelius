"""Heartbeat protocol for service-to-service health checks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Heartbeat:
    seq: int
    timestamp: float
    source: str

    def is_stale(self, max_age: float = 10.0) -> bool:
        return time.monotonic() - self.timestamp > max_age


@dataclass
class HeartbeatMonitor:
    interval: float = 5.0
    timeout: float = 15.0
    _last_heartbeats: dict[str, Heartbeat] = field(default_factory=dict, repr=False)
    _seq: int = 0

    def beat(self, source: str) -> Heartbeat:
        self._seq += 1
        hb = Heartbeat(seq=self._seq, timestamp=time.monotonic(), source=source)
        self._last_heartbeats[source] = hb
        return hb

    def is_alive(self, source: str) -> bool:
        hb = self._last_heartbeats.get(source)
        if hb is None:
            return False
        return not hb.is_stale(self.timeout)

    def dead_services(self) -> list[str]:
        return [s for s in self._last_heartbeats if not self.is_alive(s)]

    def active_services(self) -> list[str]:
        return [s for s in self._last_heartbeats if self.is_alive(s)]


HEARTBEAT_MONITOR = HeartbeatMonitor()