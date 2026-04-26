from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class McpHealthChecker:
    server_name: str
    timeout: float = 5.0
    _last_latency: float = 0.0
    _total_pings: int = 0
    _failures: int = 0

    def ping(self) -> float:
        start = time.perf_counter()
        self._last_latency = (time.perf_counter() - start) * 1000
        self._total_pings += 1
        return self._last_latency

    def status(self) -> HealthStatus:
        if self._last_latency > self.timeout * 1000:
            return HealthStatus.UNHEALTHY
        elif self._last_latency > self.timeout * 500:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_stats(self) -> dict:
        return {
            "total_pings": self._total_pings,
            "failures": self._failures,
            "last_latency_ms": self._last_latency,
        }

    def reset(self) -> None:
        self._last_latency = 0.0
        self._total_pings = 0
        self._failures = 0
