from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "HealthStatus",
    "HealthReport",
    "BackendHealthChecker",
    "BACKEND_HEALTH_REGISTRY",
]


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HealthReport:
    backend_name: str
    status: HealthStatus
    latency_ms: float
    error: str | None = None


class BackendHealthChecker:
    def __init__(self, backends: dict[str, Any] | None = None) -> None:
        self._backends: dict[str, Any] = backends if backends is not None else {}

    def check(self, name: str, backend: Any) -> HealthReport:
        start = time.monotonic()
        try:
            if hasattr(backend, "health"):
                ok = backend.health()
            elif hasattr(backend, "is_available"):
                ok = backend.is_available()
            else:
                return HealthReport(
                    backend_name=name,
                    status=HealthStatus.UNKNOWN,
                    latency_ms=(time.monotonic() - start) * 1000,
                    error="backend has no health() or is_available() method",
                )
            latency_ms = (time.monotonic() - start) * 1000
            status = HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY
            return HealthReport(backend_name=name, status=status, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthReport(
                backend_name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(exc),
            )

    def check_all(self) -> list[HealthReport]:
        return [self.check(name, backend) for name, backend in self._backends.items()]

    def register(self, name: str, backend: Any) -> None:
        self._backends[name] = backend

    def healthy_backends(self) -> list[str]:
        return [r.backend_name for r in self.check_all() if r.status == HealthStatus.HEALTHY]

    def p99_latency_ms(self, reports: list[HealthReport]) -> float:
        if not reports:
            return 0.0
        sorted_latencies = sorted(r.latency_ms for r in reports)
        idx = min(math.ceil(len(sorted_latencies) * 0.99) - 1, len(sorted_latencies) - 1)
        return sorted_latencies[idx]


BACKEND_HEALTH_REGISTRY: dict[str, type[BackendHealthChecker]] = {"default": BackendHealthChecker}
