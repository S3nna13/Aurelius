"""Health checker: component checks, aggregated status, readiness/liveness."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0


class HealthChecker:
    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], HealthCheck]] = {}

    def register(self, name: str, check_fn: Callable[[], HealthCheck]) -> None:
        self._checks[name] = check_fn

    def run(self, name: str) -> HealthCheck:
        check_fn = self._checks[name]
        try:
            return check_fn()
        except Exception as e:
            return HealthCheck(name=name, status=HealthStatus.UNHEALTHY, message=str(e))

    def run_all(self) -> dict[str, HealthCheck]:
        results: dict[str, HealthCheck] = {}
        for name in self._checks:
            results[name] = self.run(name)
        return results

    def aggregate_status(self, checks: dict[str, HealthCheck]) -> HealthStatus:
        if not checks:
            return HealthStatus.UNKNOWN
        statuses = {c.status for c in checks.values()}
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(c.status == HealthStatus.HEALTHY for c in checks.values()):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def readiness(self) -> dict:
        checks = self.run_all()
        status = self.aggregate_status(checks)
        return {
            "status": status.value,
            "checks": {
                name: {"status": check.status.value, "message": check.message}
                for name, check in checks.items()
            },
        }

    def liveness(self) -> dict:
        try:
            self.run_all()
            return {"alive": True}
        except Exception as e:
            return {"alive": False, "error": str(e)}


HEALTH_CHECKER = HealthChecker()
