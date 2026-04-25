"""Server health endpoint with liveness and readiness probes.

Aurelius LLM Project — stdlib only per Kubernetes best practices.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class HealthProbeType(Enum):
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class HealthProbeResult:
    """Result of a health probe check."""
    probe_type: HealthProbeType
    healthy: bool
    message: str
    timestamp_ns: int

    def to_dict(self) -> dict:
        return {
            "probe_type": self.probe_type.value,
            "healthy": self.healthy,
            "message": self.message,
            "timestamp_ns": self.timestamp_ns,
        }


class HealthProbe:
    """Health probe with configurable checks.
    
    Supports liveness (process is alive), readiness (can serve traffic),
    and startup (initialization complete) probes.
    """
    
    def __init__(
        self,
        on_liveness: Callable[[], bool] | None = None,
        on_readiness: Callable[[], bool] | None = None,
        on_startup: Callable[[], bool] | None = None,
    ) -> None:
        self._on_liveness = on_liveness or (lambda: True)
        self._on_readiness = on_readiness or (lambda: True)
        self._on_startup = on_startup or (lambda: True)
        self._startup_done = False
    
    def check(self, probe_type: HealthProbeType) -> HealthProbeResult:
        """Run a health probe check."""
        import time
        
        if probe_type == HealthProbeType.LIVENESS:
            healthy = self._on_liveness()
            msg = "liveness check OK" if healthy else "liveness check failed"
        elif probe_type == HealthProbeType.READINESS:
            if not self._startup_done:
                healthy = self._on_startup()
                self._startup_done = healthy
            healthy = self._on_readiness() if self._startup_done else False
            msg = "ready" if healthy else "not ready"
        else:  # STARTUP
            healthy = self._on_startup()
            self._startup_done = healthy
            msg = "startup complete" if healthy else "startup in progress"
        
        return HealthProbeResult(
            probe_type=probe_type,
            healthy=healthy,
            message=msg,
            timestamp_ns=int(time.time_ns()),
        )
    
    def check_all(self) -> dict[HealthProbeType, HealthProbeResult]:
        """Run all health probes."""
        return {pt: self.check(pt) for pt in HealthProbeType}
    
    def to_healthz_response(self) -> str:
        """Format /healthz response."""
        results = self.check_all()
        all_healthy = all(r.healthy for r in results.values())
        status = 200 if all_healthy else 503
        body = ",".join(
            f"{pt.value}={int(r.healthy)}" for pt, r in results.items()
        )
        return f"HTTP/1.1 {status} {'OK' if all_healthy else 'Service Unavailable'}\nContent-Type: text/plain\n\n{body}"


HEALTH_PROBE = HealthProbe()