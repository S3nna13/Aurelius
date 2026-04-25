"""Deployment health monitor with configurable alert thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class HealthMetric:
    name: str
    value: float
    threshold: float
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def healthy(self) -> bool:
        return self.value < self.threshold


@dataclass
class DeploymentHealthMonitor:
    _metrics: list[HealthMetric] = field(default_factory=list, repr=False)

    def record(self, metric: HealthMetric) -> None:
        self._metrics.append(metric)

    def status(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for m in self._metrics:
            result[m.name] = m.healthy
        return result

    def failures(self) -> list[HealthMetric]:
        return [m for m in self._metrics if not m.healthy]

    def recent(self, limit: int = 10) -> list[HealthMetric]:
        return self._metrics[-limit:]

    def clear(self) -> None:
        self._metrics.clear()


DEPLOYMENT_HEALTH = DeploymentHealthMonitor()