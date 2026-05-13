"""Compatibility self-upgrade surface for the middle tier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricRecord:
    name: str
    value: float
    target: float
    unit: str = ""


class SelfUpgradeSystem:
    def __init__(self) -> None:
        self._metrics: list[MetricRecord] = []
        self._cycle_count = 0
        self._last_improvement: float | None = None

    def record_metric(self, name: str, value: float, target: float, unit: str = "") -> None:
        self._metrics.append(MetricRecord(name=name, value=value, target=target, unit=unit))

    def run_upgrade_cycle(self) -> dict[str, Any]:
        self._cycle_count += 1
        if not self._metrics:
            self._last_improvement = 0.0
            return {"status": "idle", "improvement": self._last_improvement}

        improvements = []
        for metric in self._metrics:
            denom = abs(metric.target) if metric.target else 1.0
            delta = abs(metric.value - metric.target)
            improvements.append(max(0.0, 1.0 - (delta / denom)))

        self._last_improvement = round(sum(improvements) / len(improvements), 3)
        return {
            "status": "completed",
            "improvement": self._last_improvement,
            "metrics": [metric.name for metric in self._metrics],
        }

    def get_summary(self) -> dict[str, Any]:
        return {
            "system": "Self-Upgrade Layer",
            "status": "active" if self._metrics else "idle",
            "cycle_count": self._cycle_count,
            "last_improvement": self._last_improvement,
            "metrics_tracked": [metric.name for metric in self._metrics],
        }
