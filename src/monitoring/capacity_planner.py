"""Capacity planner: forecast metric trends and time-to-limit."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CapacityForecast:
    metric_name: str
    current_value: float
    trend: TrendDirection
    slope: float
    forecast_at_horizon: float
    days_to_limit: float | None
    headroom_percent: float


@dataclass
class CapacityPlannerConfig:
    forecast_horizon_days: int = 30
    headroom_target_percent: float = 20.0
    limit_map: dict[str, float] = field(default_factory=dict)


class CapacityPlanner:
    def __init__(self, config: CapacityPlannerConfig | None = None) -> None:
        self.config = config or CapacityPlannerConfig()

    def _linear_regression(self, values: list[float]) -> tuple[float, float]:
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return 0.0, values[0]
        xs = list(range(n))
        mean_x = math.fsum(xs) / n
        mean_y = math.fsum(values) / n
        num = 0.0
        den = 0.0
        for x, y in zip(xs, values):
            num += (x - mean_x) * (y - mean_y)
            den += (x - mean_x) ** 2
        if den == 0.0:
            return 0.0, mean_y
        slope = num / den
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def analyze(
        self,
        name: str,
        values: list[float],
        timestamps_days: list[float] | None = None,
    ) -> CapacityForecast:
        if not values:
            return CapacityForecast(
                metric_name=name,
                current_value=0.0,
                trend=TrendDirection.UNKNOWN,
                slope=0.0,
                forecast_at_horizon=0.0,
                days_to_limit=None,
                headroom_percent=100.0,
            )
        current = values[-1]
        n = len(values)
        mean = math.fsum(values) / n
        slope, intercept = self._linear_regression(values)

        # Decide trend
        if n < 2:
            trend = TrendDirection.UNKNOWN
        else:
            threshold = 0.01 * abs(mean) if mean != 0 else 0.0
            if slope > threshold:
                trend = TrendDirection.INCREASING
            elif slope < -threshold:
                trend = TrendDirection.DECREASING
            else:
                trend = TrendDirection.STABLE

        horizon = self.config.forecast_horizon_days
        forecast_at_horizon = intercept + slope * (n - 1 + horizon)

        limit = self.config.limit_map.get(name)
        if limit is not None and slope > 0 and current < limit:
            days_to_limit = (limit - current) / slope
        else:
            days_to_limit = None

        if limit is not None and limit > 0:
            headroom = (limit - current) / limit * 100.0
        else:
            headroom = 100.0

        return CapacityForecast(
            metric_name=name,
            current_value=current,
            trend=trend,
            slope=slope,
            forecast_at_horizon=forecast_at_horizon,
            days_to_limit=days_to_limit,
            headroom_percent=headroom,
        )

    def analyze_all(self, metrics: dict[str, list[float]]) -> list[CapacityForecast]:
        return [self.analyze(name, vals) for name, vals in metrics.items()]

    def at_risk(
        self, forecasts: list[CapacityForecast], threshold_days: float = 30
    ) -> list[CapacityForecast]:
        return [
            f
            for f in forecasts
            if f.days_to_limit is not None and f.days_to_limit <= threshold_days
        ]

    def report(self, forecasts: list[CapacityForecast]) -> str:
        lines: list[str] = []
        lines.append(f"Capacity Report ({len(forecasts)} metrics)")
        lines.append("=" * 40)
        at_risk = self.at_risk(forecasts, self.config.forecast_horizon_days)
        if at_risk:
            lines.append(f"AT RISK ({len(at_risk)}):")
            for f in at_risk:
                lines.append(
                    f"  - {f.metric_name}: current={f.current_value:.2f} "
                    f"trend={f.trend.value} days_to_limit={f.days_to_limit:.1f}"
                )
        else:
            lines.append("AT RISK: none")
        lines.append("")
        lines.append("All metrics:")
        for f in forecasts:
            d2l = (
                f"{f.days_to_limit:.1f}d"
                if f.days_to_limit is not None
                else "n/a"
            )
            lines.append(
                f"  - {f.metric_name}: trend={f.trend.value} slope={f.slope:.4f} "
                f"headroom={f.headroom_percent:.1f}% days_to_limit={d2l}"
            )
        return "\n".join(lines)


CAPACITY_PLANNER_REGISTRY = {"default": CapacityPlanner}
