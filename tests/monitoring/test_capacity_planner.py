"""Tests for CapacityPlanner."""
from __future__ import annotations

import pytest

from src.monitoring.capacity_planner import (
    CapacityForecast,
    CapacityPlanner,
    CapacityPlannerConfig,
    TrendDirection,
    CAPACITY_PLANNER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class TestTrendDirectionEnum:
    def test_increasing(self):
        assert TrendDirection.INCREASING == "increasing"

    def test_decreasing(self):
        assert TrendDirection.DECREASING == "decreasing"

    def test_stable(self):
        assert TrendDirection.STABLE == "stable"

    def test_unknown(self):
        assert TrendDirection.UNKNOWN == "unknown"

    def test_four_members(self):
        assert len(TrendDirection) == 4


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

class TestForecast:
    def test_fields(self):
        f = CapacityForecast(
            metric_name="m",
            current_value=10.0,
            trend=TrendDirection.INCREASING,
            slope=1.0,
            forecast_at_horizon=40.0,
            days_to_limit=20.0,
            headroom_percent=50.0,
        )
        assert f.metric_name == "m"
        assert f.current_value == 10.0

    def test_frozen(self):
        f = CapacityForecast(
            "m", 1.0, TrendDirection.STABLE, 0.0, 1.0, None, 100.0
        )
        with pytest.raises(Exception):
            f.slope = 2.0  # type: ignore


class TestConfig:
    def test_defaults(self):
        c = CapacityPlannerConfig()
        assert c.forecast_horizon_days == 30
        assert c.headroom_target_percent == 20.0
        assert c.limit_map == {}

    def test_custom(self):
        c = CapacityPlannerConfig(forecast_horizon_days=7, limit_map={"cpu": 100.0})
        assert c.forecast_horizon_days == 7
        assert c.limit_map["cpu"] == 100.0


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------

class TestLinearRegression:
    def setup_method(self):
        self.p = CapacityPlanner()

    def test_empty(self):
        slope, intercept = self.p._linear_regression([])
        assert slope == 0.0
        assert intercept == 0.0

    def test_single(self):
        slope, intercept = self.p._linear_regression([5.0])
        assert slope == 0.0
        assert intercept == 5.0

    def test_linear_up(self):
        slope, _ = self.p._linear_regression([0.0, 1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)

    def test_linear_down(self):
        slope, _ = self.p._linear_regression([4.0, 3.0, 2.0, 1.0, 0.0])
        assert slope == pytest.approx(-1.0)

    def test_flat(self):
        slope, intercept = self.p._linear_regression([5.0, 5.0, 5.0])
        assert slope == pytest.approx(0.0)
        assert intercept == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def setup_method(self):
        self.p = CapacityPlanner()

    def test_empty_returns_unknown(self):
        f = self.p.analyze("m", [])
        assert f.trend == TrendDirection.UNKNOWN
        assert f.days_to_limit is None

    def test_increasing_trend(self):
        f = self.p.analyze("m", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert f.trend == TrendDirection.INCREASING

    def test_decreasing_trend(self):
        f = self.p.analyze("m", [5.0, 4.0, 3.0, 2.0, 1.0])
        assert f.trend == TrendDirection.DECREASING

    def test_stable_trend(self):
        f = self.p.analyze("m", [5.0, 5.0, 5.0, 5.0])
        assert f.trend == TrendDirection.STABLE

    def test_current_value(self):
        f = self.p.analyze("m", [1.0, 2.0, 3.0])
        assert f.current_value == 3.0

    def test_slope_correct(self):
        f = self.p.analyze("m", [0.0, 1.0, 2.0, 3.0])
        assert f.slope == pytest.approx(1.0)

    def test_forecast_horizon(self):
        p = CapacityPlanner(CapacityPlannerConfig(forecast_horizon_days=10))
        f = p.analyze("m", [0.0, 1.0, 2.0])
        # intercept=0, slope=1, n=3 → forecast = 0 + 1*(2+10) = 12
        assert f.forecast_at_horizon == pytest.approx(12.0)

    def test_days_to_limit(self):
        p = CapacityPlanner(CapacityPlannerConfig(limit_map={"m": 100.0}))
        f = p.analyze("m", [0.0, 1.0, 2.0, 3.0, 4.0])
        # slope=1, current=4 → (100-4)/1 = 96
        assert f.days_to_limit == pytest.approx(96.0)

    def test_days_to_limit_none_when_no_limit(self):
        f = self.p.analyze("m", [0.0, 1.0, 2.0])
        assert f.days_to_limit is None

    def test_days_to_limit_none_when_decreasing(self):
        p = CapacityPlanner(CapacityPlannerConfig(limit_map={"m": 100.0}))
        f = p.analyze("m", [10.0, 9.0, 8.0, 7.0])
        assert f.days_to_limit is None

    def test_headroom_with_limit(self):
        p = CapacityPlanner(CapacityPlannerConfig(limit_map={"m": 100.0}))
        f = p.analyze("m", [1.0, 2.0, 20.0])
        # current=20, limit=100 → headroom = 80%
        assert f.headroom_percent == pytest.approx(80.0)

    def test_headroom_no_limit_100(self):
        f = self.p.analyze("m", [1.0, 2.0, 3.0])
        assert f.headroom_percent == 100.0


# ---------------------------------------------------------------------------
# analyze_all
# ---------------------------------------------------------------------------

class TestAnalyzeAll:
    def setup_method(self):
        self.p = CapacityPlanner()

    def test_empty(self):
        assert self.p.analyze_all({}) == []

    def test_multiple(self):
        results = self.p.analyze_all(
            {"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]}
        )
        assert len(results) == 2

    def test_names_preserved(self):
        results = self.p.analyze_all({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        names = {r.metric_name for r in results}
        assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# at_risk
# ---------------------------------------------------------------------------

class TestAtRisk:
    def setup_method(self):
        self.p = CapacityPlanner()

    def test_empty(self):
        assert self.p.at_risk([]) == []

    def test_filters_none(self):
        forecasts = [
            CapacityForecast("a", 0, TrendDirection.STABLE, 0.0, 0.0, None, 100.0),
        ]
        assert self.p.at_risk(forecasts) == []

    def test_filters_over_threshold(self):
        forecasts = [
            CapacityForecast(
                "a", 0, TrendDirection.INCREASING, 1.0, 0.0, 5.0, 50.0
            ),
            CapacityForecast(
                "b", 0, TrendDirection.INCREASING, 1.0, 0.0, 100.0, 50.0
            ),
        ]
        at_risk = self.p.at_risk(forecasts, threshold_days=30)
        assert len(at_risk) == 1
        assert at_risk[0].metric_name == "a"

    def test_threshold_inclusive(self):
        forecasts = [
            CapacityForecast(
                "a", 0, TrendDirection.INCREASING, 1.0, 0.0, 30.0, 50.0
            ),
        ]
        assert len(self.p.at_risk(forecasts, threshold_days=30)) == 1


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def setup_method(self):
        self.p = CapacityPlanner()

    def test_empty_report(self):
        out = self.p.report([])
        assert "Capacity Report" in out

    def test_mentions_at_risk(self):
        forecasts = [
            CapacityForecast(
                "a", 0, TrendDirection.INCREASING, 1.0, 0.0, 5.0, 50.0
            ),
        ]
        out = self.p.report(forecasts)
        assert "AT RISK" in out
        assert "a" in out

    def test_no_at_risk_label(self):
        forecasts = [
            CapacityForecast("a", 0, TrendDirection.STABLE, 0.0, 0.0, None, 100.0),
        ]
        out = self.p.report(forecasts)
        assert "none" in out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_default(self):
        assert CAPACITY_PLANNER_REGISTRY["default"] is CapacityPlanner
