"""Tests for MetricAggregator (15+ tests)."""

from __future__ import annotations

import time

from src.monitoring.metric_aggregator import (
    METRIC_AGGREGATOR,
    AggregationWindow,
    MetricAggregator,
    MetricPoint,
)


class TestMetricPoint:
    def test_fields_stored(self):
        p = MetricPoint(name="cpu", value=42.0, timestamp=1.0)
        assert p.name == "cpu"
        assert p.value == 42.0
        assert p.timestamp == 1.0

    def test_labels_default_empty(self):
        p = MetricPoint(name="cpu", value=0.0, timestamp=0.0)
        assert p.labels == {}

    def test_labels_stored(self):
        p = MetricPoint(name="cpu", value=0.0, timestamp=0.0, labels={"host": "a"})
        assert p.labels["host"] == "a"


class TestAggregationWindow:
    def test_values(self):
        assert AggregationWindow.LAST_1M == "1m"
        assert AggregationWindow.LAST_5M == "5m"
        assert AggregationWindow.LAST_15M == "15m"
        assert AggregationWindow.LAST_1H == "1h"

    def test_is_str(self):
        assert isinstance(AggregationWindow.LAST_1M, str)


class TestMetricAggregatorRecord:
    def setup_method(self):
        self.ma = MetricAggregator()

    def test_record_returns_metric_point(self):
        p = self.ma.record("cpu", 50.0)
        assert isinstance(p, MetricPoint)

    def test_record_name_stored(self):
        p = self.ma.record("cpu", 50.0)
        assert p.name == "cpu"

    def test_record_value_stored(self):
        p = self.ma.record("latency", 123.4)
        assert p.value == 123.4

    def test_record_timestamp_recent(self):
        before = time.time()
        p = self.ma.record("cpu", 1.0)
        after = time.time()
        assert before <= p.timestamp <= after

    def test_record_labels_passed(self):
        p = self.ma.record("cpu", 1.0, labels={"env": "prod"})
        assert p.labels["env"] == "prod"

    def test_record_labels_none_defaults_empty(self):
        p = self.ma.record("cpu", 1.0, labels=None)
        assert p.labels == {}

    def test_record_multiple_metrics(self):
        self.ma.record("cpu", 1.0)
        self.ma.record("mem", 2.0)
        assert sorted(self.ma.list_metrics()) == ["cpu", "mem"]


class TestMetricAggregatorGetWindow:
    def setup_method(self):
        self.ma = MetricAggregator()

    def test_get_window_returns_list(self):
        self.ma.record("cpu", 1.0)
        result = self.ma.get_window("cpu", AggregationWindow.LAST_1M)
        assert isinstance(result, list)

    def test_get_window_unknown_metric_empty(self):
        result = self.ma.get_window("unknown", AggregationWindow.LAST_1M)
        assert result == []

    def test_get_window_includes_recent_points(self):
        self.ma.record("cpu", 1.0)
        self.ma.record("cpu", 2.0)
        result = self.ma.get_window("cpu", AggregationWindow.LAST_1M)
        assert len(result) == 2


class TestMetricAggregatorStats:
    def setup_method(self):
        self.ma = MetricAggregator()

    def test_stats_empty_metric_returns_zeros(self):
        s = self.ma.stats("cpu", AggregationWindow.LAST_1M)
        assert s["count"] == 0
        assert s["mean"] == 0.0

    def test_stats_keys_present(self):
        self.ma.record("cpu", 1.0)
        s = self.ma.stats("cpu", AggregationWindow.LAST_1M)
        for key in ("count", "mean", "min", "max", "p50", "p95", "p99", "stddev"):
            assert key in s

    def test_stats_count(self):
        for v in [1.0, 2.0, 3.0]:
            self.ma.record("cpu", v)
        s = self.ma.stats("cpu", AggregationWindow.LAST_1M)
        assert s["count"] == 3

    def test_stats_mean(self):
        for v in [1.0, 2.0, 3.0]:
            self.ma.record("x", v)
        s = self.ma.stats("x", AggregationWindow.LAST_1M)
        assert abs(s["mean"] - 2.0) < 1e-9

    def test_stats_min_max(self):
        for v in [10.0, 20.0, 30.0]:
            self.ma.record("y", v)
        s = self.ma.stats("y", AggregationWindow.LAST_1M)
        assert s["min"] == 10.0
        assert s["max"] == 30.0

    def test_stats_p50(self):
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            self.ma.record("z", v)
        s = self.ma.stats("z", AggregationWindow.LAST_1M)
        assert 2.5 <= s["p50"] <= 3.5

    def test_stats_stddev_zero_for_constant(self):
        for _ in range(5):
            self.ma.record("c", 7.0)
        s = self.ma.stats("c", AggregationWindow.LAST_1M)
        assert abs(s["stddev"]) < 1e-9


class TestMetricAggregatorRate:
    def setup_method(self):
        self.ma = MetricAggregator()

    def test_rate_zero_for_unknown_metric(self):
        assert self.ma.rate("cpu", AggregationWindow.LAST_1M) == 0.0

    def test_rate_positive_after_records(self):
        for _ in range(10):
            self.ma.record("req", 1.0)
        r = self.ma.rate("req", AggregationWindow.LAST_1M)
        assert r > 0.0

    def test_rate_is_count_over_window_seconds(self):
        for _ in range(60):
            self.ma.record("req", 1.0)
        r = self.ma.rate("req", AggregationWindow.LAST_1M)
        assert abs(r - 1.0) < 1e-9


class TestMetricAggregatorListFlush:
    def setup_method(self):
        self.ma = MetricAggregator()

    def test_list_metrics_empty(self):
        assert self.ma.list_metrics() == []

    def test_list_metrics_after_records(self):
        self.ma.record("a", 1.0)
        self.ma.record("b", 2.0)
        assert set(self.ma.list_metrics()) == {"a", "b"}

    def test_flush_returns_count(self):
        for _ in range(5):
            self.ma.record("cpu", 1.0)
        n = self.ma.flush("cpu")
        assert n == 5

    def test_flush_clears_metric(self):
        self.ma.record("cpu", 1.0)
        self.ma.flush("cpu")
        assert "cpu" not in self.ma.list_metrics()

    def test_flush_unknown_metric_returns_zero(self):
        assert self.ma.flush("nonexistent") == 0

    def test_flush_does_not_affect_other_metrics(self):
        self.ma.record("a", 1.0)
        self.ma.record("b", 2.0)
        self.ma.flush("a")
        assert "b" in self.ma.list_metrics()


class TestMetricAggregatorRingBuffer:
    def test_max_points_respected(self):
        ma = MetricAggregator(max_points_per_metric=5)
        for i in range(10):
            ma.record("cpu", float(i))
        result = ma.get_window("cpu", AggregationWindow.LAST_1H)
        assert len(result) == 5

    def test_oldest_dropped_when_full(self):
        ma = MetricAggregator(max_points_per_metric=3)
        for i in range(5):
            ma.record("cpu", float(i))
        result = ma.get_window("cpu", AggregationWindow.LAST_1H)
        values = [p.value for p in result]
        assert 0.0 not in values
        assert 4.0 in values


class TestSingleton:
    def test_metric_aggregator_exists(self):
        assert METRIC_AGGREGATOR is not None
        assert isinstance(METRIC_AGGREGATOR, MetricAggregator)
