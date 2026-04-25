"""Tests for MetricsCollector (~50 tests)."""
from __future__ import annotations

import time
import pytest
from src.monitoring.metrics_collector import (
    MetricType,
    MetricSample,
    MetricsCollector,
    METRICS_COLLECTOR,
)


# ---------------------------------------------------------------------------
# MetricType enum
# ---------------------------------------------------------------------------

class TestMetricTypeEnum:
    def test_counter_value(self):
        assert MetricType.COUNTER == "counter"

    def test_gauge_value(self):
        assert MetricType.GAUGE == "gauge"

    def test_histogram_value(self):
        assert MetricType.HISTOGRAM == "histogram"

    def test_is_str(self):
        assert isinstance(MetricType.COUNTER, str)

    def test_all_three_members(self):
        assert len(MetricType) == 3


# ---------------------------------------------------------------------------
# MetricSample dataclass
# ---------------------------------------------------------------------------

class TestMetricSample:
    def test_fields_set(self):
        s = MetricSample(name="foo", value=1.0, metric_type=MetricType.GAUGE)
        assert s.name == "foo"
        assert s.value == 1.0
        assert s.metric_type == MetricType.GAUGE

    def test_timestamp_auto(self):
        before = time.monotonic()
        s = MetricSample(name="x", value=0.0, metric_type=MetricType.COUNTER)
        after = time.monotonic()
        assert before <= s.timestamp <= after

    def test_labels_default_empty(self):
        s = MetricSample(name="x", value=0.0, metric_type=MetricType.GAUGE)
        assert s.labels == {}

    def test_labels_set(self):
        s = MetricSample(name="x", value=0.0, metric_type=MetricType.GAUGE, labels={"env": "prod"})
        assert s.labels["env"] == "prod"

    def test_value_float(self):
        s = MetricSample(name="x", value=3.14, metric_type=MetricType.HISTOGRAM)
        assert s.value == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# MetricsCollector.record
# ---------------------------------------------------------------------------

class TestRecord:
    def setup_method(self):
        self.mc = MetricsCollector()

    def test_returns_metric_sample(self):
        result = self.mc.record("cpu", 0.5)
        assert isinstance(result, MetricSample)

    def test_name_set(self):
        result = self.mc.record("mem", 100.0)
        assert result.name == "mem"

    def test_value_set(self):
        result = self.mc.record("lat", 42.0)
        assert result.value == 42.0

    def test_default_type_gauge(self):
        result = self.mc.record("x", 1.0)
        assert result.metric_type == MetricType.GAUGE

    def test_explicit_type(self):
        result = self.mc.record("x", 1.0, MetricType.HISTOGRAM)
        assert result.metric_type == MetricType.HISTOGRAM

    def test_labels_captured(self):
        result = self.mc.record("x", 1.0, region="us-east")
        assert result.labels["region"] == "us-east"

    def test_multiple_labels(self):
        result = self.mc.record("x", 1.0, region="us", env="prod")
        assert result.labels == {"region": "us", "env": "prod"}

    def test_stored_in_buffer(self):
        self.mc.record("hits", 1.0)
        assert len(self.mc.get_samples("hits")) == 1

    def test_multiple_records_accumulated(self):
        for i in range(5):
            self.mc.record("hits", float(i))
        assert len(self.mc.get_samples("hits")) == 5


# ---------------------------------------------------------------------------
# MetricsCollector.increment
# ---------------------------------------------------------------------------

class TestIncrement:
    def setup_method(self):
        self.mc = MetricsCollector()

    def test_creates_sample(self):
        self.mc.increment("req")
        samples = self.mc.get_samples("req")
        assert len(samples) == 1

    def test_type_is_counter(self):
        self.mc.increment("req")
        s = self.mc.get_samples("req")[0]
        assert s.metric_type == MetricType.COUNTER

    def test_default_by_one(self):
        self.mc.increment("req")
        s = self.mc.get_samples("req")[0]
        assert s.value == 1.0

    def test_custom_by(self):
        self.mc.increment("req", by=5.0)
        s = self.mc.get_samples("req")[0]
        assert s.value == 5.0

    def test_labels_forwarded(self):
        self.mc.increment("req", endpoint="/api")
        s = self.mc.get_samples("req")[0]
        assert s.labels["endpoint"] == "/api"


# ---------------------------------------------------------------------------
# MetricsCollector.get_samples
# ---------------------------------------------------------------------------

class TestGetSamples:
    def setup_method(self):
        self.mc = MetricsCollector()

    def test_unknown_metric_returns_empty(self):
        assert self.mc.get_samples("nonexistent") == []

    def test_returns_all_samples(self):
        for v in [1.0, 2.0, 3.0]:
            self.mc.record("m", v)
        samples = self.mc.get_samples("m")
        assert [s.value for s in samples] == [1.0, 2.0, 3.0]

    def test_last_n_limits(self):
        for v in range(10):
            self.mc.record("m", float(v))
        samples = self.mc.get_samples("m", last_n=3)
        assert len(samples) == 3

    def test_last_n_newest_last(self):
        for v in range(10):
            self.mc.record("m", float(v))
        samples = self.mc.get_samples("m", last_n=3)
        assert [s.value for s in samples] == [7.0, 8.0, 9.0]

    def test_last_n_larger_than_buffer(self):
        self.mc.record("m", 1.0)
        samples = self.mc.get_samples("m", last_n=100)
        assert len(samples) == 1

    def test_window_size_ring_buffer(self):
        mc = MetricsCollector(window_size=5)
        for i in range(10):
            mc.record("m", float(i))
        samples = mc.get_samples("m")
        # ring buffer keeps last 5
        assert len(samples) == 5
        assert samples[-1].value == 9.0


# ---------------------------------------------------------------------------
# MetricsCollector.summary
# ---------------------------------------------------------------------------

class TestSummary:
    def setup_method(self):
        self.mc = MetricsCollector()

    def test_empty_returns_all_zeros(self):
        s = self.mc.summary("missing")
        assert s == {"count": 0, "sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0,
                     "p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_count(self):
        for v in [1, 2, 3, 4, 5]:
            self.mc.record("m", float(v))
        assert self.mc.summary("m")["count"] == 5

    def test_sum(self):
        for v in [1, 2, 3]:
            self.mc.record("m", float(v))
        assert self.mc.summary("m")["sum"] == pytest.approx(6.0)

    def test_mean(self):
        for v in [2, 4, 6]:
            self.mc.record("m", float(v))
        assert self.mc.summary("m")["mean"] == pytest.approx(4.0)

    def test_min(self):
        for v in [5, 1, 3]:
            self.mc.record("m", float(v))
        assert self.mc.summary("m")["min"] == pytest.approx(1.0)

    def test_max(self):
        for v in [5, 1, 3]:
            self.mc.record("m", float(v))
        assert self.mc.summary("m")["max"] == pytest.approx(5.0)

    def test_p50_within_range(self):
        for v in range(1, 101):
            self.mc.record("m", float(v))
        s = self.mc.summary("m")
        assert s["min"] <= s["p50"] <= s["max"]

    def test_p95_within_range(self):
        for v in range(1, 101):
            self.mc.record("m", float(v))
        s = self.mc.summary("m")
        assert s["p50"] <= s["p95"] <= s["max"]

    def test_p99_within_range(self):
        for v in range(1, 101):
            self.mc.record("m", float(v))
        s = self.mc.summary("m")
        assert s["p95"] <= s["p99"] <= s["max"]

    def test_single_sample(self):
        self.mc.record("m", 7.0)
        s = self.mc.summary("m")
        assert s["count"] == 1
        assert s["mean"] == pytest.approx(7.0)
        assert s["p50"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# MetricsCollector.metric_names
# ---------------------------------------------------------------------------

class TestMetricNames:
    def setup_method(self):
        self.mc = MetricsCollector()

    def test_empty_initially(self):
        assert self.mc.metric_names() == []

    def test_includes_recorded_metrics(self):
        self.mc.record("alpha", 1.0)
        self.mc.record("beta", 2.0)
        names = self.mc.metric_names()
        assert "alpha" in names
        assert "beta" in names

    def test_no_duplicates(self):
        self.mc.record("m", 1.0)
        self.mc.record("m", 2.0)
        assert self.mc.metric_names().count("m") == 1


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_metrics_collector_exists(self):
        assert METRICS_COLLECTOR is not None
        assert isinstance(METRICS_COLLECTOR, MetricsCollector)
