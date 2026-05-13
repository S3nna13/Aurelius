"""Tests for MetricsCollector."""

from __future__ import annotations

import threading

from src.observability.metrics_collector import MetricsCollector


class TestCounters:
    def test_increment_default(self) -> None:
        m = MetricsCollector()
        assert m.increment("reqs") == 1.0
        assert m.increment("reqs") == 2.0

    def test_increment_custom(self) -> None:
        m = MetricsCollector()
        assert m.increment("bytes", 1024) == 1024.0

    def test_counter_value_missing(self) -> None:
        m = MetricsCollector()
        assert m.counter_value("missing") == 0.0

    def test_reset_counter(self) -> None:
        m = MetricsCollector()
        m.increment("c", 5)
        assert m.reset_counter("c") == 5.0
        assert m.counter_value("c") == 0.0


class TestHistograms:
    def test_record_and_summary(self) -> None:
        m = MetricsCollector()
        for v in [1, 2, 3, 4, 5]:
            m.record("lat", v)
        summary = m.histogram_summary("lat")
        assert summary["count"] == 5
        assert summary["min"] == 1
        assert summary["max"] == 5
        assert summary["mean"] == 3.0

    def test_histogram_p99(self) -> None:
        m = MetricsCollector()
        for i in range(100):
            m.record("h", float(i))
        summary = m.histogram_summary("h")
        assert summary["p99"] == 99.0

    def test_histogram_empty(self) -> None:
        m = MetricsCollector()
        summary = m.histogram_summary("empty")
        assert summary["count"] == 0
        assert summary["min"] is None

    def test_histogram_values(self) -> None:
        m = MetricsCollector()
        m.record("h", 1.5)
        m.record("h", 2.5)
        assert m.histogram_values("h") == [1.5, 2.5]

    def test_reset_histogram(self) -> None:
        m = MetricsCollector()
        m.record("h", 1)
        prev = m.reset_histogram("h")
        assert prev == [1]
        assert m.histogram_values("h") == []

    def test_histogram_maxlen(self) -> None:
        m = MetricsCollector()
        for i in range(20_000):
            m.record("h", float(i))
        assert len(m.histogram_values("h")) == 10_000


class TestGauges:
    def test_gauge(self) -> None:
        m = MetricsCollector()
        assert m.gauge("temp", 36.6) == 36.6
        assert m.gauge_value("temp") == 36.6

    def test_gauge_missing(self) -> None:
        m = MetricsCollector()
        assert m.gauge_value("missing") is None

    def test_gauge_age(self) -> None:
        m = MetricsCollector()
        m.gauge("temp", 1.0)
        age = m.gauge_age("temp")
        assert age is not None
        assert age >= 0.0


class TestBulk:
    def test_snapshot(self) -> None:
        m = MetricsCollector()
        m.increment("c", 1)
        m.gauge("g", 2)
        m.record("h", 3)
        snap = m.snapshot()
        assert snap["counters"] == {"c": 1.0}
        assert snap["gauges"]["g"]["value"] == 2.0
        assert snap["histograms"]["h"] == [3.0]

    def test_reset_all(self) -> None:
        m = MetricsCollector()
        m.increment("c", 1)
        m.gauge("g", 2)
        m.record("h", 3)
        m.reset_all()
        assert m.counter_value("c") == 0.0
        assert m.gauge_value("g") is None
        assert m.histogram_values("h") == []


class TestThreadSafety:
    def test_concurrent_increments(self) -> None:
        m = MetricsCollector()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(500):
                    m.increment("c", 1.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.counter_value("c") == 5000.0

    def test_concurrent_histogram(self) -> None:
        m = MetricsCollector()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(100):
                    m.record("h", float(i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.histogram_summary("h")["count"] == 1000
