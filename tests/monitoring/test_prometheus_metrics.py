from __future__ import annotations

from src.monitoring.prometheus_metrics import METRICS_COLLECTOR, MetricsCollector


class TestMetricsCollector:
    def test_increment_counter_default(self):
        mc = MetricsCollector()
        mc.increment_counter("requests")
        assert mc.read_counter("requests") == 1.0

    def test_increment_counter_multiple(self):
        mc = MetricsCollector()
        mc.increment_counter("requests", 5.0)
        mc.increment_counter("requests", 3.0)
        assert mc.read_counter("requests") == 8.0

    def test_set_and_read_gauge(self):
        mc = MetricsCollector()
        mc.set_gauge("temperature", 36.5)
        assert mc.read_gauge("temperature") == 36.5

    def test_observe_histogram(self):
        mc = MetricsCollector()
        mc.observe_histogram("latency_ms", 10.0)
        mc.observe_histogram("latency_ms", 20.0)
        assert len(mc.read_histogram("latency_ms")) == 2

    def test_histogram_summary(self):
        mc = MetricsCollector()
        for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            mc.observe_histogram("test", float(v))
        summary = mc.histogram_summary("test")
        assert summary["count"] == 10
        assert summary["sum"] == 55.0
        assert summary["avg"] == 5.5
        assert summary["min"] == 1.0
        assert summary["max"] == 10.0

    def test_gauge_default_none(self):
        mc = MetricsCollector()
        assert mc.read_gauge("nonexistent") is None

    def test_counter_with_labels(self):
        mc = MetricsCollector()
        mc.increment_counter("requests", labels={"method": "GET"})
        mc.increment_counter("requests", labels={"method": "POST"})
        assert mc.read_counter("requests", labels={"method": "GET"}) == 1.0
        assert mc.read_counter("requests", labels={"method": "POST"}) == 1.0

    def test_export_text(self):
        mc = MetricsCollector()
        mc.increment_counter("test_counter")
        mc.set_gauge("test_gauge", 42.0)
        text = mc.export_text()
        assert "test_counter" in text
        assert "test_gauge" in text

    def test_reset(self):
        mc = MetricsCollector()
        mc.increment_counter("x")
        mc.reset()
        assert mc.read_counter("x") == 0.0

    def test_module_instance(self):
        assert METRICS_COLLECTOR is not None
        assert hasattr(METRICS_COLLECTOR, "increment_counter")
