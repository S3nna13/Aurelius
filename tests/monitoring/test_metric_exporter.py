from __future__ import annotations

from src.monitoring.metric_exporter import (
    METRICS_EXPORTER,
    MetricExporter,
    _escape_label,
    _fmt_counter,
    _fmt_gauge,
    _fmt_histogram,
    _fmt_labels,
    handle_metrics,
)
from src.monitoring.metrics_collector import METRICS_COLLECTOR, MetricType


class TestHelpers:
    def test_escape_label_backslash(self):
        assert _escape_label("a\\b") == "a\\\\b"

    def test_escape_label_quote(self):
        assert _escape_label('a"b') == 'a\\"b'

    def test_escape_label_newline(self):
        assert _escape_label("a\nb") == "a\\nb"

    def test_escape_label_noop(self):
        assert _escape_label("plain") == "plain"

    def test_fmt_labels_empty(self):
        assert _fmt_labels({}) == ""

    def test_fmt_labels_single(self):
        assert _fmt_labels({"k": "v"}) == '{k="v"}'

    def test_fmt_labels_sorted(self):
        assert _fmt_labels({"b": "2", "a": "1"}) == '{a="1",b="2"}'

    def test_fmt_labels_escaped(self):
        assert _fmt_labels({"k": 'val"ue'}) == '{k="val\\"ue"}'


class TestFormatCounter:
    def test_basic_counter(self):
        lines = _fmt_counter("requests_total", 42.0, {})
        assert "# HELP requests_total Counter metric" in lines
        assert "# TYPE requests_total counter" in lines
        assert "requests_total 42.0" in lines

    def test_counter_with_labels(self):
        lines = _fmt_counter("requests_total", 1.0, {"method": "GET"})
        assert 'requests_total{method="GET"} 1.0' in lines


class TestFormatGauge:
    def test_basic_gauge(self):
        lines = _fmt_gauge("temperature", 36.5, {})
        assert "# HELP temperature Gauge metric" in lines
        assert "# TYPE temperature gauge" in lines
        assert "temperature 36.5" in lines

    def test_gauge_with_labels(self):
        lines = _fmt_gauge("temp", 25.0, {"host": "srv1"})
        assert 'temp{host="srv1"} 25.0' in lines


class TestFormatHistogram:
    def test_has_help_and_type(self):
        lines = _fmt_histogram("latency_s", [0.1, 0.2], {})
        assert "# HELP latency_s Histogram metric" in lines
        assert "# TYPE latency_s histogram" in lines

    def test_has_bucket_lines(self):
        lines = _fmt_histogram("latency_s", [0.1, 0.2], {})
        buckets = [line for line in lines if "_bucket" in line]
        assert len(buckets) > 0

    def test_inf_bucket_present(self):
        lines = _fmt_histogram("x", [100.0], {})
        inf_line = [line for line in lines if 'le="+Inf"' in line]
        assert len(inf_line) == 1

    def test_count_and_sum(self):
        lines = _fmt_histogram("x", [1.0, 2.0, 3.0], {})
        assert "x_count 3" in lines
        assert "x_sum 6.0" in lines


class TestMetricExporter:
    def setup_method(self):
        METRICS_COLLECTOR.record("test_counter", 5.0, MetricType.COUNTER)
        METRICS_COLLECTOR.record("test_gauge", 42.0, MetricType.GAUGE)
        METRICS_COLLECTOR.record("test_histogram", 0.1, MetricType.HISTOGRAM)
        METRICS_COLLECTOR.record("test_histogram", 0.2, MetricType.HISTOGRAM)

    def teardown_method(self):
        for name in list(METRICS_COLLECTOR.metric_names()):
            METRICS_COLLECTOR._buffers.pop(name, None)

    def test_exporter_instance(self):
        assert isinstance(METRICS_EXPORTER, MetricExporter)

    def test_generate_returns_string(self):
        result = METRICS_EXPORTER.generate()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_counter(self):
        result = METRICS_EXPORTER.generate()
        assert "# TYPE test_counter counter" in result

    def test_contains_gauge(self):
        result = METRICS_EXPORTER.generate()
        assert "# TYPE test_gauge gauge" in result

    def test_contains_histogram(self):
        result = METRICS_EXPORTER.generate()
        assert "# TYPE test_histogram histogram" in result

    def test_counter_value(self):
        result = METRICS_EXPORTER.generate()
        assert "test_counter 5.0" in result

    def test_gauge_value(self):
        result = METRICS_EXPORTER.generate()
        assert "test_gauge 42.0" in result

    def test_histogram_count(self):
        result = METRICS_EXPORTER.generate()
        assert "test_histogram_count 2" in result

    def test_histogram_sum(self):
        result = METRICS_EXPORTER.generate()
        assert "test_histogram_sum 0.3" in result

    def test_exporter_with_labels(self):
        METRICS_COLLECTOR.record("labeled_metric", 10.0, MetricType.GAUGE, env="prod")
        result = METRICS_EXPORTER.generate()
        assert 'env="prod"' in result
        METRICS_COLLECTOR._buffers.pop("labeled_metric", None)

    def test_empty_exporter(self):
        from src.monitoring.metrics_collector import MetricsCollector

        empty = MetricExporter(collector=MetricsCollector())
        assert empty.generate() == ""


class TestHandleMetrics:
    def test_returns_string(self):
        METRICS_COLLECTOR.record("m", 1.0, MetricType.GAUGE)
        result = handle_metrics()
        assert isinstance(result, str)
        assert "# TYPE m gauge" in result
        METRICS_COLLECTOR._buffers.pop("m", None)

    def test_returns_prometheus_format(self):
        METRICS_COLLECTOR.record("m", 1.0, MetricType.GAUGE)
        result = handle_metrics()
        assert "m" in result
        METRICS_COLLECTOR._buffers.pop("m", None)
