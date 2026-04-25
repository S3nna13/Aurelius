"""Tests for src/monitoring/metric_exporter.py"""
from __future__ import annotations

import json

import pytest

from src.monitoring.metric_exporter import (
    METRIC_EXPORTER_REGISTRY,
    MetricExporter,
    MetricPoint,
)


@pytest.fixture
def exporter():
    return MetricExporter(max_points=100)


class TestMetricPoint:
    def test_fields(self):
        pt = MetricPoint(name="cpu", value=0.5, timestamp=1.0, labels={"env": "prod"})
        assert pt.name == "cpu"
        assert pt.value == 0.5
        assert pt.labels["env"] == "prod"


class TestAdd:
    def test_add_stores_point(self, exporter):
        pt = exporter.add("requests", 100.0)
        assert isinstance(pt, MetricPoint)
        assert pt.name == "requests"
        assert len(exporter) == 1

    def test_add_with_labels(self, exporter):
        exporter.add("latency", 42.0, {"endpoint": "/api"})
        assert exporter._points[0].labels["endpoint"] == "/api"


class TestExportPrometheus:
    def test_prometheus_format(self, exporter):
        exporter.add("requests_total", 100.0)
        exporter.add("requests_total", 200.0, {"env": "prod"})
        output = exporter.export("prometheus")
        lines = output.strip().split("\n")
        assert len(lines) == 2
        assert all("requests_total" in line for line in lines)
        assert " 200.0 " in lines[1]

    def test_prometheus_with_labels(self, exporter):
        exporter.add("cpu_util", 75.0, {"host": "server1"})
        output = exporter.export("prometheus")
        assert 'host="server1"' in output


class TestExportJSON:
    def test_json_valid(self, exporter):
        exporter.add("requests", 100.0)
        output = exporter.export("json")
        obj = json.loads(output)
        assert isinstance(obj, list)
        assert len(obj) == 1

    def test_json_all_fields(self, exporter):
        exporter.add("requests", 100.0, {"env": "prod"})
        output = exporter.export("json")
        obj = json.loads(output)
        pt = obj[0]
        assert "name" in pt
        assert "value" in pt
        assert "timestamp" in pt
        assert "labels" in pt


class TestExportCSV:
    def test_csv_has_header(self, exporter):
        exporter.add("requests", 100.0)
        output = exporter.export("csv")
        lines = output.strip().split("\n")
        header = lines[0].rstrip("\r")
        assert header == "name,value,timestamp,labels"

    def test_csv_has_rows(self, exporter):
        exporter.add("requests", 100.0)
        exporter.add("errors", 5.0)
        output = exporter.export("csv")
        lines = [line.rstrip("\r") for line in output.strip().split("\n")]
        assert len(lines) == 3


class TestExportInfluxDB:
    def test_influxdb_line_protocol(self, exporter):
        exporter.add("cpu_util", 75.0, {"host": "server1"})
        output = exporter.export("influxdb")
        lines = output.strip().split("\n")
        assert len(lines) == 1
        line = lines[0]
        assert line.startswith("cpu_util")
        assert "host=server1" in line
        assert "value=75.0" in line
        parts = line.rsplit(" ", 1)
        assert len(parts) == 2
        ts = int(parts[1])
        assert ts > 0

    def test_influxdb_ns_timestamp(self, exporter):
        exporter.add("requests", 100.0)
        output = exporter.export("influxdb")
        line = output.strip().split("\n")[0]
        ts_part = line.rsplit(" ", 1)[-1]
        assert len(ts_part) >= 18


class TestClear:
    def test_clear(self, exporter):
        exporter.add("requests", 100.0)
        exporter.clear()
        assert len(exporter) == 0


class TestLen:
    def test_len(self, exporter):
        assert len(exporter) == 0
        exporter.add("a", 1.0)
        assert len(exporter) == 1


class TestValueError:
    def test_unknown_format(self, exporter):
        with pytest.raises(ValueError):
            exporter.export("unknown_format")


class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in METRIC_EXPORTER_REGISTRY