"""Tests for DashboardExporter."""
from __future__ import annotations

import json

import pytest

from src.monitoring.dashboard_exporter import (
    DashboardExporter,
    ExportFormat,
    MetricPoint,
    DASHBOARD_EXPORTER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class TestExportFormat:
    def test_json(self):
        assert ExportFormat.JSON == "json"

    def test_prometheus(self):
        assert ExportFormat.PROMETHEUS == "prometheus"

    def test_influx(self):
        assert ExportFormat.INFLUXDB_LINE == "influxdb_line"

    def test_csv(self):
        assert ExportFormat.CSV == "csv"

    def test_four_members(self):
        assert len(ExportFormat) == 4


# ---------------------------------------------------------------------------
# MetricPoint
# ---------------------------------------------------------------------------

class TestMetricPoint:
    def test_fields(self):
        p = MetricPoint("m", 1.0, {"env": "prod"}, 100.0)
        assert p.name == "m"
        assert p.value == 1.0
        assert p.labels == {"env": "prod"}
        assert p.timestamp == 100.0

    def test_frozen(self):
        p = MetricPoint("m", 1.0, {}, 0.0)
        with pytest.raises(Exception):
            p.value = 2.0  # type: ignore


# ---------------------------------------------------------------------------
# to_json / from_json
# ---------------------------------------------------------------------------

class TestJson:
    def setup_method(self):
        self.e = DashboardExporter()

    def test_empty(self):
        assert self.e.to_json([]) == "[]"

    def test_single_point(self):
        p = MetricPoint("cpu", 0.5, {"host": "a"}, 100.0)
        out = self.e.to_json([p])
        data = json.loads(out)
        assert data[0]["name"] == "cpu"
        assert data[0]["value"] == 0.5

    def test_multiple_points(self):
        pts = [
            MetricPoint("a", 1.0, {}, 1.0),
            MetricPoint("b", 2.0, {}, 2.0),
        ]
        data = json.loads(self.e.to_json(pts))
        assert len(data) == 2

    def test_roundtrip(self):
        original = [
            MetricPoint("cpu", 0.5, {"host": "a"}, 100.0),
            MetricPoint("mem", 1024.0, {"region": "us"}, 200.0),
        ]
        s = self.e.to_json(original)
        got = self.e.from_json(s)
        assert got == original

    def test_from_json_empty(self):
        assert self.e.from_json("[]") == []


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------

class TestPrometheus:
    def setup_method(self):
        self.e = DashboardExporter()

    def test_empty(self):
        assert self.e.to_prometheus([]) == ""

    def test_no_labels(self):
        p = MetricPoint("cpu", 0.5, {}, 1.0)
        out = self.e.to_prometheus([p])
        assert out == "cpu 0.5 1000"

    def test_with_labels(self):
        p = MetricPoint("cpu", 0.5, {"host": "a"}, 1.0)
        out = self.e.to_prometheus([p])
        assert 'cpu{host="a"} 0.5 1000' == out

    def test_multiple_labels_format(self):
        p = MetricPoint("cpu", 0.5, {"host": "a", "env": "prod"}, 1.0)
        out = self.e.to_prometheus([p])
        assert out.startswith("cpu{")
        assert 'host="a"' in out
        assert 'env="prod"' in out

    def test_multiple_points(self):
        pts = [
            MetricPoint("a", 1.0, {}, 1.0),
            MetricPoint("b", 2.0, {}, 2.0),
        ]
        out = self.e.to_prometheus(pts)
        assert out.count("\n") == 1

    def test_timestamp_ms(self):
        p = MetricPoint("x", 1.0, {}, 2.5)
        out = self.e.to_prometheus([p])
        assert out.endswith("2500")


# ---------------------------------------------------------------------------
# InfluxDB
# ---------------------------------------------------------------------------

class TestInflux:
    def setup_method(self):
        self.e = DashboardExporter()

    def test_empty(self):
        assert self.e.to_influxdb([]) == ""

    def test_no_labels(self):
        p = MetricPoint("cpu", 0.5, {}, 1.0)
        out = self.e.to_influxdb([p])
        assert out == "cpu value=0.5 1000000000"

    def test_with_tags(self):
        p = MetricPoint("cpu", 0.5, {"host": "a"}, 1.0)
        out = self.e.to_influxdb([p])
        assert out.startswith("cpu,host=a ")
        assert "value=0.5" in out

    def test_timestamp_ns(self):
        p = MetricPoint("x", 1.0, {}, 1.0)
        out = self.e.to_influxdb([p])
        assert out.endswith("1000000000")

    def test_multiple_tags(self):
        p = MetricPoint("cpu", 0.5, {"a": "1", "b": "2"}, 1.0)
        out = self.e.to_influxdb([p])
        assert "a=1" in out
        assert "b=2" in out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestCSV:
    def setup_method(self):
        self.e = DashboardExporter()

    def test_empty_has_header(self):
        out = self.e.to_csv([])
        assert out.strip() == "name,value,labels_json,timestamp"

    def test_header_first(self):
        p = MetricPoint("cpu", 0.5, {}, 1.0)
        out = self.e.to_csv([p])
        first = out.splitlines()[0]
        assert first == "name,value,labels_json,timestamp"

    def test_row_contains_name(self):
        p = MetricPoint("cpu", 0.5, {"env": "prod"}, 1.0)
        out = self.e.to_csv([p])
        lines = out.splitlines()
        assert "cpu" in lines[1]

    def test_row_contains_labels_json(self):
        p = MetricPoint("cpu", 0.5, {"env": "prod"}, 1.0)
        out = self.e.to_csv([p])
        assert '"env"' in out or "'env'" in out or "env" in out

    def test_multiple_rows(self):
        pts = [
            MetricPoint("a", 1.0, {}, 1.0),
            MetricPoint("b", 2.0, {}, 2.0),
        ]
        out = self.e.to_csv(pts)
        assert len(out.splitlines()) == 3


# ---------------------------------------------------------------------------
# export dispatcher
# ---------------------------------------------------------------------------

class TestExport:
    def setup_method(self):
        self.e = DashboardExporter()
        self.pts = [MetricPoint("cpu", 0.5, {}, 1.0)]

    def test_json(self):
        out = self.e.export(self.pts, ExportFormat.JSON)
        assert json.loads(out)[0]["name"] == "cpu"

    def test_prometheus(self):
        out = self.e.export(self.pts, ExportFormat.PROMETHEUS)
        assert out.startswith("cpu")

    def test_influx(self):
        out = self.e.export(self.pts, ExportFormat.INFLUXDB_LINE)
        assert "value=0.5" in out

    def test_csv(self):
        out = self.e.export(self.pts, ExportFormat.CSV)
        assert "name,value" in out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_default(self):
        assert DASHBOARD_EXPORTER_REGISTRY["default"] is DashboardExporter
