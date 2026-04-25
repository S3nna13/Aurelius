"""Tests for metric exporter."""
from __future__ import annotations

import json

from src.monitoring.metric_exporter import (
    METRIC_EXPORTER_REGISTRY,
    ExportFormat,
    MetricExporter,
    MetricPoint,
)


def test_add():
    e = MetricExporter()
    p = e.add("accuracy", 0.95, {"model": "test"})
    if not isinstance(p, MetricPoint):
        raise ValueError(f"Expected MetricPoint, got {type(p)}")
    if p.name != "accuracy":
        raise ValueError(f"Expected name=accuracy, got {p.name}")
    if p.value != 0.95:
        raise ValueError(f"Expected value=0.95, got {p.value}")
    if p.labels.get("model") != "test":
        raise ValueError(f"Expected labels={{'model': 'test'}}, got {p.labels}")


def test_add_no_labels():
    e = MetricExporter()
    p = e.add("loss", 0.1)
    if len(p.labels) != 0:
        raise ValueError(f"Expected empty labels, got {p.labels}")


def test_export_prometheus_format():
    e = MetricExporter()
    e.add("accuracy", 0.95, {"model": "test"})
    output = e.export("prometheus")
    if not isinstance(output, str):
        raise ValueError(f"Expected str output, got {type(output)}")
    if "accuracy{" in output or 'accuracy{' in output:
        if 'model="test"' not in output:
            raise ValueError(f"Expected prometheus format with labels, got {output}")
    else:
        if "accuracy" not in output:
            raise ValueError(f"Expected 'accuracy' in prometheus output, got {output}")


def test_export_json_valid():
    e = MetricExporter()
    e.add("loss", 0.5)
    e.add("acc", 0.9)
    output = e.export("json")
    obj = json.loads(output)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list, got {type(obj)}")
    if len(obj) != 2:
        raise ValueError(f"Expected 2 points, got {len(obj)}")
    for p in obj:
        if "name" not in p:
            raise ValueError(f"Expected 'name' in point, got {p}")
        if "value" not in p:
            raise ValueError(f"Expected 'value' in point, got {p}")
        if "timestamp" not in p:
            raise ValueError(f"Expected 'timestamp' in point, got {p}")
        if "labels" not in p:
            raise ValueError(f"Expected 'labels' in point, got {p}")


def test_export_csv_header():
    e = MetricExporter()
    e.add("m1", 1.0)
    output = e.export("csv").strip()
    lines = output.split("\n")
    if lines[0] != "name,value,timestamp,labels":
        raise ValueError(f"Expected CSV header, got {lines[0]!r}")
    if len(lines) < 2:
        raise ValueError(f"Expected at least 2 rows, got {len(lines)}")
    if len(lines) < 2:
        raise ValueError(f"Expected at least 2 rows, got {len(lines)}")


def test_export_influxdb_format():
    e = MetricExporter()
    e.add("cpu_usage", 80.0, {"host": "node1"})
    output = e.export("influxdb")
    if "cpu_usage" not in output:
        raise ValueError(f"Expected 'cpu_usage' in influxdb output, got {output}")
    if "host=node1" not in output:
        raise ValueError(f"Expected 'host=node1' tag, got {output}")


def test_clear():
    e = MetricExporter()
    e.add("m1", 1.0)
    e.add("m2", 2.0)
    e.clear()
    if len(e) != 0:
        raise ValueError(f"Expected 0 points after clear, got {len(e)}")


def test_len():
    e = MetricExporter()
    if len(e) != 0:
        raise ValueError(f"Expected 0 for empty, got {len(e)}")
    e.add("m1", 1.0)
    if len(e) != 1:
        raise ValueError(f"Expected 1, got {len(e)}")
    e.add("m2", 2.0)
    if len(e) != 2:
        raise ValueError(f"Expected 2, got {len(e)}")


def test_unknown_format():
    e = MetricExporter()
    caught = False
    try:
        e.export("unknown_format")
    except ValueError:
        caught = True
    if not caught:
        raise ValueError("Expected ValueError for unknown format")


def test_registry():
    if "default" not in METRIC_EXPORTER_REGISTRY:
        raise ValueError("default not in METRIC_EXPORTER_REGISTRY")
    inst = METRIC_EXPORTER_REGISTRY["default"]
    if not isinstance(inst, MetricExporter):
        raise ValueError(f"Expected MetricExporter instance, got {type(inst)}")


def test_exportformat_strenum():
    if not isinstance(ExportFormat.PROMETHEUS, str):
        raise ValueError(f"Expected str from StrEnum, got {type(ExportFormat.PROMETHEUS)}")
    if ExportFormat.PROMETHEUS != "prometheus":
        raise ValueError(f"Expected 'prometheus', got {ExportFormat.PROMETHEUS}")