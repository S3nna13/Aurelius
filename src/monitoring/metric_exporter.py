"""Metric exporting to Prometheus, JSON, CSV, and InfluxDB line protocol."""
from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum


class ExportFormat(StrEnum):
    PROMETHEUS = "prometheus"
    JSON = "json"
    CSV = "csv"
    INFLUXDB = "influxdb"


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricExporter:
    max_points: int = 10000
    _points: list[MetricPoint] = field(default_factory=list)

    def add(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> MetricPoint:
        point = MetricPoint(name=name, value=value, timestamp=time.time(), labels=labels or {})
        self._points.append(point)
        if len(self._points) > self.max_points:
            self._points.pop(0)
        return point

    def export(self, fmt: str) -> str:
        if fmt == "prometheus":
            return self._export_prometheus()
        elif fmt == "json":
            return self._export_json()
        elif fmt == "csv":
            return self._export_csv()
        elif fmt == "influxdb":
            return self._export_influxdb()
        raise ValueError(f"Unknown export format: {fmt!r}")

    def _export_prometheus(self) -> str:
        lines = []
        for p in self._points:
            label_str = ""
            if p.labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in p.labels.items()) + "}"
            lines.append(f"{p.name}{label_str} {p.value} {int(p.timestamp * 1000)}")
        return "\n".join(lines)

    def _export_json(self) -> str:
        data = [
            {"name": p.name, "value": p.value, "timestamp": p.timestamp, "labels": p.labels}
            for p in self._points
        ]
        return json.dumps(data, indent=2)

    def _export_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["name", "value", "timestamp", "labels"])
        for p in self._points:
            writer.writerow([p.name, p.value, p.timestamp, json.dumps(p.labels)])
        return output.getvalue()

    def _export_influxdb(self) -> str:
        lines = []
        for p in self._points:
            label_str = ""
            if p.labels:
                label_str = "," + ",".join(f"{k}={v}" for k, v in p.labels.items())
            ts_ns = int(p.timestamp * 1e9)
            lines.append(f"{p.name}{label_str} value={p.value} {ts_ns}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)


METRIC_EXPORTER_REGISTRY: dict[str, object] = {"default": MetricExporter()}