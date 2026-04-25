"""Aurelius monitoring: metric_exporter — exports collected metrics in multiple formats."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import List


class ExportFormat(Enum):
    PROMETHEUS = "prometheus"
    JSON = "json"
    CSV = "csv"
    INFLUX = "influx"


@dataclass(frozen=True)
class MetricPoint:
    name: str
    value: float
    labels: dict  # dict[str, str]
    timestamp: float


class MetricExporter:
    """Collects MetricPoint objects and exports them in various wire formats."""

    def __init__(self) -> None:
        self._points: List[MetricPoint] = []

    def add(self, point: MetricPoint) -> None:
        self._points.append(point)

    def export(self, fmt: ExportFormat) -> str:
        if fmt is ExportFormat.PROMETHEUS:
            return self._export_prometheus()
        if fmt is ExportFormat.JSON:
            return self._export_json()
        if fmt is ExportFormat.CSV:
            return self._export_csv()
        if fmt is ExportFormat.INFLUX:
            return self._export_influx()
        raise ValueError(f"Unknown format: {fmt}")

    def _label_str_prometheus(self, labels: dict) -> str:
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"

    def _export_prometheus(self) -> str:
        lines = []
        for p in self._points:
            ts_ms = int(p.timestamp * 1000)
            label_part = self._label_str_prometheus(p.labels)
            lines.append(f"{p.name}{label_part} {p.value} {ts_ms}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _export_json(self) -> str:
        records = [
            {"name": p.name, "value": p.value, "labels": p.labels, "timestamp": p.timestamp}
            for p in self._points
        ]
        return json.dumps(records)

    def _export_csv(self) -> str:
        rows = ["name,value,labels,timestamp"]
        for p in self._points:
            label_part = ";".join(f"{k}={v}" for k, v in p.labels.items())
            rows.append(f"{p.name},{p.value},{label_part},{p.timestamp}")
        return "\n".join(rows) + "\n"

    def _export_influx(self) -> str:
        lines = []
        for p in self._points:
            ts_ns = int(p.timestamp * 1e9)
            label_part = ",".join(f"{k}={v}" for k, v in p.labels.items())
            tag_set = f",{label_part}" if label_part else ""
            lines.append(f"{p.name}{tag_set} value={p.value} {ts_ns}")
        return "\n".join(lines) + ("\n" if lines else "")

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)


METRIC_EXPORTER_REGISTRY: dict = {"default": MetricExporter}

REGISTRY = METRIC_EXPORTER_REGISTRY
