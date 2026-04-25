"""Dashboard exporter: export metrics to multiple dashboard formats."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from enum import Enum


class ExportFormat(str, Enum):
    JSON = "json"
    PROMETHEUS = "prometheus"
    INFLUXDB_LINE = "influxdb_line"
    CSV = "csv"


@dataclass(frozen=True)
class MetricPoint:
    name: str
    value: float
    labels: dict[str, str]
    timestamp: float


class DashboardExporter:
    def to_json(self, points: list[MetricPoint]) -> str:
        payload = [
            {
                "name": p.name,
                "value": p.value,
                "labels": dict(p.labels),
                "timestamp": p.timestamp,
            }
            for p in points
        ]
        return json.dumps(payload)

    def to_prometheus(self, points: list[MetricPoint]) -> str:
        lines: list[str] = []
        for p in points:
            if p.labels:
                label_str = (
                    "{"
                    + ",".join(
                        f'{k}="{_escape_prom(str(v))}"' for k, v in p.labels.items()
                    )
                    + "}"
                )
            else:
                label_str = ""
            ts_ms = int(p.timestamp * 1000)
            lines.append(f"{p.name}{label_str} {p.value} {ts_ms}")
        return "\n".join(lines)

    def to_influxdb(self, points: list[MetricPoint]) -> str:
        lines: list[str] = []
        for p in points:
            if p.labels:
                tag_set = ",".join(
                    f"{_escape_influx(k)}={_escape_influx(str(v))}"
                    for k, v in p.labels.items()
                )
                prefix = f"{p.name},{tag_set}"
            else:
                prefix = p.name
            ts_ns = int(p.timestamp * 1_000_000_000)
            lines.append(f"{prefix} value={p.value} {ts_ns}")
        return "\n".join(lines)

    def to_csv(self, points: list[MetricPoint]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["name", "value", "labels_json", "timestamp"])
        for p in points:
            writer.writerow(
                [p.name, p.value, json.dumps(dict(p.labels)), p.timestamp]
            )
        return buf.getvalue()

    def export(self, points: list[MetricPoint], fmt: ExportFormat) -> str:
        if fmt == ExportFormat.JSON:
            return self.to_json(points)
        if fmt == ExportFormat.PROMETHEUS:
            return self.to_prometheus(points)
        if fmt == ExportFormat.INFLUXDB_LINE:
            return self.to_influxdb(points)
        if fmt == ExportFormat.CSV:
            return self.to_csv(points)
        raise ValueError(f"Unknown format: {fmt}")

    def from_json(self, json_str: str) -> list[MetricPoint]:
        data = json.loads(json_str)
        return [
            MetricPoint(
                name=d["name"],
                value=float(d["value"]),
                labels=dict(d.get("labels", {})),
                timestamp=float(d["timestamp"]),
            )
            for d in data
        ]


def _escape_prom(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_influx(v: str) -> str:
    return v.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


DASHBOARD_EXPORTER_REGISTRY = {"default": DashboardExporter}
