from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from src.monitoring import MONITORING_REGISTRY
from src.monitoring.metrics_collector import MetricType

DEFAULT_BUCKETS = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
]


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape_label(v)}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


def _fmt_counter(name: str, value: float, labels: dict[str, str]) -> list[str]:
    lp = _fmt_labels(labels)
    return [
        f"# HELP {name} Counter metric",
        f"# TYPE {name} counter",
        f"{name}{lp} {value}",
    ]


def _fmt_gauge(name: str, value: float, labels: dict[str, str]) -> list[str]:
    lp = _fmt_labels(labels)
    return [
        f"# HELP {name} Gauge metric",
        f"# TYPE {name} gauge",
        f"{name}{lp} {value}",
    ]


def _fmt_histogram(name: str, samples: list[float], labels: dict[str, str]) -> list[str]:
    lp = _fmt_labels(labels)
    n = len(samples)
    total = sum(samples)
    lines = [
        f"# HELP {name} Histogram metric",
        f"# TYPE {name} histogram",
    ]
    buckets = sorted(set(DEFAULT_BUCKETS + [float("inf")]))
    for bound in buckets:
        count = sum(1 for s in samples if s <= bound)
        le = "+Inf" if bound == float("inf") else str(bound)
        bl = dict(labels)
        bl["le"] = le
        lines.append(f"{name}_bucket{_fmt_labels(bl)} {count}")
    lines.append(f"{name}_count{lp} {n}")
    lines.append(f"{name}_sum{lp} {total}")
    return lines


class MetricExporter:
    def __init__(self, collector: Any | None = None) -> None:
        self._lock = threading.Lock()
        self._collector = collector

    def generate(self) -> str:
        mc = self._collector or MONITORING_REGISTRY.get("metrics")
        if mc is None:
            return ""
        by_name: dict[str, list[Any]] = defaultdict(list)
        for name in mc.metric_names():
            for sample in mc.get_samples(name):
                by_name[name].append(sample)
        lines: list[str] = []
        for name in sorted(by_name):
            samples = by_name[name]
            metric_type = samples[0].metric_type
            by_labels: dict[tuple, list[Any]] = defaultdict(list)
            for s in samples:
                key = tuple(sorted(s.labels.items())) if s.labels else ()
                by_labels[key].append(s)
            for label_key, group in by_labels.items():
                labels = dict(label_key)
                values = [s.value for s in group]
                if metric_type == MetricType.COUNTER:
                    lines.extend(_fmt_counter(name, sum(values), labels))
                elif metric_type == MetricType.GAUGE:
                    lines.extend(_fmt_gauge(name, values[-1], labels))
                elif metric_type == MetricType.HISTOGRAM:
                    lines.extend(_fmt_histogram(name, values, labels))
        return "\n".join(lines) + "\n" if lines else ""


METRICS_EXPORTER = MetricExporter()


def handle_metrics() -> str:
    """Return Prometheus-formatted metrics. Intended for /metrics endpoints."""
    return METRICS_EXPORTER.generate()
