"""Request metrics collector for the API server.

Tracks request counts, latencies, status codes, and active connections.
Exposes a /metrics endpoint for Prometheus scraping.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

# Rate limiter backend indicator for Prometheus (0=memory, 1=redis)
_RATE_LIMITER_BACKEND_NAME = "memory"


class MetricsCollector:
    """Thread-safe metrics collection for the stdlib HTTP server."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count: int = 0
        self._active_connections: int = 0
        self._status_codes: dict[int, int] = defaultdict(int)
        self._method_counts: dict[str, int] = defaultdict(int)
        self._path_counts: dict[str, int] = defaultdict(int)
        self._latencies: list[float] = []
        self._start_time: float = time.time()
        self._errors: dict[str, int] = defaultdict(int)
        self._rate_limit_rejections: int = 0
        self._validation_failures: int = 0

    def record_request(
        self,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
        error_type: str | None = None,
    ) -> None:
        with self._lock:
            self._request_count += 1
            self._status_codes[status] += 1
            self._method_counts[method] += 1
            self._path_counts[path] += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > 10000:
                self._latencies = self._latencies[-5000:]
            if error_type:
                self._errors[error_type] += 1

    def connection_opened(self) -> None:
        with self._lock:
            self._active_connections += 1

    def connection_closed(self) -> None:
        with self._lock:
            self._active_connections -= 1

    def set_rate_limiter_backend(self, name: str) -> None:
        """Update the rate limiter backend indicator (e.g., 'memory' or 'redis')."""
        with self._lock:
            global _RATE_LIMITER_BACKEND_NAME
            _RATE_LIMITER_BACKEND_NAME = name

    def record_rate_limit_rejection(self) -> None:
        with self._lock:
            self._rate_limit_rejections += 1

    def record_validation_failure(self) -> None:
        with self._lock:
            self._validation_failures += 1

    def snapshot(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            latencies = sorted(self._latencies) if self._latencies else [0.0]
            n = len(latencies)
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self._request_count,
                "active_connections": self._active_connections,
                "requests_per_second": round(self._request_count / max(uptime, 1), 2),
                "status_codes": dict(self._status_codes),
                "methods": dict(self._method_counts),
                "paths": dict(self._path_counts),
                "latency_ms": {
                    "min": round(latencies[0], 2),
                    "p50": round(latencies[n // 2], 2),
                    "p95": round(latencies[int(n * 0.95)], 2),
                    "p99": round(latencies[int(n * 0.99)], 2),
                    "max": round(latencies[-1], 2),
                    "avg": round(sum(latencies) / max(n, 1), 2),
                },
                "errors": dict(self._errors),
            "rate_limit_rejections": self._rate_limit_rejections,
            "validation_failures": self._validation_failures,
            }

    def prometheus_text(self) -> str:
        """Return Prometheus-format text."""
        snap = self.snapshot()
        lines = [
            "# HELP aurelius_requests_total Total HTTP requests",
            "# TYPE aurelius_requests_total counter",
            f"aurelius_requests_total {snap['total_requests']}",
            "",
            "# HELP aurelius_active_connections Active connections",
            "# TYPE aurelius_active_connections gauge",
            f"aurelius_active_connections {snap['active_connections']}",
            "",
            "# HELP aurelius_request_duration_ms Request latency",
            "# TYPE aurelius_request_duration_ms gauge",
            f'aurelius_request_duration_ms{{quantile="p50"}} {snap["latency_ms"]["p50"]}',
            f'aurelius_request_duration_ms{{quantile="p95"}} {snap["latency_ms"]["p95"]}',
            f'aurelius_request_duration_ms{{quantile="p99"}} {snap["latency_ms"]["p99"]}',
            "",
            "# HELP aurelius_uptime_seconds Server uptime",
            "# TYPE aurelius_uptime_seconds gauge",
            f"aurelius_uptime_seconds {snap['uptime_seconds']}",
            "",
            "# HELP aurelius_requests_per_second Requests per second",
            "# TYPE aurelius_requests_per_second gauge",
            f"aurelius_requests_per_second {snap['requests_per_second']}",

            "",
            "# HELP aurelius_rate_limit_rejected_total Total requests rejected by rate limiter",
            "# TYPE aurelius_rate_limit_rejected_total counter",
            f"aurelius_rate_limit_rejected_total {snap['rate_limit_rejections']}",
            "",
            "# HELP aurelius_validation_failures_total Total parameter validation failures",
            "# TYPE aurelius_validation_failures_total counter",
            f"aurelius_validation_failures_total {snap['validation_failures']}",
            "",
            "# HELP aurelius_rate_limiter_backend Rate limiter backend (0=memory, 1=redis)",
            "# TYPE aurelius_rate_limiter_backend gauge",
            f"aurelius_rate_limiter_backend {1 if _RATE_LIMITER_BACKEND_NAME == 'redis' else 0}",
        ]
        for code, count in snap["status_codes"].items():
            lines.append(f'aurelius_http_status_total{{code="{code}"}} {count}')
        return "\n".join(lines) + "\n"


METRICS = MetricsCollector()
