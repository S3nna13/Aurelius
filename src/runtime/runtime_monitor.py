from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: HealthStatus
    latency_ms: float
    details: str = ""


@dataclass(frozen=True)
class RuntimeMetrics:
    timestamp_s: float
    request_count: int
    error_count: int
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


class RuntimeMonitor:
    def __init__(self) -> None:
        self._latencies: deque[float] = deque(maxlen=1000)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self._first_record_s: float | None = None

    def record_request(self, latency_ms: float, success: bool = True) -> None:
        t0 = time.perf_counter()
        if self._first_record_s is None:
            self._first_record_s = t0
        self._latencies.append(latency_ms)
        self.total_requests += 1
        if not success:
            self.total_errors += 1

    def check_latency(self, threshold_ms: float = 100.0) -> HealthCheck:
        t0 = time.perf_counter()
        p99 = _percentile(list(self._latencies), 99.0)
        if not self._latencies:
            status = HealthStatus.UNKNOWN
        elif p99 <= threshold_ms:
            status = HealthStatus.HEALTHY
        elif p99 <= threshold_ms * 2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        lat = (time.perf_counter() - t0) * 1000.0
        return HealthCheck(
            name="latency",
            status=status,
            latency_ms=lat,
            details=f"p99={p99:.2f}ms threshold={threshold_ms:.2f}ms",
        )

    def check_error_rate(self, threshold_pct: float = 5.0) -> HealthCheck:
        t0 = time.perf_counter()
        if self.total_requests == 0:
            status = HealthStatus.UNKNOWN
            err_pct = 0.0
        else:
            err_pct = (self.total_errors / self.total_requests) * 100.0
            if err_pct <= threshold_pct:
                status = HealthStatus.HEALTHY
            elif err_pct <= threshold_pct * 2:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
        lat = (time.perf_counter() - t0) * 1000.0
        return HealthCheck(
            name="error_rate",
            status=status,
            latency_ms=lat,
            details=f"errors={err_pct:.2f}% threshold={threshold_pct:.2f}%",
        )

    def health_summary(self) -> list[HealthCheck]:
        lat_check = self.check_latency()
        err_check = self.check_error_rate()
        if lat_check.status == HealthStatus.HEALTHY and err_check.status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.HEALTHY
        elif HealthStatus.UNHEALTHY in (lat_check.status, err_check.status):
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in (lat_check.status, err_check.status):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN
        overall = HealthCheck(
            name="overall",
            status=overall_status,
            latency_ms=lat_check.latency_ms + err_check.latency_ms,
            details="composite",
        )
        return [lat_check, err_check, overall]

    def snapshot(self) -> RuntimeMetrics:
        now = time.perf_counter()
        values = list(self._latencies)
        p50 = _percentile(values, 50.0)
        p99 = _percentile(values, 99.0)
        if self._first_record_s is None or now <= self._first_record_s:
            rps = 0.0
        else:
            elapsed = now - self._first_record_s
            rps = self.total_requests / elapsed if elapsed > 0 else 0.0
        return RuntimeMetrics(
            timestamp_s=now,
            request_count=self.total_requests,
            error_count=self.total_errors,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            throughput_rps=rps,
        )

    def reset(self) -> None:
        self._latencies.clear()
        self.total_requests = 0
        self.total_errors = 0
        self._first_record_s = None


RUNTIME_MONITOR_REGISTRY: dict[str, type[RuntimeMonitor]] = {"default": RuntimeMonitor}
