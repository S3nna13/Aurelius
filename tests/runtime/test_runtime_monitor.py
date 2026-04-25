import time

import pytest

from src.runtime.runtime_monitor import (
    RUNTIME_MONITOR_REGISTRY,
    HealthCheck,
    HealthStatus,
    RuntimeMetrics,
    RuntimeMonitor,
)


class TestHealthStatus:
    def test_members(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheck:
    def test_fields(self):
        hc = HealthCheck(name="x", status=HealthStatus.HEALTHY, latency_ms=1.0, details="ok")
        assert hc.name == "x"
        assert hc.status is HealthStatus.HEALTHY
        assert hc.latency_ms == 1.0
        assert hc.details == "ok"

    def test_default_details(self):
        hc = HealthCheck(name="x", status=HealthStatus.HEALTHY, latency_ms=0.0)
        assert hc.details == ""

    def test_frozen(self):
        hc = HealthCheck(name="x", status=HealthStatus.HEALTHY, latency_ms=0.0)
        with pytest.raises(Exception):
            hc.name = "y"  # type: ignore[misc]


class TestRuntimeMetrics:
    def test_fields(self):
        m = RuntimeMetrics(1.0, 10, 1, 5.0, 50.0, 20.0)
        assert m.timestamp_s == 1.0
        assert m.request_count == 10
        assert m.error_count == 1
        assert m.p50_latency_ms == 5.0
        assert m.p99_latency_ms == 50.0
        assert m.throughput_rps == 20.0


class TestRuntimeMonitor:
    def test_initial_state(self):
        m = RuntimeMonitor()
        assert m.total_requests == 0
        assert m.total_errors == 0

    def test_record_request_accumulates(self):
        m = RuntimeMonitor()
        for i in range(5):
            m.record_request(10.0 + i)
        assert m.total_requests == 5
        assert m.total_errors == 0

    def test_record_errors(self):
        m = RuntimeMonitor()
        m.record_request(10.0, success=True)
        m.record_request(20.0, success=False)
        m.record_request(30.0, success=False)
        assert m.total_requests == 3
        assert m.total_errors == 2

    def test_check_latency_unknown_empty(self):
        m = RuntimeMonitor()
        hc = m.check_latency()
        assert hc.status is HealthStatus.UNKNOWN

    def test_check_latency_healthy(self):
        m = RuntimeMonitor()
        for _ in range(100):
            m.record_request(5.0)
        hc = m.check_latency(threshold_ms=100.0)
        assert hc.status is HealthStatus.HEALTHY

    def test_check_latency_degraded(self):
        m = RuntimeMonitor()
        for _ in range(100):
            m.record_request(150.0)
        hc = m.check_latency(threshold_ms=100.0)
        assert hc.status is HealthStatus.DEGRADED

    def test_check_latency_unhealthy(self):
        m = RuntimeMonitor()
        for _ in range(100):
            m.record_request(500.0)
        hc = m.check_latency(threshold_ms=100.0)
        assert hc.status is HealthStatus.UNHEALTHY

    def test_check_latency_name(self):
        m = RuntimeMonitor()
        m.record_request(5.0)
        hc = m.check_latency()
        assert hc.name == "latency"

    def test_check_error_rate_unknown_empty(self):
        m = RuntimeMonitor()
        hc = m.check_error_rate()
        assert hc.status is HealthStatus.UNKNOWN

    def test_check_error_rate_healthy(self):
        m = RuntimeMonitor()
        for _ in range(100):
            m.record_request(1.0, success=True)
        hc = m.check_error_rate(threshold_pct=5.0)
        assert hc.status is HealthStatus.HEALTHY

    def test_check_error_rate_degraded(self):
        m = RuntimeMonitor()
        for i in range(100):
            m.record_request(1.0, success=i >= 8)
        hc = m.check_error_rate(threshold_pct=5.0)
        assert hc.status is HealthStatus.DEGRADED

    def test_check_error_rate_unhealthy(self):
        m = RuntimeMonitor()
        for i in range(100):
            m.record_request(1.0, success=i >= 50)
        hc = m.check_error_rate(threshold_pct=5.0)
        assert hc.status is HealthStatus.UNHEALTHY

    def test_check_error_rate_name(self):
        m = RuntimeMonitor()
        m.record_request(1.0)
        hc = m.check_error_rate()
        assert hc.name == "error_rate"

    def test_health_summary_length(self):
        m = RuntimeMonitor()
        m.record_request(5.0)
        summary = m.health_summary()
        assert len(summary) == 3

    def test_health_summary_all_healthy(self):
        m = RuntimeMonitor()
        for _ in range(50):
            m.record_request(5.0, success=True)
        summary = m.health_summary()
        overall = summary[-1]
        assert overall.name == "overall"
        assert overall.status is HealthStatus.HEALTHY

    def test_health_summary_degraded_if_latency_degraded(self):
        m = RuntimeMonitor()
        for _ in range(100):
            m.record_request(150.0, success=True)
        summary = m.health_summary()
        assert summary[-1].status is HealthStatus.DEGRADED

    def test_health_summary_unhealthy_wins(self):
        m = RuntimeMonitor()
        for _ in range(100):
            m.record_request(5000.0, success=True)
        summary = m.health_summary()
        assert summary[-1].status is HealthStatus.UNHEALTHY

    def test_snapshot_fields(self):
        m = RuntimeMonitor()
        for i in range(10):
            m.record_request(float(i))
        snap = m.snapshot()
        assert isinstance(snap, RuntimeMetrics)
        assert snap.request_count == 10
        assert snap.error_count == 0

    def test_snapshot_percentiles(self):
        m = RuntimeMonitor()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.record_request(v)
        snap = m.snapshot()
        assert 1.0 <= snap.p50_latency_ms <= 5.0
        assert 1.0 <= snap.p99_latency_ms <= 5.0
        assert snap.p99_latency_ms >= snap.p50_latency_ms

    def test_snapshot_throughput_positive(self):
        m = RuntimeMonitor()
        m.record_request(1.0)
        time.sleep(0.01)
        m.record_request(1.0)
        snap = m.snapshot()
        assert snap.throughput_rps >= 0.0

    def test_snapshot_empty(self):
        m = RuntimeMonitor()
        snap = m.snapshot()
        assert snap.request_count == 0
        assert snap.throughput_rps == 0.0

    def test_reset_clears_state(self):
        m = RuntimeMonitor()
        for _ in range(10):
            m.record_request(5.0, success=False)
        m.reset()
        assert m.total_requests == 0
        assert m.total_errors == 0
        assert m.snapshot().p50_latency_ms == 0.0

    def test_deque_maxlen(self):
        m = RuntimeMonitor()
        for _ in range(1500):
            m.record_request(1.0)
        snap = m.snapshot()
        assert snap.request_count == 1500
        # internal deque capped at 1000
        assert len(m._latencies) == 1000

    def test_check_latency_details(self):
        m = RuntimeMonitor()
        m.record_request(42.0)
        hc = m.check_latency(threshold_ms=100.0)
        assert "p99" in hc.details

    def test_check_error_details(self):
        m = RuntimeMonitor()
        m.record_request(1.0, success=False)
        hc = m.check_error_rate()
        assert "errors" in hc.details


class TestRegistry:
    def test_default_present(self):
        assert "default" in RUNTIME_MONITOR_REGISTRY

    def test_default_constructs(self):
        cls = RUNTIME_MONITOR_REGISTRY["default"]
        assert isinstance(cls(), RuntimeMonitor)
