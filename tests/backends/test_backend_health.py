from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from src.backends.backend_health import (
    BackendHealthChecker,
    HealthReport,
    HealthStatus,
    BACKEND_HEALTH_REGISTRY,
)


def _make_backend(health_method: str = "health", returns: bool = True, raises: Exception | None = None):
    b = MagicMock(spec=[health_method])
    method = getattr(b, health_method)
    if raises is not None:
        method.side_effect = raises
    else:
        method.return_value = returns
    return b


class TestHealthStatus(unittest.TestCase):
    def test_status_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"


class TestHealthReport(unittest.TestCase):
    def test_frozen(self):
        r = HealthReport(backend_name="x", status=HealthStatus.HEALTHY, latency_ms=1.0)
        with self.assertRaises(Exception):
            r.backend_name = "y"

    def test_error_defaults_none(self):
        r = HealthReport(backend_name="x", status=HealthStatus.HEALTHY, latency_ms=1.0)
        assert r.error is None


class TestBackendHealthCheckerCheck(unittest.TestCase):
    def test_check_healthy_via_health_method(self):
        backend = _make_backend("health", returns=True)
        checker = BackendHealthChecker()
        report = checker.check("mybackend", backend)
        assert report.backend_name == "mybackend"
        assert report.status == HealthStatus.HEALTHY
        assert report.error is None

    def test_check_unhealthy_via_health_method(self):
        backend = _make_backend("health", returns=False)
        checker = BackendHealthChecker()
        report = checker.check("mybackend", backend)
        assert report.status == HealthStatus.UNHEALTHY

    def test_check_healthy_via_is_available(self):
        backend = _make_backend("is_available", returns=True)
        checker = BackendHealthChecker()
        report = checker.check("b2", backend)
        assert report.status == HealthStatus.HEALTHY

    def test_check_unhealthy_via_is_available(self):
        backend = _make_backend("is_available", returns=False)
        checker = BackendHealthChecker()
        report = checker.check("b2", backend)
        assert report.status == HealthStatus.UNHEALTHY

    def test_check_measures_latency(self):
        def slow_health():
            time.sleep(0.01)
            return True

        backend = MagicMock()
        backend.health.side_effect = slow_health
        checker = BackendHealthChecker()
        report = checker.check("slow", backend)
        assert report.latency_ms >= 10.0

    def test_check_unknown_when_no_health_method(self):
        backend = object()
        checker = BackendHealthChecker()
        report = checker.check("no_method", backend)
        assert report.status == HealthStatus.UNKNOWN
        assert report.error is not None

    def test_check_unhealthy_on_exception(self):
        backend = _make_backend("health", raises=RuntimeError("boom"))
        checker = BackendHealthChecker()
        report = checker.check("err_backend", backend)
        assert report.status == HealthStatus.UNHEALTHY
        assert "boom" in report.error


class TestBackendHealthCheckerCheckAll(unittest.TestCase):
    def test_check_all_returns_all_reports(self):
        b1 = _make_backend("health", returns=True)
        b2 = _make_backend("health", returns=False)
        checker = BackendHealthChecker({"a": b1, "b": b2})
        reports = checker.check_all()
        assert len(reports) == 2
        names = {r.backend_name for r in reports}
        assert names == {"a", "b"}

    def test_check_all_empty(self):
        checker = BackendHealthChecker()
        assert checker.check_all() == []


class TestBackendHealthCheckerRegister(unittest.TestCase):
    def test_register_adds_backend(self):
        checker = BackendHealthChecker()
        backend = _make_backend("health", returns=True)
        checker.register("new", backend)
        reports = checker.check_all()
        assert len(reports) == 1
        assert reports[0].backend_name == "new"


class TestBackendHealthCheckerHealthyBackends(unittest.TestCase):
    def test_healthy_backends_filters(self):
        b1 = _make_backend("health", returns=True)
        b2 = _make_backend("health", returns=False)
        b3 = _make_backend("health", returns=True)
        checker = BackendHealthChecker({"a": b1, "b": b2, "c": b3})
        healthy = checker.healthy_backends()
        assert set(healthy) == {"a", "c"}

    def test_healthy_backends_empty_when_all_down(self):
        b1 = _make_backend("health", returns=False)
        checker = BackendHealthChecker({"x": b1})
        assert checker.healthy_backends() == []


class TestP99LatencyMs(unittest.TestCase):
    def _make_reports(self, latencies: list[float]) -> list[HealthReport]:
        return [
            HealthReport(backend_name=f"b{i}", status=HealthStatus.HEALTHY, latency_ms=lat)
            for i, lat in enumerate(latencies)
        ]

    def test_p99_single(self):
        checker = BackendHealthChecker()
        reports = self._make_reports([42.0])
        assert checker.p99_latency_ms(reports) == 42.0

    def test_p99_multiple(self):
        # Patch time.monotonic so this test is completely immune to wall-clock
        # timing variance when the full suite runs under load.  The p99
        # calculation is pure math on the latency_ms values that _make_reports
        # injects directly; no real clock is consulted, but the patch makes
        # that contract explicit and protects against future refactors.
        with patch("src.backends.backend_health.time.monotonic", return_value=0.0):
            checker = BackendHealthChecker()
            latencies = list(range(1, 101))
            reports = self._make_reports(latencies)
            p99 = checker.p99_latency_ms(reports)
            assert p99 == 99

    def test_p99_empty(self):
        checker = BackendHealthChecker()
        assert checker.p99_latency_ms([]) == 0.0


class TestBackendHealthRegistry(unittest.TestCase):
    def test_registry_has_default_key(self):
        assert "default" in BACKEND_HEALTH_REGISTRY

    def test_registry_value_is_checker_class(self):
        assert BACKEND_HEALTH_REGISTRY["default"] is BackendHealthChecker


if __name__ == "__main__":
    unittest.main()
