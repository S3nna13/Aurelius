"""Tests for HealthChecker (~45 tests)."""

from __future__ import annotations

import pytest

from src.monitoring.health_checker import (
    HEALTH_CHECKER,
    HealthCheck,
    HealthChecker,
    HealthStatus,
)

# ---------------------------------------------------------------------------
# HealthStatus enum
# ---------------------------------------------------------------------------


class TestHealthStatusEnum:
    def test_healthy(self):
        assert HealthStatus.HEALTHY == "healthy"

    def test_degraded(self):
        assert HealthStatus.DEGRADED == "degraded"

    def test_unhealthy(self):
        assert HealthStatus.UNHEALTHY == "unhealthy"

    def test_unknown(self):
        assert HealthStatus.UNKNOWN == "unknown"

    def test_four_members(self):
        assert len(HealthStatus) == 4

    def test_is_str(self):
        assert isinstance(HealthStatus.HEALTHY, str)


# ---------------------------------------------------------------------------
# HealthCheck dataclass
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_name_field(self):
        hc = HealthCheck(name="db", status=HealthStatus.HEALTHY)
        assert hc.name == "db"

    def test_status_field(self):
        hc = HealthCheck(name="db", status=HealthStatus.DEGRADED)
        assert hc.status == HealthStatus.DEGRADED

    def test_message_default_empty(self):
        hc = HealthCheck(name="db", status=HealthStatus.HEALTHY)
        assert hc.message == ""

    def test_message_set(self):
        hc = HealthCheck(name="db", status=HealthStatus.UNHEALTHY, message="conn refused")
        assert hc.message == "conn refused"

    def test_latency_default_zero(self):
        hc = HealthCheck(name="db", status=HealthStatus.HEALTHY)
        assert hc.latency_ms == 0.0

    def test_latency_set(self):
        hc = HealthCheck(name="db", status=HealthStatus.HEALTHY, latency_ms=12.5)
        assert hc.latency_ms == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# HealthChecker.register + run
# ---------------------------------------------------------------------------


class TestRegisterAndRun:
    def setup_method(self):
        self.hc = HealthChecker()

    def test_register_and_run_roundtrip(self):
        expected = HealthCheck(name="cache", status=HealthStatus.HEALTHY)
        self.hc.register("cache", lambda: expected)
        result = self.hc.run("cache")
        assert result is expected

    def test_run_returns_check_fn_result(self):
        self.hc.register("svc", lambda: HealthCheck("svc", HealthStatus.DEGRADED, "slow"))
        result = self.hc.run("svc")
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "slow"

    def test_run_exception_returns_unhealthy(self):
        def bad():
            raise RuntimeError("boom")

        self.hc.register("bad", bad)
        result = self.hc.run("bad")
        assert result.status == HealthStatus.UNHEALTHY

    def test_run_exception_message_contains_error(self):
        def bad():
            raise RuntimeError("connection refused")

        self.hc.register("bad", bad)
        result = self.hc.run("bad")
        assert "connection refused" in result.message

    def test_run_exception_name_preserved(self):
        def bad():
            raise ValueError("oops")

        self.hc.register("mycomp", bad)
        result = self.hc.run("mycomp")
        assert result.name == "mycomp"

    def test_register_overwrites(self):
        self.hc.register("x", lambda: HealthCheck("x", HealthStatus.HEALTHY))
        self.hc.register("x", lambda: HealthCheck("x", HealthStatus.UNHEALTHY))
        result = self.hc.run("x")
        assert result.status == HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# HealthChecker.run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    def setup_method(self):
        self.hc = HealthChecker()

    def test_run_all_returns_dict(self):
        result = self.hc.run_all()
        assert isinstance(result, dict)

    def test_run_all_empty(self):
        assert self.hc.run_all() == {}

    def test_run_all_runs_all_checks(self):
        self.hc.register("a", lambda: HealthCheck("a", HealthStatus.HEALTHY))
        self.hc.register("b", lambda: HealthCheck("b", HealthStatus.DEGRADED))
        result = self.hc.run_all()
        assert "a" in result
        assert "b" in result

    def test_run_all_correct_statuses(self):
        self.hc.register("a", lambda: HealthCheck("a", HealthStatus.HEALTHY))
        self.hc.register("b", lambda: HealthCheck("b", HealthStatus.DEGRADED))
        result = self.hc.run_all()
        assert result["a"].status == HealthStatus.HEALTHY
        assert result["b"].status == HealthStatus.DEGRADED

    def test_run_all_catches_exceptions(self):
        def bad():
            raise RuntimeError("fail")

        self.hc.register("bad", bad)
        result = self.hc.run_all()
        assert result["bad"].status == HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# HealthChecker.aggregate_status
# ---------------------------------------------------------------------------


class TestAggregateStatus:
    def setup_method(self):
        self.hc = HealthChecker()

    def _checks(self, **statuses) -> dict:
        return {name: HealthCheck(name=name, status=st) for name, st in statuses.items()}

    def test_all_healthy(self):
        checks = self._checks(a=HealthStatus.HEALTHY, b=HealthStatus.HEALTHY)
        assert self.hc.aggregate_status(checks) == HealthStatus.HEALTHY

    def test_any_unhealthy(self):
        checks = self._checks(a=HealthStatus.HEALTHY, b=HealthStatus.UNHEALTHY)
        assert self.hc.aggregate_status(checks) == HealthStatus.UNHEALTHY

    def test_unhealthy_overrides_degraded(self):
        checks = self._checks(a=HealthStatus.DEGRADED, b=HealthStatus.UNHEALTHY)
        assert self.hc.aggregate_status(checks) == HealthStatus.UNHEALTHY

    def test_any_degraded_no_unhealthy(self):
        checks = self._checks(a=HealthStatus.HEALTHY, b=HealthStatus.DEGRADED)
        assert self.hc.aggregate_status(checks) == HealthStatus.DEGRADED

    def test_empty_checks_unknown(self):
        assert self.hc.aggregate_status({}) == HealthStatus.UNKNOWN

    def test_single_healthy(self):
        checks = self._checks(a=HealthStatus.HEALTHY)
        assert self.hc.aggregate_status(checks) == HealthStatus.HEALTHY

    def test_single_unhealthy(self):
        checks = self._checks(a=HealthStatus.UNHEALTHY)
        assert self.hc.aggregate_status(checks) == HealthStatus.UNHEALTHY

    def test_single_degraded(self):
        checks = self._checks(a=HealthStatus.DEGRADED)
        assert self.hc.aggregate_status(checks) == HealthStatus.DEGRADED


# ---------------------------------------------------------------------------
# HealthChecker.readiness
# ---------------------------------------------------------------------------


class TestReadiness:
    def setup_method(self):
        self.hc = HealthChecker()

    def test_readiness_has_status_key(self):
        result = self.hc.readiness()
        assert "status" in result

    def test_readiness_has_checks_key(self):
        result = self.hc.readiness()
        assert "checks" in result

    def test_readiness_status_is_string(self):
        result = self.hc.readiness()
        assert isinstance(result["status"], str)

    def test_readiness_checks_is_dict(self):
        result = self.hc.readiness()
        assert isinstance(result["checks"], dict)

    def test_readiness_check_entry_structure(self):
        self.hc.register("db", lambda: HealthCheck("db", HealthStatus.HEALTHY))
        result = self.hc.readiness()
        assert "status" in result["checks"]["db"]
        assert "message" in result["checks"]["db"]

    def test_readiness_reflects_aggregate(self):
        self.hc.register("db", lambda: HealthCheck("db", HealthStatus.UNHEALTHY))
        result = self.hc.readiness()
        assert result["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# HealthChecker.liveness
# ---------------------------------------------------------------------------


class TestLiveness:
    def setup_method(self):
        self.hc = HealthChecker()

    def test_liveness_returns_alive_true(self):
        result = self.hc.liveness()
        assert result == {"alive": True}

    def test_liveness_with_checks_alive(self):
        self.hc.register("db", lambda: HealthCheck("db", HealthStatus.HEALTHY))
        result = self.hc.liveness()
        assert result["alive"] is True

    def test_liveness_with_failing_checks_still_alive(self):
        # run_all catches exceptions internally, so liveness should still return alive
        def bad():
            raise RuntimeError("fail")

        self.hc.register("bad", bad)
        result = self.hc.liveness()
        assert result["alive"] is True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_health_checker_exists(self):
        assert HEALTH_CHECKER is not None
        assert isinstance(HEALTH_CHECKER, HealthChecker)
