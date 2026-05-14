"""Tests for CircuitBreaker."""

from __future__ import annotations

import time

import pytest

from src.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


def _fail() -> None:
    raise RuntimeError("boom")


def _ok() -> str:
    return "ok"


class TestCircuitBreakerBasics:
    def test_closed_allows_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.call(_ok) == "ok"
        assert cb.state == "closed"

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        assert cb.state == "open"
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(_ok)

    def test_half_open_then_closed(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.05, success_threshold_half_open=1
        )
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        assert cb.state == "open"
        time.sleep(0.06)
        assert cb.call(_ok) == "ok"
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        time.sleep(0.06)
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        assert cb.state == "open"

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == "closed"

    def test_half_open_max_calls(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            half_open_max_calls=1,
            success_threshold_half_open=2,
        )
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        time.sleep(0.06)
        # first half-open call
        assert cb.call(_ok) == "ok"
        # half-open limit reached
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(_ok)

    def test_success_threshold_half_open(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            half_open_max_calls=3,
            success_threshold_half_open=2,
        )
        with pytest.raises(RuntimeError):
            cb.call(_fail)
        time.sleep(0.06)
        cb.call(_ok)
        assert cb.state == "half_open"
        cb.call(_ok)
        assert cb.state == "closed"

    def test_manual_record(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_name_in_error(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, name="test-cb")
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpenError, match="test-cb"):
            cb.call(_ok)

    def test_str_state(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
