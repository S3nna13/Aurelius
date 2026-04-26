"""Tests for src.backends.circuit_breaker."""

from __future__ import annotations

import time

import pytest

from src.backends.circuit_breaker import (
    CIRCUIT_BREAKER_REGISTRY,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


def test_registry_has_default():
    assert CIRCUIT_BREAKER_REGISTRY["default"] is CircuitBreaker


def test_initial_state_closed():
    cb = CircuitBreaker("svc")
    assert cb.state is CircuitState.CLOSED


def test_initial_failure_count_zero():
    cb = CircuitBreaker("svc")
    assert cb.failure_count == 0


def test_initial_success_count_zero():
    cb = CircuitBreaker("svc")
    assert cb.success_count == 0


def test_record_failure_increments():
    cb = CircuitBreaker("svc")
    cb.record_failure()
    assert cb.failure_count == 1


def test_failure_threshold_opens_circuit():
    cfg = CircuitBreakerConfig(failure_threshold=3)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    cb.record_failure()
    state = cb.record_failure()
    assert state is CircuitState.OPEN
    assert cb.state is CircuitState.OPEN


def test_allow_request_closed_is_true():
    cb = CircuitBreaker("svc")
    assert cb.allow_request() is True


def test_allow_request_open_is_false():
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.allow_request() is False


def test_open_transitions_to_half_open_after_timeout(monkeypatch):
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.001)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    time.sleep(0.01)
    result = cb.allow_request()
    assert cb.state is CircuitState.HALF_OPEN
    assert result is True


def test_half_open_allows_request():
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.001)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    time.sleep(0.01)
    cb.allow_request()  # transitions to HALF_OPEN
    assert cb.allow_request() is True


def test_half_open_success_threshold_closes():
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.001, success_threshold=2)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    time.sleep(0.01)
    cb.allow_request()  # → HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_half_open_failure_reopens():
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.001)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    time.sleep(0.01)
    cb.allow_request()  # → HALF_OPEN
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_record_success_closed_resets_failures():
    cb = CircuitBreaker("svc")
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.failure_count == 0


def test_reset_restores_closed():
    cfg = CircuitBreakerConfig(failure_threshold=1)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    cb.reset()
    assert cb.state is CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_reset_allows_requests():
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=9999.0)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    cb.reset()
    assert cb.allow_request() is True


def test_status_dict_keys():
    cb = CircuitBreaker("my_service")
    s = cb.status()
    assert "name" in s
    assert "state" in s
    assert "failures" in s
    assert "successes" in s


def test_status_name():
    cb = CircuitBreaker("test_cb")
    assert cb.status()["name"] == "test_cb"


def test_status_state_string():
    cb = CircuitBreaker("svc")
    assert cb.status()["state"] == "closed"


def test_status_after_failure():
    cfg = CircuitBreakerConfig(failure_threshold=2)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    assert cb.status()["failures"] == 1


def test_config_frozen():
    cfg = CircuitBreakerConfig()
    with pytest.raises(Exception):
        cfg.failure_threshold = 99  # type: ignore[misc]


def test_config_defaults():
    cfg = CircuitBreakerConfig()
    assert cfg.failure_threshold == 5
    assert cfg.recovery_timeout_s == 30.0
    assert cfg.success_threshold == 2


def test_multiple_cycles():
    cfg = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=0.001, success_threshold=1)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    time.sleep(0.01)
    cb.allow_request()  # → HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitState.CLOSED
    # Second trip
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.OPEN


def test_failure_count_not_exceeded_stays_closed():
    cfg = CircuitBreakerConfig(failure_threshold=5)
    cb = CircuitBreaker("svc", cfg)
    for _ in range(4):
        cb.record_failure()
    assert cb.state is CircuitState.CLOSED


def test_allow_request_half_open_true():
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.001)
    cb = CircuitBreaker("svc", cfg)
    cb.record_failure()
    time.sleep(0.01)
    cb.allow_request()  # transitions to HALF_OPEN
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.allow_request() is True
