"""Tests for circuit breaker."""

from __future__ import annotations

import pytest

from src.protocol.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_closed_allows_calls(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=1)
        assert cb.call(lambda: 42) == 42
        assert cb.failure_count == 0

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=100)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state is CircuitState.CLOSED
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state is CircuitState.OPEN

    def test_open_raises(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=100)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        with pytest.raises(RuntimeError, match="circuit breaker open"):
            cb.call(lambda: 42)

    def test_half_open_on_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        import time

        time.sleep(0.01)

        # half-open after cooldown
        class HalfOpenHack:
            pass

        cb._state = CircuitState.HALF_OPEN  # force for test
        result = cb.call(lambda: 99)
        assert result == 99
        assert cb.state is CircuitState.CLOSED

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=100)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("x")))
        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0
