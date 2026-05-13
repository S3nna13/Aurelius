"""Tests for RateLimiter."""

from __future__ import annotations

import time

import pytest

from src.resilience.rate_limiter import RateLimiter, RateLimitExceededError


class TestRateLimiterBasics:
    def test_initial_burst(self) -> None:
        rl = RateLimiter(rate=1.0, burst=3)
        assert rl.tokens == 3.0
        assert rl.acquire() is True
        assert rl.acquire() is True
        assert rl.acquire() is True

    def test_blocking_refill(self) -> None:
        rl = RateLimiter(rate=10.0, burst=1)
        assert rl.acquire() is True
        start = time.monotonic()
        assert rl.acquire(blocking=True) is True
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08  # ~0.1s to refill one token

    def test_non_blocking_denial(self) -> None:
        rl = RateLimiter(rate=1.0, burst=1)
        assert rl.acquire(blocking=False) is True
        assert rl.acquire(blocking=False) is False

    def test_timeout_raises(self) -> None:
        rl = RateLimiter(rate=1.0, burst=1)
        rl.acquire()
        with pytest.raises(RateLimitExceededError):
            rl.acquire(blocking=True, timeout=0.01)

    def test_execute_wrapper(self) -> None:
        rl = RateLimiter(rate=100.0, burst=10)
        assert rl.execute(lambda: 99) == 99

    def test_multi_token_acquire(self) -> None:
        rl = RateLimiter(rate=100.0, burst=5)
        assert rl.acquire(tokens=5)
        assert rl.acquire(blocking=False) is False

    def test_tokens_property_updates(self) -> None:
        rl = RateLimiter(rate=10.0, burst=2)
        rl.acquire()
        before = rl.tokens
        time.sleep(0.15)
        after = rl.tokens
        assert after > before

    def test_rate_limit_exceeded_message(self) -> None:
        rl = RateLimiter(rate=1.0, burst=1)
        rl.acquire()
        with pytest.raises(RateLimitExceededError, match="timed out"):
            rl.acquire(blocking=True, timeout=0.01)
