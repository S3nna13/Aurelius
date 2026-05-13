"""Tests for RetryPolicy."""

from __future__ import annotations

import time

import pytest

from src.resilience.retry_policy import RetryPolicy


class TestRetryPolicyBasics:
    def test_success_on_first_try(self) -> None:
        rp = RetryPolicy(max_retries=2)
        assert rp.execute(lambda: 42) == 42

    def test_retries_then_succeeds(self) -> None:
        calls = []

        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("transient")
            return "ok"

        rp = RetryPolicy(max_retries=3, base_delay=0.01)
        assert rp.execute(flaky) == "ok"
        assert len(calls) == 3

    def test_exhaustion_raises(self) -> None:
        rp = RetryPolicy(max_retries=1, base_delay=0.01)
        with pytest.raises(RuntimeError):
            rp.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    def test_non_retryable_raises_immediately(self) -> None:
        rp = RetryPolicy(max_retries=5, retryable_exceptions=(ValueError,))
        with pytest.raises(RuntimeError):
            rp.execute(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    def test_delay_bounds(self) -> None:
        rp = RetryPolicy(base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
        assert rp._compute_delay(0) == 1.0
        assert rp._compute_delay(1) == 2.0
        assert rp._compute_delay(2) == 4.0
        assert rp._compute_delay(3) == 5.0  # capped

    def test_jitter_bounds(self) -> None:
        rp = RetryPolicy(base_delay=2.0, jitter=True)
        for _ in range(50):
            d = rp._compute_delay(0)
            assert 1.0 <= d <= 2.0

    def test_jitter_off(self) -> None:
        rp = RetryPolicy(base_delay=2.0, jitter=False)
        assert rp._compute_delay(0) == 2.0

    def test_backoff_timing(self) -> None:
        rp = RetryPolicy(max_retries=2, base_delay=0.05, jitter=False)
        start = time.monotonic()
        with pytest.raises(RuntimeError):
            rp.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        elapsed = time.monotonic() - start
        # 0.05 + 0.10 = 0.15
        assert elapsed >= 0.12

    def test_custom_retryable_exceptions(self) -> None:
        rp = RetryPolicy(max_retries=1, base_delay=0.01, retryable_exceptions=(TypeError,))
        calls = []

        def fn() -> None:
            calls.append(1)
            raise TypeError("t")

        with pytest.raises(TypeError):
            rp.execute(fn)
        assert len(calls) == 2
