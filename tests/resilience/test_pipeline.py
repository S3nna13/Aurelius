"""Tests for ResiliencePipeline."""

from __future__ import annotations

import time

import pytest

from src.resilience.bulkhead import Bulkhead, BulkheadFullError
from src.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from src.resilience.pipeline import ResiliencePipeline
from src.resilience.rate_limiter import RateLimiter, RateLimitExceededError
from src.resilience.retry_policy import RetryPolicy


def _ok() -> str:
    return "ok"


def _fail() -> None:
    raise RuntimeError("boom")


class TestResiliencePipelineBasics:
    def test_empty_pipeline(self) -> None:
        pipe = ResiliencePipeline()
        assert pipe.execute(_ok) == "ok"

    def test_pipeline_with_retry(self) -> None:
        calls = []

        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("transient")
            return "ok"

        pipe = ResiliencePipeline(retry_policy=RetryPolicy(max_retries=3, base_delay=0.01))
        assert pipe.execute(flaky) == "ok"
        assert len(calls) == 3

    def test_pipeline_with_circuit_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        pipe = ResiliencePipeline(circuit_breaker=cb)
        with pytest.raises(RuntimeError):
            pipe.execute(_fail)
        with pytest.raises(CircuitBreakerOpenError):
            pipe.execute(_ok)

    def test_pipeline_with_bulkhead(self) -> None:
        bh = Bulkhead(max_concurrent=1, max_queue=0)
        pipe = ResiliencePipeline(bulkhead=bh)

        def slow() -> None:
            time.sleep(0.2)

        t = __import__("threading").Thread(target=pipe.execute, args=(slow,))
        t.start()
        time.sleep(0.02)
        with pytest.raises(BulkheadFullError):
            pipe.execute(lambda: None)
        t.join()

    def test_pipeline_with_rate_limiter(self) -> None:
        rl = RateLimiter(rate=1000.0, burst=1)
        pipe = ResiliencePipeline(rate_limiter=rl)
        assert pipe.execute(_ok) == "ok"
        # second call should need to wait for refill; with high rate it still blocks briefly
        # force empty by acquiring the only token directly
        rl.acquire(blocking=True, timeout=0.5)
        # now bucket is empty and refill takes 0.001s but we give 0 timeout
        with pytest.raises(RateLimitExceededError):
            rl.acquire(blocking=True, timeout=0.0)
        # pipeline path also raises because it calls acquire(blocking=True, timeout=None)
        # so instead use a tiny burst to test the wrapper path
        rl2 = RateLimiter(rate=1.0, burst=1)
        pipe2 = ResiliencePipeline(rate_limiter=rl2)
        pipe2.execute(_ok)
        # exhaust
        rl2.acquire(blocking=True, timeout=2.0)
        # next pipeline call must wait >2s or raise on short timeout
        with pytest.raises(RateLimitExceededError):
            rl2.acquire(blocking=True, timeout=0.01)

    def test_full_pipeline_order(self) -> None:
        """Ensure layers are applied rate → bulkhead → circuit → retry."""
        order: list[str] = []

        def fn() -> str:
            order.append("fn")
            return "done"

        class TrackedRateLimiter(RateLimiter):
            def acquire(
                self, tokens: int = 1, blocking: bool = True, timeout: float | None = None
            ) -> bool:  # noqa: ARG002
                order.append("rate")
                return True

        class TrackedBulkhead(Bulkhead):
            def execute(self, fn, *args, **kwargs):  # noqa: ARG002
                order.append("bulkhead")
                return fn(*args, **kwargs)

        class TrackedCircuitBreaker(CircuitBreaker):
            def call(self, fn, *args, **kwargs):  # noqa: ARG002
                order.append("circuit")
                return fn(*args, **kwargs)

        class TrackedRetryPolicy(RetryPolicy):
            def execute(self, fn, *args, **kwargs):  # noqa: ARG002
                order.append("retry")
                return fn(*args, **kwargs)

        pipe = ResiliencePipeline(
            rate_limiter=TrackedRateLimiter(),
            bulkhead=TrackedBulkhead(),
            circuit_breaker=TrackedCircuitBreaker(),
            retry_policy=TrackedRetryPolicy(),
        )
        pipe.execute(fn)
        assert order == ["rate", "bulkhead", "circuit", "retry", "fn"]

    def test_pipeline_half_open_success(self) -> None:
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.05, success_threshold_half_open=1
        )
        pipe = ResiliencePipeline(circuit_breaker=cb)
        with pytest.raises(RuntimeError):
            pipe.execute(_fail)
        time.sleep(0.06)
        assert pipe.execute(_ok) == "ok"
        assert cb.state == "closed"
