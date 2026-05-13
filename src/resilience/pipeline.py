"""ResiliencePipeline — composes circuit breaker, retry, bulkhead and rate limiter.

Provides a single callable wrapper that applies all four resilience
patterns in a sensible order:

  1. Rate limiter   (throttle before doing work)
  2. Bulkhead       (limit concurrency)
  3. Circuit breaker (fail fast if downstream is unhealthy)
  4. Retry policy   (retry transient failures)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bulkhead import Bulkhead
from .circuit_breaker import CircuitBreaker
from .rate_limiter import RateLimiter
from .retry_policy import RetryPolicy


@dataclass
class ResiliencePipeline:
    """Composable resilience pipeline.

    Parameters
    ----------
    circuit_breaker:
        Optional :class:`CircuitBreaker` instance.
    retry_policy:
        Optional :class:`RetryPolicy` instance.
    bulkhead:
        Optional :class:`Bulkhead` instance.
    rate_limiter:
        Optional :class:`RateLimiter` instance.
    """

    circuit_breaker: CircuitBreaker | None = None
    retry_policy: RetryPolicy | None = None
    bulkhead: Bulkhead | None = None
    rate_limiter: RateLimiter | None = None

    _name: str = field(default="pipeline", repr=False)

    def execute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run *fn* through the configured resilience layers.

        Layers are applied inside-out so that the outermost layer
        controls admission (rate limiter → bulkhead) and the innermost
        layer controls execution semantics (circuit breaker → retry).
        """
        wrapped = fn
        if self.retry_policy is not None:
            wrapped = self._wrap_retry(wrapped)
        if self.circuit_breaker is not None:
            wrapped = self._wrap_circuit_breaker(wrapped)
        if self.bulkhead is not None:
            wrapped = self._wrap_bulkhead(wrapped)
        if self.rate_limiter is not None:
            wrapped = self._wrap_rate_limiter(wrapped)
        return wrapped(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Layer wrappers
    # ------------------------------------------------------------------ #

    def _wrap_rate_limiter(self, fn: Any) -> Any:
        def _rate_limited(*args: Any, **kwargs: Any) -> Any:
            self.rate_limiter.acquire(tokens=1, blocking=True, timeout=None)  # type: ignore[union-attr]
            return fn(*args, **kwargs)

        return _rate_limited

    def _wrap_bulkhead(self, fn: Any) -> Any:
        def _bulkheaded(*args: Any, **kwargs: Any) -> Any:
            return self.bulkhead.execute(fn, *args, **kwargs)  # type: ignore[union-attr]

        return _bulkheaded

    def _wrap_circuit_breaker(self, fn: Any) -> Any:
        def _circuited(*args: Any, **kwargs: Any) -> Any:
            return self.circuit_breaker.call(fn, *args, **kwargs)  # type: ignore[union-attr]

        return _circuited

    def _wrap_retry(self, fn: Any) -> Any:
        def _retried(*args: Any, **kwargs: Any) -> Any:
            return self.retry_policy.execute(fn, *args, **kwargs)  # type: ignore[union-attr]

        return _retried
