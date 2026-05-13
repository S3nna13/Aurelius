"""Retry policy with exponential backoff and jitter.

Provides a configurable, thread-safe retry wrapper suitable for
transient-fault handling in network and I/O operations.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryPolicy:
    """Retry configuration.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts before giving up.
    base_delay:
        Initial delay between retries in seconds.
    max_delay:
        Upper bound for any individual delay in seconds.
    exponential_base:
        Multiplier applied to the delay on each retry.
    jitter:
        If ``True``, adds a random fraction of the computed delay.
    retryable_exceptions:
        Tuple of exception types that should trigger a retry.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = field(
        default_factory=lambda: (Exception,),
    )

    def execute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Call *fn* and retry on matching exceptions.

        Raises
        ------
        Exception
            The last exception raised if all retries are exhausted.
        """
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._should_retry(exc):
                    raise
                delay = self._compute_delay(attempt)
                time.sleep(delay)
        # Unreachable, but keeps type-checkers happy.
        raise last_exc  # pragma: no cover

    def _should_retry(self, exc: BaseException) -> bool:
        return isinstance(exc, self.retryable_exceptions)

    def _compute_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay
