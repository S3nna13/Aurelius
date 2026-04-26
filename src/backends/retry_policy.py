"""Retry policy with exponential backoff and optional jitter.

Provides a configurable retry loop with injectable sleep for deterministic
testing. All logic is stdlib-only; no foreign dependencies.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "RetryConfig",
    "RetryResult",
    "RetryPolicy",
    "RETRY_POLICY_REGISTRY",
]


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for a :class:`RetryPolicy`."""

    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True


@dataclass(frozen=True)
class RetryResult:
    """Outcome of a :meth:`RetryPolicy.execute` call."""

    attempts: int
    success: bool
    last_error: str
    total_delay_s: float


class RetryPolicy:
    """Executes a callable with automatic retries on failure.

    Parameters
    ----------
    config:
        Retry configuration; a default :class:`RetryConfig` is used when
        *config* is ``None``.
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self._config: RetryConfig = config if config is not None else RetryConfig()

    # ------------------------------------------------------------------
    # Delay calculation
    # ------------------------------------------------------------------

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the sleep duration (in seconds) for the given *attempt* index.

        Uses exponential back-off:  ``base_delay * (backoff_factor ** attempt)``
        clamped to ``max_delay_s``.  When ``jitter`` is enabled the result is
        multiplied by a uniform random factor in ``[0.5, 1.0)``.
        """
        cfg = self._config
        raw = cfg.base_delay_s * (cfg.backoff_factor**attempt)
        clamped = min(raw, cfg.max_delay_s)
        if cfg.jitter:
            clamped *= random.uniform(0.5, 1.0)
        return clamped

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        fn: Callable[[], object],
        on_retry: Callable[[int, Exception], None] | None = None,
        *,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> RetryResult:
        """Call *fn* up to ``max_retries + 1`` times.

        Parameters
        ----------
        fn:
            The callable to attempt.  Must be zero-argument.
        on_retry:
            Optional hook called as ``on_retry(attempt, exception)`` before
            each sleep.  *attempt* is 0-indexed.
        sleep_fn:
            Callable used for sleeping between retries.  Defaults to
            ``time.sleep``.  Pass a no-op lambda in tests to avoid real waits.

        Returns
        -------
        RetryResult
            Summary of the execution, including whether *fn* eventually
            succeeded, total attempts made, and cumulative delay waited.
        """
        cfg = self._config
        _sleep = sleep_fn if sleep_fn is not None else time.sleep

        total_delay = 0.0
        last_error = ""
        attempts = 0
        max_calls = cfg.max_retries + 1

        for attempt in range(max_calls):
            attempts += 1
            try:
                fn()
                return RetryResult(
                    attempts=attempts,
                    success=True,
                    last_error="",
                    total_delay_s=total_delay,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_calls - 1:
                    if on_retry is not None:
                        on_retry(attempt, exc)
                    delay = self.delay_for_attempt(attempt)
                    total_delay += delay
                    _sleep(delay)

        return RetryResult(
            attempts=attempts,
            success=False,
            last_error=last_error,
            total_delay_s=total_delay,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, num_attempts: int) -> list[float]:
        """Return a list of delay values for each retry (no actual sleeping).

        *num_attempts* is the total number of calls that will be made;
        the delay list has length ``num_attempts`` (one delay per attempt
        index, 0-indexed), which represents the wait *before* the next try.
        Useful for visualising the back-off curve.
        """
        return [self.delay_for_attempt(i) for i in range(num_attempts)]


RETRY_POLICY_REGISTRY: dict[str, type[RetryPolicy]] = {"default": RetryPolicy}
