"""Message retry policy for protocol reliability.

Pure functions for computing retry delays. No side effects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class BackoffStrategy(Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_max_seconds: float = 0.5

    def delay(self, attempt: int) -> float:
        """Compute delay for *attempt* (1-indexed).

        Returns 0.0 if attempt exceeds max_attempts.
        """
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        if attempt > self.max_attempts:
            return 0.0

        if self.strategy == BackoffStrategy.FIXED:
            base = self.base_delay_seconds
        elif self.strategy == BackoffStrategy.LINEAR:
            base = self.base_delay_seconds * attempt
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            base = self.base_delay_seconds * (2 ** (attempt - 1))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        base = min(base, self.max_delay_seconds)

        if self.jitter:
            base += random.uniform(0.0, self.jitter_max_seconds)

        return round(base, 3)

    def should_retry(self, attempt: int) -> bool:
        """True if *attempt* is within the retry budget."""
        return attempt < self.max_attempts


# Module-level defaults
DEFAULT_RETRY_POLICY = RetryPolicy()
RETRY_POLICY_REGISTRY: dict[str, RetryPolicy] = {
    "default": DEFAULT_RETRY_POLICY,
    "aggressive": RetryPolicy(
        max_attempts=5,
        base_delay_seconds=0.5,
        max_delay_seconds=30.0,
        strategy=BackoffStrategy.EXPONENTIAL,
    ),
    "gentle": RetryPolicy(
        max_attempts=2,
        base_delay_seconds=2.0,
        max_delay_seconds=10.0,
        strategy=BackoffStrategy.LINEAR,
    ),
}
