"""Request throttler with token bucket algorithm for DoS prevention.

Trail of Bits: rate-limit at the entry point, fail closed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket for rate limiting individual callers."""

    capacity: int = 10
    refill_rate: float = 1.0
    tokens: float = 10.0
    _last_refill: float = 0.0

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self._last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def reset(self) -> None:
        self.tokens = float(self.capacity)
        self._last_refill = time.monotonic()


@dataclass
class RequestThrottler:
    """Per-caller throttling using token buckets."""

    default_capacity: int = 10
    default_refill: float = 1.0
    _buckets: dict[str, TokenBucket] = field(default_factory=dict, repr=False)

    def allow(self, caller: str) -> bool:
        if caller not in self._buckets:
            self._buckets[caller] = TokenBucket(
                capacity=self.default_capacity,
                refill_rate=self.default_refill,
            )
        return self._buckets[caller].allow()

    def reset_caller(self, caller: str) -> None:
        bucket = self._buckets.get(caller)
        if bucket:
            bucket.reset()

    def active_callers(self) -> int:
        return len(self._buckets)


REQUEST_THROTTLER = RequestThrottler()
