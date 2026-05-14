"""Request throttler with token bucket algorithm for DoS prevention.

Trail of Bits: rate-limit at the entry point, fail closed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket for rate limiting individual callers."""

    capacity: int = 10
    refill_rate: float = 1.0
    tokens: float = 10.0
    _last_refill: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        # Ensure _lock has lock semantics (duck-typing to avoid Python 3.13 typing changes)
        if not callable(getattr(self._lock, 'acquire', None)) or not callable(getattr(self._lock, 'release', None)):
            object.__setattr__(self, "_lock", threading.Lock())
        self.tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self._last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def reset(self) -> None:
        with self._lock:
            self.tokens = float(self.capacity)
            self._last_refill = time.monotonic()


@dataclass
class RequestThrottler:
    """Per-caller throttling using token buckets."""

    default_capacity: int = 10
    default_refill: float = 1.0
    _buckets: dict[str, TokenBucket] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _max_callers: int = 10000
    _ttl_seconds: float = 300.0
    _last_access: dict[str, float] = field(default_factory=dict, repr=False)

    def allow(self, caller: str) -> bool:
        with self._lock:
            self._evict_stale()
            if caller not in self._buckets:
                if len(self._buckets) >= self._max_callers:
                    return False
                self._buckets[caller] = TokenBucket(
                    capacity=self.default_capacity,
                    refill_rate=self.default_refill,
                )
            self._last_access[caller] = time.monotonic()
            return self._buckets[caller].allow()

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale = [c for c, t in self._last_access.items() if now - t > self._ttl_seconds]
        for caller in stale:
            self._buckets.pop(caller, None)
            self._last_access.pop(caller, None)

    def reset_caller(self, caller: str) -> None:
        with self._lock:
            bucket = self._buckets.get(caller)
            if bucket:
                bucket.reset()

    def active_callers(self) -> int:
        with self._lock:
            return len(self._buckets)


REQUEST_THROTTLER = RequestThrottler()
