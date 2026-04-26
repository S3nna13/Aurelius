"""Token-bucket rate limiter for the Aurelius serving surface. DoS defense per STRIDE-DoS. RATE_LIMIT_REGISTRY."""  # noqa: E501

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for a token-bucket rate limiter."""

    requests_per_second: float
    burst_size: int
    per: str = "key"  # one of "key", "ip", "route", "global"


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    retry_after_s: float
    limit: float


class _Bucket:
    """A single token bucket with its own lock."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self._lock = threading.Lock()
        self._capacity = capacity
        self._refill_rate = refill_rate  # tokens per second
        self._tokens: float = float(capacity)
        self._last_refill: float = time.monotonic()

    def consume(self) -> tuple[bool, int, float]:
        """
        Attempt to consume one token.

        Returns:
            (allowed, remaining_int, retry_after_s)
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._last_refill = now

            # Refill tokens based on elapsed time
            self._tokens = min(
                float(self._capacity),
                self._tokens + elapsed * self._refill_rate,
            )

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                remaining = int(self._tokens)
                return True, remaining, 0.0
            else:
                # Calculate how long until next token is available
                deficit = 1.0 - self._tokens
                if self._refill_rate > 0.0:
                    retry_after_s = deficit / self._refill_rate
                else:
                    retry_after_s = 3600.0
                return False, 0, retry_after_s

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self._tokens = float(self._capacity)
            self._last_refill = time.monotonic()


class TokenBucketLimiter:
    """Thread-safe per-identifier token bucket rate limiter."""

    #: Maximum number of distinct buckets to retain.  When the limit is
    #: reached the oldest bucket is evicted (LRU).  This prevents memory
    #: exhaustion when attackers rotate identifiers.
    MAX_BUCKETS: int = 10_000

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._buckets: dict[str, _Bucket] = {}
        self._access_order: list[str] = []
        self._registry_lock = threading.Lock()

    def _get_or_create_bucket(self, identifier: str) -> _Bucket:
        with self._registry_lock:
            if identifier in self._buckets:
                # Move to most-recently-used position
                if identifier in self._access_order:
                    self._access_order.remove(identifier)
                self._access_order.append(identifier)
                return self._buckets[identifier]

            # Evict oldest bucket if at capacity
            if len(self._buckets) >= self.MAX_BUCKETS:
                oldest = self._access_order.pop(0)
                self._buckets.pop(oldest, None)

            bucket = _Bucket(
                capacity=self.config.burst_size,
                refill_rate=self.config.requests_per_second,
            )
            self._buckets[identifier] = bucket
            self._access_order.append(identifier)
            return bucket

    def _resolve_identifier(self, identifier: str) -> str:
        """For global limiters, all requests share the same bucket."""
        if self.config.per == "global":
            return "__global__"
        return identifier

    def check(self, identifier: str) -> RateLimitResult:
        """
        Check whether a request from `identifier` is allowed.

        Thread-safe. Uses token bucket algorithm with monotonic clock refill.
        """
        resolved = self._resolve_identifier(identifier)
        bucket = self._get_or_create_bucket(resolved)
        allowed, remaining, retry_after_s = bucket.consume()
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            retry_after_s=retry_after_s,
            limit=float(self.config.requests_per_second),
        )

    def reset(self, identifier: str) -> None:
        """Reset the bucket for a specific identifier."""
        resolved = self._resolve_identifier(identifier)
        with self._registry_lock:
            if resolved in self._buckets:
                self._buckets[resolved].reset()

    def reset_all(self) -> None:
        """Reset all buckets."""
        with self._registry_lock:
            for bucket in self._buckets.values():
                bucket.reset()
            self._buckets.clear()
            self._access_order.clear()


class RateLimiterChain:
    """
    Ordered chain of named TokenBucketLimiters.

    Checks each limiter in order; returns the first denied result,
    or the last allowed result if all pass.
    """

    def __init__(self, limiters: list[tuple[str, TokenBucketLimiter]]) -> None:
        self._limiters = limiters

    def check_all(self, key: str, ip: str, route: str) -> RateLimitResult:
        """
        Check all limiters in order.

        Routes the correct identifier to each limiter based on its `config.per`:
          - "key"    -> key
          - "ip"     -> ip
          - "route"  -> route
          - "global" -> (any string; limiter resolves to __global__)
        """
        last_result: RateLimitResult | None = None

        for _name, limiter in self._limiters:
            per = limiter.config.per
            if per == "key":
                identifier = key
            elif per == "ip":
                identifier = ip
            elif per == "route":
                identifier = route
            else:  # "global"
                identifier = key  # value doesn't matter; limiter normalizes it

            result = limiter.check(identifier)
            last_result = result
            if not result.allowed:
                return result

        # All passed — return last allowed result
        assert last_result is not None  # noqa: S101
        return last_result


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

DEFAULT_RATE_LIMITER = TokenBucketLimiter(
    RateLimitConfig(requests_per_second=100.0, burst_size=200)
)

RATE_LIMIT_REGISTRY: dict[str, TokenBucketLimiter] = {
    "default": DEFAULT_RATE_LIMITER,
}
