"""Advanced rate limiter with per-user and global token buckets.

Provides exponential jitter backoff for 429/5xx retry logic and thread-safe
token bucket primitives for both global and per-user rate limits.
"""

import random
import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    requests_per_second: float = 10.0
    burst: int = 20
    per_user_rps: float = 2.0
    per_user_burst: int = 5
    backoff_base: float = 1.0
    backoff_cap: float = 60.0
    backoff_jitter: float = 0.3


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: float = 0.0
    reason: str = ""


class TokenBucketV2:
    """Thread-safe token bucket with continuous refill."""

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, n: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def tokens_available(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            return min(self.burst, self._tokens + elapsed * self.rate)


class RateLimiterV2:
    """Two-tier rate limiter: global bucket + per-user bucket."""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._global_bucket = TokenBucketV2(self.config.requests_per_second, self.config.burst)
        self._user_buckets: dict[str, TokenBucketV2] = {}
        self._lock = threading.Lock()

    def _get_user_bucket(self, user_id: str) -> TokenBucketV2:
        with self._lock:
            if user_id not in self._user_buckets:
                self._user_buckets[user_id] = TokenBucketV2(
                    self.config.per_user_rps, self.config.per_user_burst
                )
            return self._user_buckets[user_id]

    def check(self, user_id: str | None = None) -> RateLimitResult:
        if not self._global_bucket.consume():
            return RateLimitResult(
                allowed=False,
                retry_after_seconds=1.0 / self.config.requests_per_second,
                reason="global_limit",
            )
        if user_id:
            bucket = self._get_user_bucket(user_id)
            if not bucket.consume():
                return RateLimitResult(
                    allowed=False,
                    retry_after_seconds=1.0 / self.config.per_user_rps,
                    reason="user_limit",
                )
        return RateLimitResult(allowed=True)

    @staticmethod
    def exponential_jitter_backoff(
        attempt: int,
        base: float = 1.0,
        cap: float = 60.0,
        jitter: float = 0.3,
    ) -> float:
        """Exponential backoff with jitter for 429/5xx retries."""
        delay = min(cap, base * (2**attempt))
        jitter_factor = 1.0 + random.uniform(-jitter, jitter)
        return max(0.0, delay * jitter_factor)


RATE_LIMITER_V2 = RateLimiterV2()

# Register in the shared SERVING_REGISTRY if available.
try:
    from src.serving import SERVING_REGISTRY  # type: ignore

    SERVING_REGISTRY["rate_limiter_v2"] = RATE_LIMITER_V2
except ImportError:
    pass
