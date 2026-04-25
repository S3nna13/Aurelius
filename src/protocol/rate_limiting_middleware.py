"""Rate limiting middleware for the Aurelius protocol layer.

Supports three algorithms:
  - TOKEN_BUCKET  : classic leaky-bucket refill (per-client burst tokens).
  - FIXED_WINDOW  : count requests inside a fixed wall-clock window slot.
  - SLIDING_LOG   : timestamp log; evict entries older than window_s.

Pure stdlib only.  No external dependencies.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


# ---------------------------------------------------------------------------
# Algorithm enum
# ---------------------------------------------------------------------------


class RateLimitAlgorithm(str, Enum):
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_LOG  = "sliding_log"


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitConfig:
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    rate:      float               = 100.0   # tokens refilled per second
    burst:     int                 = 10      # max tokens / max requests per window
    window_s:  float               = 1.0     # window length in seconds


@dataclass(frozen=True)
class RateLimitResult:
    allowed:       bool
    remaining:     int
    reset_after_s: float
    client_id:     str


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RateLimitingMiddleware:
    """Protocol-layer rate limiter.

    Usage::

        mw = RateLimitingMiddleware()
        result = mw.check("client-abc")
        if not result.allowed:
            raise RuntimeError("rate limited")
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config: RateLimitConfig = config if config is not None else RateLimitConfig()

        # TOKEN_BUCKET: {client_id: [tokens_float, last_refill_time]}
        self._buckets: Dict[str, List[float]] = {}

        # FIXED_WINDOW: {client_id: [count, window_start]}
        self._windows: Dict[str, List[float]] = {}

        # SLIDING_LOG: {client_id: [timestamp, ...]}
        self._logs: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, client_id: str, now: float | None = None) -> RateLimitResult:
        """Evaluate whether *client_id* is within rate limits at time *now*."""
        t = now if now is not None else time.monotonic()
        algo = self.config.algorithm

        if algo == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(client_id, t)
        if algo == RateLimitAlgorithm.FIXED_WINDOW:
            return self._check_fixed_window(client_id, t)
        # SLIDING_LOG
        return self._check_sliding_log(client_id, t)

    def reset(self, client_id: str) -> None:
        """Clear all rate-limit state for *client_id*."""
        self._buckets.pop(client_id, None)
        self._windows.pop(client_id, None)
        if client_id in self._logs:
            del self._logs[client_id]

    def stats(self) -> dict:
        """Return aggregate statistics."""
        all_clients: set = (
            set(self._buckets) | set(self._windows) | set(self._logs)
        )
        return {
            "total_clients": len(all_clients),
            "algorithm": self.config.algorithm.value,
        }

    # ------------------------------------------------------------------
    # Algorithm implementations
    # ------------------------------------------------------------------

    def _check_token_bucket(self, client_id: str, now: float) -> RateLimitResult:
        burst = self.config.burst
        rate  = self.config.rate

        if client_id not in self._buckets:
            # Start with a full bucket
            self._buckets[client_id] = [float(burst), now]

        tokens, last = self._buckets[client_id]
        elapsed = max(0.0, now - last)
        tokens  = min(float(burst), tokens + elapsed * rate)

        if tokens >= 1.0:
            tokens -= 1.0
            allowed = True
        else:
            allowed = False

        self._buckets[client_id] = [tokens, now]

        remaining     = int(tokens)
        reset_after_s = 0.0 if allowed else (1.0 - tokens) / rate if rate > 0 else 0.0

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_after_s=reset_after_s,
            client_id=client_id,
        )

    def _check_fixed_window(self, client_id: str, now: float) -> RateLimitResult:
        burst    = self.config.burst
        window_s = self.config.window_s

        if client_id not in self._windows:
            self._windows[client_id] = [0.0, now]

        count, window_start = self._windows[client_id]

        # If we are past the current window, reset
        if now >= window_start + window_s:
            window_start = now
            count        = 0.0

        if count < burst:
            count  += 1
            allowed = True
        else:
            allowed = False

        self._windows[client_id] = [count, window_start]

        remaining     = max(0, burst - int(count))
        reset_after_s = (window_start + window_s) - now

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_after_s=max(0.0, reset_after_s),
            client_id=client_id,
        )

    def _check_sliding_log(self, client_id: str, now: float) -> RateLimitResult:
        burst    = self.config.burst
        window_s = self.config.window_s
        log      = self._logs[client_id]

        # Evict timestamps older than window_s
        cutoff = now - window_s
        # Keep only entries within the window
        self._logs[client_id] = [ts for ts in log if ts > cutoff]
        log = self._logs[client_id]

        if len(log) < burst:
            log.append(now)
            allowed = True
        else:
            allowed = False

        remaining = max(0, burst - len(log))
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_after_s=0.0,
            client_id=client_id,
        )


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

RATE_LIMIT_MIDDLEWARE_REGISTRY: dict = {
    "default": RateLimitingMiddleware,
}
