"""Load shedder for Aurelius serving infrastructure.

Decides whether an incoming request should be shed (dropped) based on one of
four policies: DROP_TAIL, PRIORITY_QUEUE, TOKEN_BUCKET, or ADAPTIVE.

Also maintains a rolling p99 latency estimate (ring buffer, 1 000 samples) and
exposes aggregate stats via ``get_stats()``.

Pure stdlib only.  Thread-safe via threading.Lock.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import List


class ShedPolicy(str, Enum):
    DROP_TAIL = "drop_tail"
    PRIORITY_QUEUE = "priority_queue"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


class LoadShedder:
    """Evaluates whether a request should be shed under a given policy."""

    _RING_SIZE = 1000

    def __init__(
        self,
        max_depth: int = 100,
        token_rate: float = 100.0,   # tokens/second replenished
        bucket_capacity: float = 100.0,
        p99_threshold_ms: float = 2000.0,
    ) -> None:
        self._max_depth = max_depth
        self._p99_threshold = p99_threshold_ms

        # Token bucket state
        self._token_rate = token_rate
        self._bucket_capacity = bucket_capacity
        self._tokens = bucket_capacity
        self._last_refill = time.monotonic()

        # Ring buffer for latency samples
        self._ring: List[float] = [0.0] * self._RING_SIZE
        self._ring_pos: int = 0
        self._ring_count: int = 0

        # Shed counter
        self._shed_count: int = 0
        self._current_queue_depth: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def should_shed(
        self,
        queue_depth: int,
        latency_ms: float,
        policy: ShedPolicy,
    ) -> bool:
        """Return ``True`` if the request should be dropped."""
        with self._lock:
            self._current_queue_depth = queue_depth

            if policy == ShedPolicy.DROP_TAIL:
                result = queue_depth > self._max_depth

            elif policy == ShedPolicy.PRIORITY_QUEUE:
                # Placeholder — priority routing handles this upstream
                result = False

            elif policy == ShedPolicy.TOKEN_BUCKET:
                self._refill_tokens()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    result = False
                else:
                    result = True

            elif policy == ShedPolicy.ADAPTIVE:
                p99 = self._compute_p99()
                result = p99 > self._p99_threshold

            else:
                result = False

            if result:
                self._shed_count += 1
            return result

    # ------------------------------------------------------------------
    # Latency recording
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float) -> None:
        """Add *latency_ms* to the rolling ring buffer."""
        with self._lock:
            self._ring[self._ring_pos] = latency_ms
            self._ring_pos = (self._ring_pos + 1) % self._RING_SIZE
            if self._ring_count < self._RING_SIZE:
                self._ring_count += 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return current telemetry snapshot."""
        with self._lock:
            return {
                "queue_depth": self._current_queue_depth,
                "p99_latency": self._compute_p99(),
                "shed_count": self._shed_count,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill_tokens(self) -> None:
        """Replenish token bucket based on elapsed time (must hold _lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._token_rate
        self._tokens = min(self._bucket_capacity, self._tokens + new_tokens)
        self._last_refill = now

    def _compute_p99(self) -> float:
        """Compute p99 from the ring buffer samples (must hold _lock)."""
        if self._ring_count == 0:
            return 0.0
        samples = sorted(self._ring[: self._ring_count])
        idx = int(len(samples) * 0.99)
        idx = min(idx, len(samples) - 1)
        return samples[idx]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

SERVING_REGISTRY: dict = {
    "load_shedder": LoadShedder(),
}
