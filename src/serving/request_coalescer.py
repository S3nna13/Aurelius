"""In-flight request coalescer for the Aurelius serving surface.

When multiple identical requests arrive simultaneously, only one is
computed and all waiters receive the same result.  This eliminates
thundering-herd load on the model backend without adding external
dependencies.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class _InflightEntry:
    __slots__ = ("event", "started_at", "waiter_count", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.started_at = time.monotonic()
        self.waiter_count = 1
        self.result: object | None = None
        self.error: BaseException | None = None


class RequestCoalescer:
    """Deduplicate in-flight identical requests.

    Parameters
    ----------
    ttl_s:
        Maximum time to hold an in-flight slot before treating it as
        stale and allowing a new computation (default 30 s).
    max_inflight:
        Hard cap on concurrent in-flight keys to prevent memory
        exhaustion when requests hang or are very slow (default 1_024).
    """

    def __init__(self, ttl_s: float = 30.0, max_inflight: int = 1_024) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self._ttl = ttl_s
        self._max_inflight = max_inflight
        self._inflight: dict[str, _InflightEntry] = {}
        self._lock = threading.Lock()

    def coalesce(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Return the result for *key*, computing it once even under contention.

        *compute_fn* is invoked at most once per in-flight window.  All
        concurrent callers with the same *key* block until the first
        caller finishes *compute_fn* and then receive the same result
        (or the same exception is re-raised).
        """
        entry: _InflightEntry | None = None
        is_leader = False

        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                age = time.monotonic() - existing.started_at
                if age < self._ttl:
                    existing.waiter_count += 1
                    entry = existing
                else:
                    # Stale entry — replace
                    entry = _InflightEntry()
                    self._inflight[key] = entry
                    is_leader = True
            else:
                if len(self._inflight) >= self._max_inflight:
                    raise RuntimeError(
                        f"request coalescer at capacity ({self._max_inflight} in-flight)"
                    )
                entry = _InflightEntry()
                self._inflight[key] = entry
                is_leader = True

        if not is_leader:
            entry.event.wait()
            if entry.error is not None:
                raise entry.error
            return entry.result  # type: ignore[return-value]

        # Leader path — actually compute
        try:
            entry.result = compute_fn()
        except BaseException as exc:
            entry.error = exc
        finally:
            entry.event.set()
            with self._lock:
                self._inflight.pop(key, None)

        if entry.error is not None:
            raise entry.error
        return entry.result  # type: ignore[return-value]

    def stats(self) -> dict:
        """Return current in-flight statistics."""
        with self._lock:
            return {
                "inflight_count": len(self._inflight),
                "total_waiters": sum(e.waiter_count for e in self._inflight.values()),
            }

    def clear(self) -> None:
        """Drop all in-flight entries.  Waiting threads will wake with an error."""
        with self._lock:
            for entry in self._inflight.values():
                entry.error = RuntimeError("Coalescer cleared")
                entry.event.set()
            self._inflight.clear()


#: Module-level registry for dependency injection / tests.
REQUEST_COALESCER_REGISTRY: dict[str, RequestCoalescer] = {}
