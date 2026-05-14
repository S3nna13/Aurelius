"""Bulkhead pattern — limits concurrent operations per resource.

Provides isolation by capping the number of simultaneous executions
and queuing excess requests up to a configurable limit.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class BulkheadFullError(Exception):
    """Raised when the bulkhead has no capacity and the wait queue is full."""


@dataclass
class Bulkhead:
    """Thread-safe bulkhead.

    Parameters
    ----------
    max_concurrent:
        Maximum number of concurrent executions allowed.
    max_queue:
        Maximum number of waiting requests.
    queue_timeout:
        Seconds a queued request will wait for a slot before raising.
    name:
        Optional identifier for logging / debugging.
    """

    max_concurrent: int = 10
    max_queue: int = 100
    queue_timeout: float = 5.0
    name: str = "default"

    _semaphore: threading.Semaphore = field(init=False, repr=False)
    _queue: deque[threading.Event] = field(default_factory=lambda: deque(maxlen=0), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _active: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_semaphore", threading.Semaphore(self.max_concurrent))
        object.__setattr__(self, "_queue", deque(maxlen=self.max_queue))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def active_count(self) -> int:
        """Number of currently executing operations."""
        with self._lock:
            return self._active

    @property
    def queue_count(self) -> int:
        """Number of operations waiting for a slot."""
        with self._lock:
            return len(self._queue)

    def execute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* within the bulkhead.

        Acquires a concurrency slot, queuing if necessary.  Raises
        :exc:`BulkheadFullError` if the queue is saturated or the
        wait times out.
        """
        self._acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            self._release()

    # ------------------------------------------------------------------ #
    # Internal mechanics
    # ------------------------------------------------------------------ #

    def _acquire(self) -> None:
        if self._semaphore.acquire(blocking=False):
            with self._lock:
                self._active += 1
            return

        event = threading.Event()
        with self._lock:
            if len(self._queue) >= self.max_queue:
                raise BulkheadFullError(f"Bulkhead {self.name!r} queue is full")
            self._queue.append(event)

        if not event.wait(timeout=self.queue_timeout):
            with self._lock:
                try:
                    self._queue.remove(event)
                except ValueError:
                    pass
            raise BulkheadFullError(f"Bulkhead {self.name!r} queue wait timed out")

        with self._lock:
            self._active += 1

    def _release(self) -> None:
        with self._lock:
            self._active -= 1
        self._semaphore.release()
        with self._lock:
            if self._queue:
                next_event = self._queue.popleft()
                next_event.set()
