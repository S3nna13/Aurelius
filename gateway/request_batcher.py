"""Batches incoming inference requests for efficient throughput.

Requests are enqueued individually.  A batch is formed when either:

* The queue reaches *max_batch_size*, or
* The oldest enqueued request has been waiting for at least *max_wait_ms*
  milliseconds.

When a batch is formed the constituent requests are removed from the queue
and returned wrapped in a :class:`Batch` object.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass
class BatchRequest:
    """A single pending inference request.

    Args:
        request_id:  Caller-supplied identifier.
        payload:     Arbitrary request data.
        priority:    Higher numbers are batched first when the queue exceeds
                     *max_batch_size* (only the top *max_batch_size* items are
                     kept in a batch).
        enqueued_at: Monotonic timestamp of enqueue time.  Set automatically
                     to ``time.monotonic()`` when the value is ``0.0``.
    """

    request_id: str
    payload: dict
    priority: int = 0
    enqueued_at: float = 0.0

    def __post_init__(self) -> None:
        if self.enqueued_at == 0.0:
            self.enqueued_at = time.monotonic()


@dataclass(frozen=True)
class Batch:
    """An immutable group of :class:`BatchRequest` objects ready for inference."""

    batch_id: str
    requests: list[BatchRequest]
    created_at: float


class RequestBatcher:
    """Collects and groups inference requests into efficient batches.

    Args:
        max_batch_size: Maximum number of requests per batch (default 8).
        max_wait_ms:    Maximum time in milliseconds the oldest request in the
                        queue is allowed to wait before a batch is forced
                        (default 50.0).
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: list[BatchRequest] = []

    # ------------------------------------------------------------------
    # Enqueueing
    # ------------------------------------------------------------------

    def enqueue(self, request: BatchRequest) -> None:
        """Add *request* to the pending queue."""
        self._queue.append(request)

    # ------------------------------------------------------------------
    # Batch formation
    # ------------------------------------------------------------------

    def try_form_batch(self) -> Batch | None:
        """Attempt to form a :class:`Batch` from the current queue.

        A batch is returned when:

        * The queue has at least *max_batch_size* requests, **or**
        * The oldest request has been waiting >= *max_wait_ms* milliseconds.

        When a batch is formed the *max_batch_size* highest-priority requests
        are selected (highest ``priority`` value first; original enqueue order
        is preserved within the same priority).  All batched requests are
        removed from the queue.

        Returns:
            A :class:`Batch` when the conditions above are met, otherwise
            ``None``.
        """
        if not self._queue:
            return None

        now = time.monotonic()
        oldest_wait_ms = (now - self._queue[0].enqueued_at) * 1000.0

        should_batch = len(self._queue) >= self.max_batch_size or oldest_wait_ms >= self.max_wait_ms

        if not should_batch:
            return None

        # Select up to max_batch_size requests, highest priority first.
        # Stable sort: secondary key preserves relative insertion order.
        indexed = list(enumerate(self._queue))
        indexed.sort(key=lambda t: (-t[1].priority, t[0]))
        selected_indices = {t[0] for t in indexed[: self.max_batch_size]}

        selected: list[BatchRequest] = []
        remaining: list[BatchRequest] = []
        for i, req in enumerate(self._queue):
            if i in selected_indices:
                selected.append(req)
            else:
                remaining.append(req)

        self._queue = remaining

        batch = Batch(
            batch_id=uuid.uuid4().hex[:8],
            requests=selected,
            created_at=now,
        )
        return batch

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def queue_size(self) -> int:
        """Return the number of requests currently waiting in the queue."""
        return len(self._queue)

    def pending_stats(self) -> dict:
        """Return a snapshot of queue statistics.

        Returns:
            ``{"queued": int, "oldest_wait_ms": float | None}``
        """
        if not self._queue:
            return {"queued": 0, "oldest_wait_ms": None}

        now = time.monotonic()
        oldest_wait_ms = (now - self._queue[0].enqueued_at) * 1000.0
        return {"queued": len(self._queue), "oldest_wait_ms": oldest_wait_ms}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REQUEST_BATCHER_REGISTRY: dict[str, type] = {
    "default": RequestBatcher,
}
