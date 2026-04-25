"""Batched inference predictor with automatic batching and thread-safe submission."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class _PendingItem:
    future_id: str
    request_id: str
    input_tensor: torch.Tensor
    enqueued_at: float


class BatchPredictor:
    """Batches inference requests for efficient throughput.

    Requests are queued via :meth:`submit`.  A background worker forms a batch
    when either *max_batch_size* is reached or the oldest request has been
    waiting for at least *max_wait_ms* milliseconds.  The batch is then passed
    to *backend* and results are made available to :meth:`collect_results`.

    Args:
        backend: Callable that takes a list of tensors and returns a list of
            tensors (one per input).
        max_batch_size: Maximum number of requests per batch.
        max_wait_ms: Maximum time the oldest request may wait before a batch
            is forced.
    """

    def __init__(
        self,
        backend: Callable[[list[torch.Tensor]], list[torch.Tensor]],
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
    ) -> None:
        if not callable(backend):
            raise TypeError("backend must be callable")
        if not isinstance(max_batch_size, int):
            raise TypeError("max_batch_size must be an int")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_batch_size > 1024:
            raise ValueError("max_batch_size exceeds safety limit of 1024")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")

        self._backend = backend
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0

        self._lock = threading.Lock()
        self._queue: list[_PendingItem] = []
        self._results: dict[str, torch.Tensor] = {}
        self._events: dict[str, threading.Event] = {}
        self._inflight = 0
        self._flush_event = threading.Event()
        self._shutdown = False

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, request_id: str, input_tensor: torch.Tensor) -> str:
        """Queue an inference request and return a future ID."""
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a str")
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("input_tensor must be a torch.Tensor")

        future_id = uuid.uuid4().hex
        item = _PendingItem(
            future_id=future_id,
            request_id=request_id,
            input_tensor=input_tensor,
            enqueued_at=time.monotonic(),
        )
        with self._lock:
            self._queue.append(item)
            self._events[future_id] = threading.Event()
        return future_id

    def collect_results(
        self, future_ids: list[str], timeout: float = 30.0
    ) -> dict[str, torch.Tensor]:
        """Block until all *future_ids* are available.

        Raises:
            KeyError: If a future_id was never submitted.
            TimeoutError: If *timeout* seconds elapse before all results arrive.
        """
        deadline = time.monotonic() + timeout
        results: dict[str, torch.Tensor] = {}
        for fid in future_ids:
            with self._lock:
                event = self._events.get(fid)
            if event is None:
                raise KeyError(f"Unknown future_id: {fid}")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not event.wait(timeout=max(remaining, 0.0)):
                raise TimeoutError(f"Timed out waiting for result {fid}")
            with self._lock:
                results[fid] = self._results.pop(fid)
                del self._events[fid]
        return results

    def flush(self) -> None:
        """Force immediate processing of any pending batch."""
        self._flush_event.set()
        while True:
            with self._lock:
                if not self._queue and self._inflight == 0:
                    self._flush_event.clear()
                    break
            time.sleep(0.001)

    def shutdown(self) -> None:
        """Signal the background worker to stop and wait for it to exit."""
        with self._lock:
            self._shutdown = True
        self._worker.join(timeout=5.0)

    def _worker_loop(self) -> None:
        while True:
            batch: list[_PendingItem] = []
            with self._lock:
                if self._shutdown:
                    break
                if self._queue:
                    oldest_wait = time.monotonic() - self._queue[0].enqueued_at
                    should_process = (
                        len(self._queue) >= self._max_batch_size
                        or oldest_wait >= self._max_wait_s
                        or self._flush_event.is_set()
                    )
                    if should_process:
                        batch = self._queue[: self._max_batch_size]
                        self._queue = self._queue[self._max_batch_size :]
                        self._inflight += len(batch)
                        self._flush_event.clear()

            if batch:
                tensors = [item.input_tensor for item in batch]
                outputs = self._backend(tensors)
                with self._lock:
                    for item, out in zip(batch, outputs, strict=True):
                        self._results[item.future_id] = out
                        self._events[item.future_id].set()
                    self._inflight -= len(batch)
                continue

            time.sleep(0.001)
