"""Thread-safe priority queue for serving requests."""

from __future__ import annotations

import heapq
import itertools
import threading
from typing import Any


class RequestPriorityQueue:
    """Priority queue backed by ``heapq`` with FIFO ordering within the same priority.

    Priority ``0`` is the highest.  A monotonic counter guarantees FIFO
    ordering when multiple items share the same priority level.

    All public methods are protected by an internal ``threading.Lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heap: list[tuple[int, int, str]] = []
        self._entries: dict[str, tuple[int, int, dict[str, Any]]] = {}
        self._counter = itertools.count()

    def enqueue(self, request_id: str, priority: int, payload: dict[str, Any]) -> None:
        """Add *request_id* to the queue with the given *priority* and *payload*.

        Raises:
            ValueError: If *request_id* is empty, longer than 128 characters,
                or already present in the queue.
            ValueError: If *priority* is not an integer in ``[0, 99]``.
        """
        if not isinstance(request_id, str):
            raise ValueError("request_id must be a str")
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if len(request_id) > 128:
            raise ValueError("request_id must be <= 128 characters")
        if not isinstance(priority, int):
            raise ValueError("priority must be an int")
        if priority < 0 or priority > 99:
            raise ValueError("priority must be in [0, 99]")

        with self._lock:
            if request_id in self._entries:
                raise ValueError(f"request_id already in queue: {request_id}")
            seq = next(self._counter)
            heapq.heappush(self._heap, (priority, seq, request_id))
            self._entries[request_id] = (priority, seq, payload)

    def dequeue(self) -> tuple[str, dict[str, Any]] | None:
        """Remove and return the highest-priority item.

        Returns ``None`` when the queue is empty.
        """
        with self._lock:
            while self._heap:
                priority, seq, request_id = heapq.heappop(self._heap)
                entry = self._entries.pop(request_id, None)
                if entry is not None:
                    return (request_id, entry[2])
            return None

    def peek(self) -> tuple[str, dict[str, Any]] | None:
        """Return the highest-priority item without removing it.

        Returns ``None`` when the queue is empty.
        """
        with self._lock:
            while self._heap:
                priority, seq, request_id = self._heap[0]
                if request_id in self._entries:
                    return (request_id, self._entries[request_id][2])
                heapq.heappop(self._heap)
            return None

    def remove(self, request_id: str) -> bool:
        """Remove *request_id* from the queue if it exists.

        Returns ``True`` if the item was present and removed.
        """
        with self._lock:
            if request_id in self._entries:
                del self._entries[request_id]
                return True
            return False

    def __len__(self) -> int:
        """Return the number of active items in the queue."""
        with self._lock:
            return len(self._entries)

    def list_by_priority(self) -> list[tuple[int, str, dict[str, Any]]]:
        """Return all active items sorted by priority (lowest number first),
        with FIFO ordering within the same priority.
        """
        with self._lock:
            items = [
                (priority, request_id, payload)
                for request_id, (priority, seq, payload) in self._entries.items()
            ]
            items.sort(key=lambda item: (item[0], self._entries[item[1]][1]))
            return items


PRIORITY_QUEUE_REGISTRY: dict[str, RequestPriorityQueue] = {
    "default": RequestPriorityQueue(),
}
