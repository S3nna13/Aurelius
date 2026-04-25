"""Priority-queue batch inference processor for Aurelius."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from queue import PriorityQueue


class QueueFullError(Exception):
    pass


@dataclass
class BatchRequest:
    request_id: str
    prompts: list[str]
    max_tokens: int = 256
    temperature: float = 1.0
    priority: int = 5


@dataclass
class BatchResult:
    request_id: str
    outputs: list[str]
    token_counts: list[int]
    latency_ms: float
    error: str | None = None


@dataclass(order=True)
class _PrioritizedRequest:
    priority: int
    sequence: int
    request: BatchRequest = field(compare=False)


class BatchProcessor:
    """Continuous batching — accumulate requests, process in priority order."""

    def __init__(self, max_batch_size: int = 32, max_queue_size: int = 512) -> None:
        self._max_batch_size = max_batch_size
        self._max_queue_size = max_queue_size
        self._queue: list[_PrioritizedRequest] = []
        self._sequence: int = 0

    def enqueue(self, request: BatchRequest) -> str:
        if len(self._queue) >= self._max_queue_size:
            raise QueueFullError("Batch queue is full")
        entry = _PrioritizedRequest(
            priority=request.priority,
            sequence=self._sequence,
            request=request,
        )
        self._sequence += 1
        self._queue.append(entry)
        self._queue.sort()
        return request.request_id

    def dequeue_batch(self) -> list[BatchRequest]:
        batch = self._queue[: self._max_batch_size]
        self._queue = self._queue[self._max_batch_size :]
        return [e.request for e in batch]

    def process_batch(self, requests: list[BatchRequest]) -> list[BatchResult]:
        latency_ms = len(requests) * 10.0
        results: list[BatchResult] = []
        for req in requests:
            outputs = [
                f"output_{i} for {p[:20]}"
                for i, p in enumerate(req.prompts)
            ]
            token_counts = [len(o.split()) for o in outputs]
            results.append(
                BatchResult(
                    request_id=req.request_id,
                    outputs=outputs,
                    token_counts=token_counts,
                    latency_ms=latency_ms,
                )
            )
        return results

    def run_once(self) -> list[BatchResult]:
        batch = self.dequeue_batch()
        if not batch:
            return []
        return self.process_batch(batch)

    def queue_depth(self) -> int:
        return len(self._queue)

    def stats(self) -> dict:
        return {
            "queued": len(self._queue),
            "max_queue_size": self._max_queue_size,
            "max_batch_size": self._max_batch_size,
        }
