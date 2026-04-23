"""Continuous batching v2: priority queues, preemption, memory budgeting."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RequestPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class BatchRequest:
    prompt_tokens: int
    max_new_tokens: int
    priority: RequestPriority = RequestPriority.NORMAL
    generated_tokens: int = 0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class BatchingConfig:
    max_batch_tokens: int = 4096
    max_batch_size: int = 32
    preemption_enabled: bool = True


class ContinuousBatcherV2:
    """Continuous batching scheduler with priority, preemption, and memory budgeting."""

    _PRIORITY_SCORES = {
        RequestPriority.CRITICAL: 0,
        RequestPriority.HIGH: 1,
        RequestPriority.NORMAL: 2,
        RequestPriority.LOW: 3,
    }

    def __init__(self, config: Optional[BatchingConfig] = None) -> None:
        self.config = config if config is not None else BatchingConfig()
        self._queue: List[BatchRequest] = []

    def enqueue(self, request: BatchRequest) -> None:
        """Add a request to the queue."""
        self._queue.append(request)

    def _priority_score(self, req: BatchRequest) -> int:
        """Return numeric priority (lower = higher priority)."""
        return self._PRIORITY_SCORES.get(req.priority, 2)

    def next_batch(self) -> List[BatchRequest]:
        """Greedily fill up to max_batch_tokens and max_batch_size.

        Sort queue by priority_score, take from front while token budget allows.
        """
        sorted_queue = sorted(self._queue, key=self._priority_score)
        selected: List[BatchRequest] = []
        token_budget = self.config.max_batch_tokens

        for req in sorted_queue:
            if len(selected) >= self.config.max_batch_size:
                break
            req_tokens = req.prompt_tokens + req.generated_tokens
            if req_tokens <= token_budget:
                selected.append(req)
                token_budget -= req_tokens

        return selected

    def mark_done(self, request_id: str) -> bool:
        """Remove a request from the queue by ID. Returns True if found."""
        for i, req in enumerate(self._queue):
            if req.request_id == request_id:
                self._queue.pop(i)
                return True
        return False

    def preempt_lowest(self, n: int = 1) -> List[BatchRequest]:
        """If preemption_enabled: remove n lowest-priority requests, return them."""
        if not self.config.preemption_enabled:
            return []

        # Sort descending by priority score (highest score = lowest priority)
        sorted_by_lowest = sorted(self._queue, key=self._priority_score, reverse=True)
        to_remove = sorted_by_lowest[:n]

        for req in to_remove:
            self._queue.remove(req)

        return to_remove

    def queue_depth(self) -> int:
        """Return current number of requests in the queue."""
        return len(self._queue)

    def token_utilization(self, batch: List[BatchRequest]) -> float:
        """Fraction of max_batch_tokens used by batch."""
        total = sum(req.prompt_tokens + req.generated_tokens for req in batch)
        return total / self.config.max_batch_tokens
