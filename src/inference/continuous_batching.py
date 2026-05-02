"""Continuous batching utilities for autoregressive serving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class GenerationRequest:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_step: int = 0
    generated_tokens: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(self.max_new_tokens - self.generated_tokens, 0)

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens


@dataclass(frozen=True)
class BatchStep:
    prefill_ids: list[str]
    decode_ids: list[str]

    @property
    def all_ids(self) -> list[str]:
        return self.prefill_ids + self.decode_ids


class ContinuousBatchScheduler:
    """Simple fair scheduler for interleaving prefill and decode work."""

    def __init__(self, max_batch_size: int, max_prefill_tokens: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_prefill_tokens <= 0:
            raise ValueError("max_prefill_tokens must be positive")
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self._pending_prefill: deque[GenerationRequest] = deque()
        self._active_decode: deque[GenerationRequest] = deque()
        self._known_ids: set[str] = set()
        self.current_step = 0

    def add_request(self, request: GenerationRequest) -> None:
        """Register a new request for future scheduling."""
        if request.request_id in self._known_ids:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        self._known_ids.add(request.request_id)
        self._pending_prefill.append(request)

    def has_pending(self) -> bool:
        return bool(self._pending_prefill or self._active_decode)

    def active_request_ids(self) -> list[str]:
        return [request.request_id for request in self._active_decode]

    def schedule_step(self) -> BatchStep:
        """Construct the next mixed prefill/decode batch."""
        prefill_ids: list[str] = []
        decode_ids: list[str] = []
        remaining_slots = self.max_batch_size
        remaining_prefill_tokens = self.max_prefill_tokens

        active_count = len(self._active_decode)
        for _ in range(min(active_count, remaining_slots)):
            request = self._active_decode.popleft()
            if request.finished:
                self._known_ids.discard(request.request_id)
                continue
            decode_ids.append(request.request_id)
            self._active_decode.append(request)
            remaining_slots -= 1

        deferred: deque[GenerationRequest] = deque()
        while self._pending_prefill and remaining_slots > 0:
            request = self._pending_prefill.popleft()
            if request.prompt_tokens <= remaining_prefill_tokens:
                prefill_ids.append(request.request_id)
                self._active_decode.append(request)
                remaining_prefill_tokens -= request.prompt_tokens
                remaining_slots -= 1
            else:
                deferred.append(request)
        self._pending_prefill.extendleft(reversed(deferred))

        self.current_step += 1
        return BatchStep(prefill_ids=prefill_ids, decode_ids=decode_ids)

    def mark_step_complete(self, completed_ids: list[str], generated_tokens: int = 1) -> None:
        """Update active requests after one decoding iteration."""
        completed = set(completed_ids)
        kept: deque[GenerationRequest] = deque()
        while self._active_decode:
            request = self._active_decode.popleft()
            if request.request_id in completed:
                request.generated_tokens += generated_tokens
            if request.finished:
                self._known_ids.discard(request.request_id)
            else:
                kept.append(request)
        self._active_decode = kept
