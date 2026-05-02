"""Orca-style continuous batching scheduler.

Reference: Yu et al., "Orca: A Distributed Serving System for Transformer-Based
Generative Models", OSDI 2022.

This module implements token-level iteration scheduling only. It does not run
a model forward pass; callers drive the scheduler by:

    step = scheduler.build_step()
    if step is not None:
        token_map = step_fn(step)  # user-supplied model runner
        scheduler.receive_tokens(token_map)
    for finished in scheduler.completed():
        ...

Each call to ``build_step`` admits queued requests up to ``max_batch_size`` and
emits a :class:`BatchStep` describing per-request inputs: full prompt tokens on
the very first step (prefill) or just the most recently generated token on
subsequent steps (decode).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

QUEUED = "queued"
PREFILL = "prefill"
DECODING = "decoding"
COMPLETED = "completed"


@dataclass
class InferenceRequest:
    """A single inference request tracked by the scheduler."""

    request_id: str
    prompt_tokens: list[int]
    max_new_tokens: int
    eos_token_id: int
    state: str = QUEUED
    generated: list[int] = field(default_factory=list)


@dataclass
class BatchStep:
    """A single iteration-level batch to be executed by the caller."""

    request_ids: list[str]
    input_ids: list[list[int]]
    is_prefill: list[bool]


class ContinuousBatchingScheduler:
    """Orca-style continuous batching scheduler.

    Parameters
    ----------
    max_batch_size:
        Maximum number of requests that can be active (prefill or decoding) at
        any one time.
    max_seq_len:
        Hard cap on prompt length + generated length. A request whose total
        sequence length exceeds this after receiving a token is completed.
    """

    def __init__(self, max_batch_size: int = 16, max_seq_len: int = 2048) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._queue: OrderedDict[str, InferenceRequest] = OrderedDict()
        self._active: OrderedDict[str, InferenceRequest] = OrderedDict()
        self._completed: OrderedDict[str, InferenceRequest] = OrderedDict()
        self._known_ids: set = set()

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------
    def enqueue(self, request: InferenceRequest) -> None:
        """Add a new request to the FIFO queue."""
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if request.request_id in self._known_ids:
            raise ValueError(f"duplicate request_id: {request.request_id!r}")
        if not request.prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        request.state = QUEUED
        self._queue[request.request_id] = request
        self._known_ids.add(request.request_id)

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------
    def _admit(self) -> None:
        while self._queue and len(self._active) < self.max_batch_size:
            rid, req = self._queue.popitem(last=False)
            req.state = PREFILL
            self._active[rid] = req

    def build_step(self) -> BatchStep | None:
        """Construct the next iteration-level batch.

        Returns ``None`` when there are no active requests and the queue is
        empty (nothing to execute).
        """
        self._admit()
        if not self._active:
            return None

        request_ids: list[str] = []
        input_ids: list[list[int]] = []
        is_prefill: list[bool] = []

        for rid, req in self._active.items():
            if req.state == PREFILL:
                request_ids.append(rid)
                input_ids.append(list(req.prompt_tokens))
                is_prefill.append(True)
            elif req.state == DECODING:
                # Feed only the most recently generated token.
                if not req.generated:
                    raise RuntimeError(f"request {rid!r} is decoding but has no generated tokens")
                request_ids.append(rid)
                input_ids.append([req.generated[-1]])
                is_prefill.append(False)
            else:
                raise RuntimeError(f"request {rid!r} in unexpected state {req.state!r}")

        return BatchStep(
            request_ids=request_ids,
            input_ids=input_ids,
            is_prefill=is_prefill,
        )

    # ------------------------------------------------------------------
    # Token ingestion
    # ------------------------------------------------------------------
    def receive_tokens(self, request_id_to_token: dict[str, int]) -> None:
        """Apply freshly sampled tokens to their active requests."""
        if not isinstance(request_id_to_token, dict):
            raise TypeError("request_id_to_token must be a dict")

        # Validate all ids first; do not mutate on error.
        for rid in request_id_to_token:
            if rid not in self._active:
                raise KeyError(f"unknown or inactive request_id: {rid!r}")

        for rid, token in request_id_to_token.items():
            req = self._active[rid]
            req.generated.append(int(token))

            total_len = len(req.prompt_tokens) + len(req.generated)
            hit_eos = int(token) == req.eos_token_id
            hit_max_new = len(req.generated) >= req.max_new_tokens
            hit_max_seq = total_len > self.max_seq_len

            if hit_eos or hit_max_new or hit_max_seq:
                req.state = COMPLETED
                del self._active[rid]
                self._completed[rid] = req
            else:
                req.state = DECODING

    # ------------------------------------------------------------------
    # Completion drain & stats
    # ------------------------------------------------------------------
    def completed(self) -> list[InferenceRequest]:
        """Drain and return all completed requests in completion order."""
        drained = list(self._completed.values())
        self._completed.clear()
        return drained

    def stats(self) -> dict[str, int]:
        return {
            "active": len(self._active),
            "queued": len(self._queue),
            "completed": len(self._completed),
        }


__all__ = [
    "InferenceRequest",
    "BatchStep",
    "ContinuousBatchingScheduler",
]
