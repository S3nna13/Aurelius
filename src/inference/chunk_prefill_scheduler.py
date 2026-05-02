"""Chunked Prefill Scheduler — Sarathi-Serve / vLLM 2024 style.

Reference: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM
Inference with Sarathi-Serve", arXiv:2403.02310, 2024.

Motivation
----------
Naive batched inference either blocks decode requests while prefilling a long
prompt (causing latency spikes) or never batches prefill with decode.  Chunked
prefill splits long prompts into fixed-size chunks and interleaves prefill
chunks with decode steps in each batch, keeping GPU utilization high while
bounding per-step decode latency.

Algorithm per ``schedule_batch``
---------------------------------
1. Fill decode slots first (up to ``max_decode_tokens_per_batch`` requests).
2. Fill prefill slots with remaining capacity (up to
   ``max_prefill_tokens_per_batch`` total tokens).  Each prefill request
   contributes ``min(chunk_size, remaining_prompt_tokens)`` tokens.
3. Advance internal state: increment ``prefill_done`` / ``tokens_generated``.
4. Transition requests whose prefill is complete to ``"decoding"``.
5. Mark decode requests that have hit ``max_output_tokens`` as ``"completed"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """A single inference request."""

    request_id: str
    prompt_tokens: int  # total prompt length in tokens
    max_output_tokens: int  # maximum tokens to generate
    priority: int = 0  # lower value = higher priority


@dataclass
class RequestState:
    """Mutable per-request scheduler state."""

    request: Request
    prefill_done: int = 0  # tokens prefilled so far
    tokens_generated: int = 0
    status: str = "prefilling"  # "prefilling" | "decoding" | "completed"


@dataclass
class BatchSlot:
    """A single request's contribution to one batch."""

    request_id: str
    slot_type: str  # "prefill" or "decode"
    n_tokens: int  # tokens processed this slot


@dataclass
class ChunkPrefillConfig:
    """Scheduler hyper-parameters."""

    max_batch_size: int = 32
    chunk_size: int = 512  # max prefill tokens per request per batch
    max_prefill_tokens_per_batch: int = 4096
    max_decode_tokens_per_batch: int = 32


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ChunkPrefillScheduler:
    """Interleaved chunked-prefill / decode scheduler.

    Parameters
    ----------
    config:
        Scheduler configuration.  Defaults to :class:`ChunkPrefillConfig`.
    """

    def __init__(self, config: ChunkPrefillConfig | None = None) -> None:
        self._config = config if config is not None else ChunkPrefillConfig()
        # Insertion-ordered dict keyed by request_id.
        self._states: dict[str, RequestState] = {}
        # Telemetry accumulators.
        self._batch_prefill_tokens: list[int] = []
        self._batch_decode_slots: list[int] = []

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Enqueue a new request.

        Raises
        ------
        ValueError
            If a request with the same ``request_id`` already exists.
        """
        if request.request_id in self._states:
            raise ValueError(f"duplicate request_id: {request.request_id!r}")
        self._states[request.request_id] = RequestState(request=request)

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def schedule_batch(self) -> list[BatchSlot]:
        """Build the next batch by interleaving decode and prefill slots.

        Returns
        -------
        list[BatchSlot]
            Ordered list of slots; decode slots appear before prefill slots.
        """
        cfg = self._config
        slots: list[BatchSlot] = []

        # Sort active requests by priority (lower value = higher priority).
        # Stable sort preserves insertion order for equal-priority items.
        active = sorted(
            [s for s in self._states.values() if s.status != "completed"],
            key=lambda s: s.request.priority,
        )

        # --- Phase 1: decode slots ---
        decode_count = 0
        for state in active:
            if state.status != "decoding":
                continue
            if decode_count >= cfg.max_decode_tokens_per_batch:
                break
            if len(slots) >= cfg.max_batch_size:
                break
            slots.append(
                BatchSlot(
                    request_id=state.request.request_id,
                    slot_type="decode",
                    n_tokens=1,
                )
            )
            decode_count += 1

        # --- Phase 2: prefill slots ---
        prefill_tokens_used = 0
        for state in active:
            if state.status != "prefilling":
                continue
            if prefill_tokens_used >= cfg.max_prefill_tokens_per_batch:
                break
            if len(slots) >= cfg.max_batch_size:
                break
            remaining = state.request.prompt_tokens - state.prefill_done
            chunk = min(
                cfg.chunk_size, remaining, cfg.max_prefill_tokens_per_batch - prefill_tokens_used
            )
            if chunk <= 0:
                continue
            slots.append(
                BatchSlot(
                    request_id=state.request.request_id,
                    slot_type="prefill",
                    n_tokens=chunk,
                )
            )
            prefill_tokens_used += chunk

        # --- Phase 3: advance state ---
        for slot in slots:
            state = self._states[slot.request_id]
            if slot.slot_type == "prefill":
                state.prefill_done += slot.n_tokens
                # Transition once all prompt tokens have been prefilled.
                if state.prefill_done >= state.request.prompt_tokens:
                    state.prefill_done = state.request.prompt_tokens
                    state.status = "decoding"
            else:  # decode
                state.tokens_generated += 1
                if state.tokens_generated >= state.request.max_output_tokens:
                    state.status = "completed"

        # --- Telemetry ---
        self._batch_prefill_tokens.append(
            sum(s.n_tokens for s in slots if s.slot_type == "prefill")
        )
        self._batch_decode_slots.append(sum(1 for s in slots if s.slot_type == "decode"))

        return slots

    # ------------------------------------------------------------------
    # External completion signal
    # ------------------------------------------------------------------

    def complete_decode(self, request_id: str) -> None:
        """Mark a decoding request as completed (e.g., EOS generated).

        Parameters
        ----------
        request_id:
            ID of the request to mark completed.

        Raises
        ------
        KeyError
            If the request_id is unknown.
        """
        if request_id not in self._states:
            raise KeyError(f"unknown request_id: {request_id!r}")
        self._states[request_id].status = "completed"

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def pending_requests(self) -> int:
        """Number of requests that are not completed."""
        return sum(1 for s in self._states.values() if s.status != "completed")

    def prefilling_requests(self) -> int:
        """Number of requests currently in the prefilling phase."""
        return sum(1 for s in self._states.values() if s.status == "prefilling")

    def decoding_requests(self) -> int:
        """Number of requests currently in the decoding phase."""
        return sum(1 for s in self._states.values() if s.status == "decoding")

    def completed_requests(self) -> int:
        """Number of completed requests."""
        return sum(1 for s in self._states.values() if s.status == "completed")

    # ------------------------------------------------------------------
    # Utilization
    # ------------------------------------------------------------------

    def utilization(self) -> dict:
        """Mean per-batch statistics across all batches scheduled so far.

        Returns
        -------
        dict
            ``{"prefill_tokens_per_batch": float, "decode_slots_per_batch": float}``
        """
        return {
            "prefill_tokens_per_batch": (
                mean(self._batch_prefill_tokens) if self._batch_prefill_tokens else 0.0
            ),
            "decode_slots_per_batch": (
                mean(self._batch_decode_slots) if self._batch_decode_slots else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BatchSlot",
    "ChunkPrefillConfig",
    "ChunkPrefillScheduler",
    "Request",
    "RequestState",
]
