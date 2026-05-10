"""Continuous (iteration-level) batching for LLM inference throughput.

Instead of static batching where all sequences finish together, continuous
batching inserts new requests into the batch at each iteration as slots free up,
dramatically improving GPU utilization.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class Request:
    """A single inference request with its state."""

    request_id: str
    input_ids: list[int]  # prompt tokens
    max_new_tokens: int = 256
    temperature: float = 1.0
    # internal state:
    generated_ids: list[int] = field(default_factory=list)
    is_finished: bool = False
    finish_reason: str = ""  # "length" | "eos" | "stop"


class ContinuousBatcher:
    """
    Continuous (iteration-level) batching manager.

    At each step:
    1. If batch has room (< max_batch_size), pull from waiting_queue and add
    2. Run one decode step on all active sequences together
    3. Append next tokens, mark finished sequences
    4. Return finished results as they complete

    Args:
        model: AureliusTransformer (or any model with (loss, logits, pkv) API)
        tokenizer_encode: callable str->list[int]
        eos_token_id: int (default 2)
        max_batch_size: int (default 8)
        max_seq_len: int (default 512)
    """

    def __init__(
        self,
        model,
        tokenizer_encode: Callable[[str], list[int]],
        eos_token_id: int = 2,
        max_batch_size: int = 8,
        max_seq_len: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.eos_token_id = eos_token_id
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.waiting_queue: deque[Request] = deque()
        self.active_requests: list[Request] = []
        self.completed: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the waiting queue."""
        self.waiting_queue.append(request)

    def _fill_batch(self) -> None:
        """Move requests from waiting_queue to active_requests up to max_batch_size."""
        while self.waiting_queue and len(self.active_requests) < self.max_batch_size:
            self.active_requests.append(self.waiting_queue.popleft())

    def step(self) -> list[Request]:
        """
        Run one iteration:
        1. Fill batch from queue
        2. Build padded input tensor from active_requests (prompt+generated so far)
        3. Forward pass -> logits -> sample/argmax next token
        4. Append token, check EOS/length
        5. Remove finished requests, add to completed
        Returns: list of newly completed Request objects this step
        """
        self._fill_batch()

        if not self.active_requests:
            return []

        # Build token sequences: prompt + generated so far, truncated to max_seq_len
        sequences = []
        for req in self.active_requests:
            seq = req.input_ids + req.generated_ids
            seq = seq[-self.max_seq_len :]  # truncate if too long
            sequences.append(seq)

        # Pad to the same length (right-pad with 0)
        max_len = max(len(s) for s in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(seq + [0] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long)

        # Forward pass: model returns (loss, logits, past_key_values)
        with torch.no_grad():
            _loss, logits, _pkv = self.model(input_ids)
        # logits shape: (B, T, vocab_size)

        # For each sequence, pick the logits at the last non-pad position
        newly_completed: list[Request] = []

        for i, req in enumerate(self.active_requests):
            seq_len = len(sequences[i])  # actual (non-padded) length
            last_pos = seq_len - 1  # index of last real token
            token_logits = logits[i, last_pos, :]  # (vocab_size,)

            # Sample or argmax
            if req.temperature == 0.0:
                next_token = int(torch.argmax(token_logits).item())
            else:
                scaled = token_logits / req.temperature
                probs = torch.softmax(scaled, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            req.generated_ids.append(next_token)

            # Check termination conditions
            if next_token == self.eos_token_id:
                req.is_finished = True
                req.finish_reason = "eos"
            elif len(req.generated_ids) >= req.max_new_tokens:
                req.is_finished = True
                req.finish_reason = "length"

        # Partition active requests into finished and still-running
        still_active = []
        for req in self.active_requests:
            if req.is_finished:
                self.completed.append(req)
                newly_completed.append(req)
            else:
                still_active.append(req)

        self.active_requests = still_active
        return newly_completed

    def run_until_complete(self, requests: list[Request]) -> list[Request]:
        """Add all requests and step until all complete. Returns completed requests."""
        for req in requests:
            self.add_request(req)

        while self.waiting_queue or self.active_requests:
            self.step()

        return list(self.completed)

    @property
    def stats(self) -> dict:
        """Return {'active': int, 'waiting': int, 'completed': int}"""
        return {
            "active": len(self.active_requests),
            "waiting": len(self.waiting_queue),
            "completed": len(self.completed),
        }
