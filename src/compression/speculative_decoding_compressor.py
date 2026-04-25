"""Speculative decoding compressor: draft-verify token stream compression."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import torch


@dataclass
class SDCConfig:
    draft_steps: int = 4
    acceptance_threshold: float = 0.8


class _DraftEntry(NamedTuple):
    token_id: int
    logprob: float


class DraftBuffer:
    """Circular buffer of draft token predictions.

    Parameters
    ----------
    capacity:
        Maximum number of draft tokens to store (wraps when full).
    """

    def __init__(self, capacity: int = 16) -> None:
        self._capacity = capacity
        self._buf: deque[_DraftEntry] = deque(maxlen=capacity)

    def push(self, token_id: int, logprob: float) -> None:
        """Append a draft prediction to the buffer."""
        self._buf.append(_DraftEntry(token_id=token_id, logprob=logprob))

    def accept_prefix(self, n: int) -> list[int]:
        """Return the first *n* token IDs from the buffer (non-destructive)."""
        n = min(n, len(self._buf))
        return [entry.token_id for entry in list(self._buf)[:n]]

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class SpeculativeCompressionMetrics:
    acceptance_rate: float
    tokens_per_step: float
    compression_ratio: float


class SpeculativeDecodingCompressor:
    """Token stream compressor using the speculative / draft-verify pattern.

    The compressor evaluates draft tokens against a target distribution and
    accepts tokens whose target log-probability exceeds
    ``log(acceptance_threshold)``.

    Parameters
    ----------
    config:
        :class:`SDCConfig` controlling draft steps and acceptance threshold.
    """

    def __init__(self, config: SDCConfig | None = None) -> None:
        self.config = config or SDCConfig()
        self._log_threshold = math.log(self.config.acceptance_threshold)
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0
        self._total_steps = 0
        self._draft_buffer = DraftBuffer(capacity=self.config.draft_steps * 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_step(
        self,
        draft_logits: torch.Tensor,
        target_logprobs: torch.Tensor,
    ) -> list[int]:
        """Accept draft tokens where target_logprob >= log(acceptance_threshold).

        Parameters
        ----------
        draft_logits:
            1-D or 2-D tensor of draft model logits.  If 2-D, shape is
            ``(seq_len, vocab_size)``; the greedy token is taken per position.
        target_logprobs:
            Log-probabilities from the target model for each draft position,
            shape ``(seq_len,)`` or ``(seq_len, vocab_size)``.

        Returns
        -------
        list[int]
            Accepted token IDs (may be empty).
        """
        # --- normalise draft_logits to token ids ---
        if draft_logits.dim() == 1:
            # single position: treat as logits over vocab
            draft_token_ids = [int(draft_logits.argmax().item())]
            n_draft = 1
        else:
            draft_token_ids = draft_logits.argmax(dim=-1).tolist()
            n_draft = len(draft_token_ids)

        # --- normalise target_logprobs to per-token scalar ---
        if target_logprobs.dim() == 2:
            # gather log-prob of the draft token at each position
            t_ids = torch.tensor(draft_token_ids, dtype=torch.long)
            per_token_lp = target_logprobs[
                torch.arange(n_draft), t_ids
            ]
        else:
            per_token_lp = target_logprobs[:n_draft]

        # --- accept / reject ---
        accepted: list[int] = []
        for token_id, lp in zip(draft_token_ids, per_token_lp.tolist()):
            if lp >= self._log_threshold:
                accepted.append(token_id)
            else:
                break  # stop at first rejection (prefix acceptance)

        self._total_draft_tokens += n_draft
        self._total_accepted_tokens += len(accepted)
        self._total_steps += 1

        # push accepted tokens into draft buffer
        lp_list = per_token_lp.tolist()
        for i, token_id in enumerate(accepted):
            self._draft_buffer.push(token_id, lp_list[i])

        return accepted

    def get_metrics(self) -> SpeculativeCompressionMetrics:
        """Return aggregated compression metrics."""
        if self._total_steps == 0:
            return SpeculativeCompressionMetrics(
                acceptance_rate=0.0,
                tokens_per_step=0.0,
                compression_ratio=1.0,
            )
        acceptance_rate = (
            self._total_accepted_tokens / self._total_draft_tokens
            if self._total_draft_tokens > 0
            else 0.0
        )
        tokens_per_step = self._total_accepted_tokens / self._total_steps
        # compression ratio: accepted / draft (higher = better acceptance)
        compression_ratio = (
            self._total_accepted_tokens / self._total_draft_tokens
            if self._total_draft_tokens > 0
            else 1.0
        )
        return SpeculativeCompressionMetrics(
            acceptance_rate=acceptance_rate,
            tokens_per_step=tokens_per_step,
            compression_ratio=compression_ratio,
        )

    @property
    def draft_buffer(self) -> DraftBuffer:
        return self._draft_buffer
