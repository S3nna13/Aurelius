"""Sliding-window causal attention mask builder (local attention band).

Implements the standard **causal + distance** mask used by sliding-window
transformers (e.g. Longformer-style local windows, Beltagy et al. 2020) as a
dense additive mask suitable for ``softmax(QK^T / sqrt(d) + mask)``.

Returned tensor has shape ``[1, 1, T, T]`` so it can broadcast over batch and
heads without extra copies in reference implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SlidingWindowCausalMaskBuilder:
    """Build additive attention masks for causal sliding-window attention.

    Parameters
    ----------
    window_size:
        Maximum distance ``i - j`` (inclusive) that position ``i`` may attend
        to at column ``j``. Must be >= 1.
    neg_value:
        Large negative used for masked positions. Defaults to ``-inf`` for
        floating dtypes and a finite floor for low-precision dtypes where
        ``-inf`` is undesirable in tests.
    """

    window_size: int
    neg_value: float | None = None

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")

    def build(self, seq_len: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return additive mask ``[1, 1, seq_len, seq_len]`` (0 = keep, neg = mask)."""
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        causal = col <= row
        dist = row - col
        band = dist < self.window_size
        allowed = causal & band
        neg = self.neg_value
        if neg is None:
            neg = float("-inf") if dtype in (torch.float32, torch.float64, torch.bfloat16) else -1e4
        mask = torch.full((seq_len, seq_len), neg, device=device, dtype=dtype)
        mask = mask.masked_fill(allowed, 0.0)
        return mask.view(1, 1, seq_len, seq_len)


__all__ = ["SlidingWindowCausalMaskBuilder"]
