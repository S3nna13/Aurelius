"""Token-level augmentation transforms for 1D token ID tensors.

Three transforms that operate directly on token IDs without
any tokenizer dependency. The caller passes token IDs directly.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class RandomTokenMask:
    """Replace random tokens with mask_id or a random token from [0, vocab_size).

    Args:
        p: Probability of masking each token.
        mask_id: Token ID to use for masking. If None, use a random token.
        vocab_size: Required when mask_id is None.
        seed: Optional seed for reproducibility.
    """

    p: float = 0.15
    mask_id: int | None = None
    vocab_size: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mask_id is None and self.vocab_size is None:
            raise ValueError("vocab_size is required when mask_id is None")

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply masking. Input: 1D (S,) or 2D (B, S) tensor. Returns same shape."""
        output = input_ids.clone()

        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed)

        mask = torch.rand(input_ids.shape, generator=gen) < self.p

        if self.mask_id is not None:
            output[mask] = self.mask_id
        else:
            assert self.vocab_size is not None
            output[mask] = torch.randint(
                0, self.vocab_size, (mask.sum().item(),), generator=gen
            )

        return output


@dataclass
class TokenDropout:
    """Delete random tokens from the sequence.

    Args:
        p: Probability of dropping each token.
    """

    p: float = 0.1

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply dropout. Input must be 1D (S,). Returns shortened 1D tensor.
        Never drops all tokens -- guarantees at least 1 token remains."""
        keep_mask = torch.rand(len(input_ids)) >= self.p

        if keep_mask.sum() == 0:
            keep_mask[0] = True

        return input_ids[keep_mask]


@dataclass
class SpanCorruption:
    """T5-style span corruption: replace random spans with sentinel tokens.

    Randomly selects spans of mean_span_length, replaces each with a single
    sentinel token ID (sentinel_start, sentinel_start+1, ...).

    Args:
        p: Fraction of tokens to corrupt.
        mean_span_length: Average length of corrupted spans.
        sentinel_start: First sentinel token ID (e.g., vocab_size - 100).
    """

    p: float = 0.15
    mean_span_length: int = 3
    sentinel_start: int = 32000

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply span corruption. Input: 1D (S,). Returns 1D tensor (shorter).
        If no spans selected, returns input unchanged."""
        S = len(input_ids)
        if S == 0:
            return input_ids.clone()

        n_corrupt = max(1, int(S * self.p))
        n_spans = max(1, n_corrupt // self.mean_span_length)

        # Build a boolean mask of which positions are corrupted
        corrupt_mask = torch.zeros(S, dtype=torch.bool)
        for i in range(n_spans):
            start = torch.randint(0, S, ()).item()
            length = max(
                1,
                int(
                    torch.poisson(torch.tensor(float(self.mean_span_length))).item()
                ),
            )
            end = min(S, start + length)
            corrupt_mask[start:end] = True

        # Rebuild sequence: copy non-corrupt tokens, insert sentinels for spans
        result: list[int] = []
        in_span = False
        span_idx = 0
        for i in range(S):
            if corrupt_mask[i]:
                if not in_span:
                    result.append(self.sentinel_start + span_idx)
                    span_idx += 1
                    in_span = True
            else:
                in_span = False
                result.append(input_ids[i].item())

        if not result:
            return input_ids[:1].clone()

        return torch.tensor(result, dtype=input_ids.dtype)
