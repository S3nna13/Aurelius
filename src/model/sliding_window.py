"""Sliding window attention mask generation.

Restricts each token to attend only to the W most recent tokens,
producing additive masks compatible with scaled dot-product attention.
"""

import torch
from dataclasses import dataclass


@dataclass
class SlidingWindowConfig:
    window_size: int = 512  # each token attends to this many past tokens
    include_self: bool = True  # always attend to current position


def make_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a (seq_len, seq_len) additive attention mask for sliding window attention.

    Mask values:
    - 0.0 where attention is allowed (within causal window)
    - -inf where attention is blocked (future tokens or beyond window)

    Token i attends to tokens max(0, i - window_size + 1) .. i (inclusive).

    Returns:
        (seq_len, seq_len) float tensor.
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    # Allowed: col <= row (causal) AND col >= row - window_size + 1 (window)
    allowed = (cols <= rows) & (cols >= rows - window_size + 1)
    mask = torch.where(
        allowed,
        torch.tensor(0.0, device=device),
        torch.tensor(float("-inf"), device=device),
    )
    return mask


def make_sliding_window_mask_batched(
    seq_len: int,
    window_size: int,
    batch_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return (batch_size, 1, seq_len, seq_len) mask for use with SDPA."""
    mask = make_sliding_window_mask(seq_len, window_size, device)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def combine_with_prefix_mask(
    sliding_mask: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> torch.Tensor:
    """Combine a sliding window mask with a prefix-LM mask.

    Result allows attention where EITHER mask allows it
    (i.e., take the elementwise maximum).

    Args:
        sliding_mask: (S, S) sliding window mask
        prefix_mask: (S, S) prefix-LM mask (both additive, 0.0 or -inf)

    Returns:
        (S, S) combined mask.
    """
    return torch.maximum(sliding_mask, prefix_mask)
