"""Sparse attention mask: local window + global tokens (Longformer-style)."""

import torch
from dataclasses import dataclass


@dataclass
class SparseAttentionConfig:
    window_size: int = 256          # local window radius (total = 2*window_size + 1)
    causal: bool = True             # if True, only look backward (causal window)


def make_longformer_mask(
    seq_len: int,
    window_size: int,
    global_token_indices: list[int] | None = None,
    causal: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build Longformer-style sparse attention mask.

    Combines:
    1. Local window attention: each token attends to ±window_size neighbors
       (or only backward if causal=True)
    2. Global tokens: specified indices attend to ALL tokens and vice versa

    Mask convention: 0.0 = allowed, -inf = blocked (additive mask for SDPA).

    Args:
        seq_len: sequence length
        window_size: local attention radius
        global_token_indices: indices of global tokens (attend to/from everywhere)
        causal: if True, only attend to past positions (no future in window)
        device: target device

    Returns:
        (seq_len, seq_len) float tensor
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

    # Build local window
    rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)

    if causal:
        # Attend to: col <= row AND col >= row - window_size
        allowed = (cols <= rows) & (cols >= rows - window_size)
    else:
        # Attend to: |col - row| <= window_size
        allowed = (cols - rows).abs() <= window_size

    mask[allowed] = 0.0

    # Global tokens: attend to all and all attend to them
    if global_token_indices:
        for g in global_token_indices:
            if 0 <= g < seq_len:
                mask[g, :] = 0.0      # global token attends to all
                mask[:, g] = 0.0      # all attend to global token

    return mask


def make_causal_longformer_mask(
    seq_len: int,
    window_size: int,
    global_token_indices: list[int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convenience wrapper for causal (decoder) Longformer mask."""
    return make_longformer_mask(seq_len, window_size, global_token_indices, causal=True, device=device)


def count_attended_tokens(mask: torch.Tensor) -> torch.Tensor:
    """Count how many tokens each position attends to.

    Returns (seq_len,) int tensor — 0.0 entries in mask = attended.
    """
    return (mask == 0.0).sum(dim=-1)
