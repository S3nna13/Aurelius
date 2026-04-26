"""Flash-style sliding window attention with online softmax and tiled computation.

Combines two ideas:
1. Sliding window masking — each token attends to at most `window_size` neighbours.
2. Flash Attention-style tiled computation — process query blocks one at a time,
   accumulating output with a running (max, sum) normalisation to avoid
   materialising the full T×T attention matrix.

This is intentionally distinct from:
- sliding_window.py  (additive float masks, no tiling, no online softmax)
- flash_attention.py (tiling + online softmax, but full-sequence, no window)
- local_global_attention.py (GQA + RoPE interleaved layers, no tiling)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindowConfig:
    """Hyperparameters for sliding window attention.

    Attributes:
        window_size:   Number of tokens each position can attend to (one-sided
                       for causal; full band width for bidirectional).
        global_tokens: Number of leading tokens that use *full* (global)
                       attention regardless of window_size.  Used by
                       :class:`GlobalLocalAttention`.
        causal:        If True the mask is lower-triangular (autoregressive).
        block_size:    Tile size along the query dimension for
                       :func:`tiled_attention`.
    """

    window_size: int = 512
    global_tokens: int = 64
    causal: bool = True
    block_size: int = 64


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = True,
) -> Tensor:
    """Build a boolean sliding-window attention mask.

    Args:
        seq_len:     Sequence length T.
        window_size: Maximum attention distance (exclusive upper bound on gap).
        causal:      If True, also enforce ``j <= i`` (no future tokens).

    Returns:
        Boolean tensor of shape ``(T, T)``.  ``True`` means *attend*.

    Notes:
        - Causal:      attend iff ``j <= i`` AND ``i - j < window_size``
        - Bidirectional: attend iff ``|i - j| < window_size``
    """
    i = torch.arange(seq_len).unsqueeze(1)  # (T, 1)
    j = torch.arange(seq_len).unsqueeze(0)  # (1, T)

    if causal:
        mask = (j <= i) & ((i - j) < window_size)
    else:
        mask = (i - j).abs() < window_size

    return mask  # (T, T) bool


# ---------------------------------------------------------------------------
# Online softmax
# ---------------------------------------------------------------------------


def online_softmax(scores: Tensor, mask: Tensor | None = None) -> Tensor:
    """Numerically stable softmax using the online (subtract-max) trick.

    Args:
        scores: Attention logits, arbitrary shape ``(..., S)``.
        mask:   Optional boolean tensor broadcastable to ``scores``.
                Positions where ``mask`` is ``False`` are set to ``-1e9``
                before the softmax so they become effectively zero.

    Returns:
        Softmax probabilities, same shape as ``scores``.
    """
    if mask is not None:
        # False positions should not be attended to
        scores = scores.masked_fill(~mask, -1e9)

    # Subtract max for numerical stability (online trick)
    max_scores = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True)
    return exp_scores / sum_exp


# ---------------------------------------------------------------------------
# Tiled attention with sliding window mask
# ---------------------------------------------------------------------------


def tiled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    block_size: int,
) -> Tensor:
    """Compute masked attention in tiles over the query dimension.

    Simulates the memory-efficient access pattern of Flash Attention by
    processing ``block_size`` query rows at a time.  Within each tile the
    full K/V sequence is visited and a running (row_max, row_sum)
    normalisation merges contributions from all key blocks.

    Args:
        q:          Query tensor  ``(B, H, T, D)``.
        k:          Key tensor    ``(B, H, T, D)``.
        v:          Value tensor  ``(B, H, T, D)``.
        mask:       Boolean mask  ``(T, T)``.  ``True`` = attend.
        block_size: Number of query rows per tile.

    Returns:
        Output tensor ``(B, H, T, D)``.
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    device = q.device

    output = torch.zeros_like(q)

    for q_start in range(0, T, block_size):
        q_end = min(q_start + block_size, T)
        q_block = q[:, :, q_start:q_end, :]  # (B, H, Tq, D)
        Tq = q_end - q_start

        # Running accumulators for online normalisation
        running_max = torch.full((B, H, Tq, 1), float("-inf"), device=device, dtype=q.dtype)
        running_sum = torch.zeros(B, H, Tq, 1, device=device, dtype=q.dtype)
        running_out = torch.zeros(B, H, Tq, D, device=device, dtype=q.dtype)

        for k_start in range(0, T, block_size):
            k_end = min(k_start + block_size, T)
            k_block = k[:, :, k_start:k_end, :]  # (B, H, Tk, D)
            v_block = v[:, :, k_start:k_end, :]  # (B, H, Tk, D)
            k_end - k_start

            # Scores for this tile: (B, H, Tq, Tk)
            scores = scale * torch.matmul(q_block, k_block.transpose(-2, -1))

            # Extract the sub-mask for this (q_block, k_block) tile: (Tq, Tk)
            tile_mask = mask[q_start:q_end, k_start:k_end]  # (Tq, Tk)
            scores = scores.masked_fill(~tile_mask.unsqueeze(0).unsqueeze(0), -1e9)

            # Online normalisation update
            tile_max = scores.max(dim=-1, keepdim=True).values  # (B, H, Tq, 1)
            new_max = torch.maximum(running_max, tile_max)

            # Rescale old accumulator and add new tile
            exp_tile = torch.exp(scores - new_max)  # (B, H, Tq, Tk)
            rescale = torch.exp(running_max - new_max)  # (B, H, Tq, 1)

            running_out = rescale * running_out + torch.matmul(exp_tile, v_block)
            running_sum = rescale * running_sum + exp_tile.sum(dim=-1, keepdim=True)
            running_max = new_max

        # Normalise
        output[:, :, q_start:q_end, :] = running_out / running_sum.clamp(min=1e-9)

    return output


# ---------------------------------------------------------------------------
# SlidingWindowAttention module
# ---------------------------------------------------------------------------


class SlidingWindowAttention(nn.Module):
    """Multi-head sliding window attention computed with online softmax tiling.

    Unlike :class:`~model.local_global_attention.LocalAttentionLayer` this
    module does **not** use RoPE and operates on a simple boolean mask,
    making it a lightweight drop-in for experimentation.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads (must divide ``d_model``).
        config:  :class:`SlidingWindowConfig` instance.
    """

    def __init__(self, d_model: int, n_heads: int, config: SlidingWindowConfig) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.config = config

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Run sliding-window attention over ``x``.

        Args:
            x:    Input tensor ``(B, T, d_model)``.
            mask: Optional external boolean mask ``(T, T)`` to AND with the
                  sliding-window mask.  Useful for padding.

        Returns:
            Output tensor ``(B, T, d_model)``.
        """
        B, T, _ = x.shape

        # Build the sliding-window boolean mask: (T, T)
        sw_mask = build_sliding_window_mask(
            T, self.config.window_size, causal=self.config.causal
        ).to(x.device)

        # Optionally combine with caller-supplied mask
        if mask is not None:
            sw_mask = sw_mask & mask

        # Project: (B, T, d_model) → (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Tiled attention with sliding mask
        out = tiled_attention(q, k, v, sw_mask, block_size=self.config.block_size)

        # (B, H, T, head_dim) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# GlobalLocalAttention module
# ---------------------------------------------------------------------------


class GlobalLocalAttention(nn.Module):
    """Attention that gives leading tokens global reach and the rest local reach.

    The first ``config.global_tokens`` positions attend to every other position
    (full attention), while the remaining positions use the sliding window.
    This replicates the Longformer / BigBird "global token" idea but computed
    inline with :func:`tiled_attention`.

    Args:
        d_model: Total model dimension.
        n_heads: Number of attention heads.
        config:  :class:`SlidingWindowConfig`; ``global_tokens`` sets the
                 boundary between global and local attention.
    """

    def __init__(self, d_model: int, n_heads: int, config: SlidingWindowConfig) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.config = config

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _build_global_local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build the combined global+local boolean mask.

        Global tokens (rows 0..global_tokens-1) attend to every position.
        Remaining tokens use the sliding-window rule.

        Returns:
            Boolean tensor ``(T, T)``.
        """
        g = min(self.config.global_tokens, seq_len)

        # Start from the sliding-window mask
        mask = build_sliding_window_mask(
            seq_len, self.config.window_size, causal=self.config.causal
        ).to(device)

        # Global token rows: attend everywhere
        mask[:g, :] = True
        # Global token columns: everyone can attend to global tokens
        # (only if non-causal; for causal, only past global tokens matter)
        if not self.config.causal:
            mask[:, :g] = True
        else:
            # For causal, rows >= g can attend to global tokens that are in the past
            i = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
            j = torch.arange(g, device=device).unsqueeze(0)  # (1, g)
            # row i can attend to global token j if j <= i (causal)
            global_col_mask = j <= i  # (T, g)
            mask[:, :g] = mask[:, :g] | global_col_mask

        return mask

    def forward(self, x: Tensor) -> Tensor:
        """Run global-local attention over ``x``.

        Args:
            x: Input tensor ``(B, T, d_model)``.

        Returns:
            Output tensor ``(B, T, d_model)``.
        """
        B, T, _ = x.shape

        mask = self._build_global_local_mask(T, x.device)  # (T, T)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = tiled_attention(q, k, v, mask, block_size=self.config.block_size)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# FLOP estimation
# ---------------------------------------------------------------------------


def compute_attention_flops(
    seq_len: int,
    d_model: int,
    n_heads: int,
    window_size: int,
) -> dict:
    """Estimate multiply-accumulate (MAC) operations for attention variants.

    Both estimates count the QKᵀ and softmax(·)V matrix products.  Projection
    layers are excluded so the comparison focuses on the attention kernel itself.

    Args:
        seq_len:     Sequence length T.
        d_model:     Total model width.
        n_heads:     Number of attention heads.
        window_size: Sliding-window size (tokens each query attends to on average).

    Returns:
        Dictionary with keys:

        ``full_attention_flops``
            FLOPs for standard full attention: ``2 * n_heads * T² * head_dim``.
        ``sliding_window_flops``
            FLOPs for sliding-window attention: ``2 * n_heads * T * min(W, T) * head_dim``.
        ``speedup_ratio``
            ``full / sliding`` (> 1 when the window is smaller than T).
    """
    head_dim = d_model // n_heads
    effective_window = min(window_size, seq_len)

    # Each of the two matmuls (QKᵀ and ·V) costs 2·M·N·K MACs
    # Full attention: (T, head_dim) × (head_dim, T) = T² · head_dim MACs each
    full_attention_flops = 2 * n_heads * seq_len * seq_len * head_dim

    # Sliding window: each query row attends to at most effective_window keys
    sliding_window_flops = 2 * n_heads * seq_len * effective_window * head_dim

    speedup_ratio = full_attention_flops / sliding_window_flops if sliding_window_flops > 0 else 1.0

    return {
        "full_attention_flops": full_attention_flops,
        "sliding_window_flops": sliding_window_flops,
        "speedup_ratio": speedup_ratio,
    }
