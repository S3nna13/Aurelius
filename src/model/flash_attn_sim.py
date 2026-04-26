"""Flash attention simulation: tiled/chunked attention computation for memory efficiency (without flash-attn library).
Implements the tiling algorithm of FlashAttention (Dao et al. 2022), computing
attention in tiles over both Q and KV dimensions to avoid materializing the
full (T, T) attention matrix. Uses the online softmax trick for numerical stability.

This is DIFFERENT from memory_efficient_attn.py which only chunks over KV.
Here we tile over both Q blocks and KV blocks simultaneously.
"""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlashAttnConfig:
    """Configuration for tiled flash attention computation.

    Attributes:
        block_size: Tile size for Q blocks.
        kv_block_size: Tile size for KV blocks.
        causal: If True, apply causal masking (no attending to future positions).
        dropout_p: Attention dropout probability (applied during training).
        scale: Attention scale factor. If None, uses 1/sqrt(head_dim).
    """

    block_size: int = 32
    kv_block_size: int = 32
    causal: bool = True
    dropout_p: float = 0.0
    scale: float | None = None


def online_softmax_update(
    prev_max: Tensor,  # (B, H, T_q, 1)
    prev_sum: Tensor,  # (B, H, T_q, 1)
    prev_out: Tensor,  # (B, H, T_q, D)
    new_scores: Tensor,  # (B, H, T_q, T_kv)
    new_values: Tensor,  # (B, H, T_kv, D)
) -> tuple[Tensor, Tensor, Tensor]:
    """Single step of the online softmax trick for incremental KV block processing.

    Updates the running (max, sum, output) accumulators with a new KV tile without
    needing to store the full attention matrix. Maintains numerical stability via
    the log-sum-exp trick.

    Args:
        prev_max: Running maximum of attention scores, shape (B, H, T_q, 1).
        prev_sum: Running sum of exp(scores - max), shape (B, H, T_q, 1).
        prev_out: Running weighted value output, shape (B, H, T_q, D).
        new_scores: Raw attention scores for current KV tile, shape (B, H, T_q, T_kv).
        new_values: Value tensor for current KV tile, shape (B, H, T_kv, D).

    Returns:
        Tuple of (new_max, new_sum, new_out) — updated running stats and output.
        new_max and new_sum have shape (B, H, T_q, 1); new_out has shape (B, H, T_q, D).
    """
    # chunk_max: (B, H, T_q, 1)
    chunk_max = new_scores.max(dim=-1, keepdim=True).values

    # new_max: element-wise max, (B, H, T_q, 1)
    new_max = torch.maximum(prev_max, chunk_max)

    # exp_new: (B, H, T_q, T_kv) — stabilized by new_max
    exp_new = torch.exp(new_scores - new_max)

    # chunk_sum: (B, H, T_q, 1)
    chunk_sum = exp_new.sum(dim=-1, keepdim=True)

    # scale_prev: correction factor for previous accumulator, (B, H, T_q, 1)
    scale_prev = torch.exp(prev_max - new_max)

    # new_sum: (B, H, T_q, 1)
    new_sum = scale_prev * prev_sum + chunk_sum

    # new_out: (B, H, T_q, D)
    # exp_new @ new_values: (B, H, T_q, T_kv) x (B, H, T_kv, D) -> (B, H, T_q, D)
    new_out = scale_prev * prev_out + torch.matmul(exp_new, new_values)

    return new_max, new_sum, new_out


def flash_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    config: FlashAttnConfig,
) -> Tensor:
    """Tiled flash-style attention forward pass.

    Processes attention in tiles of (Q_block x KV_block) without materializing the
    full (T, T) attention matrix. Uses the online softmax trick to accumulate
    partial results across KV tiles for each Q tile.

    Args:
        query: Query tensor, shape (B, H, T, D).
        key: Key tensor, shape (B, H, T, D).
        value: Value tensor, shape (B, H, T, D).
        config: FlashAttnConfig with tiling and masking parameters.

    Returns:
        Output tensor of shape (B, H, T, D).
    """
    B, H, T, D = query.shape
    device = query.device

    scale = config.scale if config.scale is not None else 1.0 / math.sqrt(D)

    # Use float32 accumulation for numerical stability
    q_f = query.float()
    k_f = key.float()
    v_f = value.float()

    # Output accumulator — final result (B, H, T, D)
    output = torch.zeros(B, H, T, D, dtype=torch.float32, device=device)

    q_block_size = config.block_size
    kv_block_size = config.kv_block_size

    # Process each Q tile independently
    for q_start in range(0, T, q_block_size):
        q_end = min(q_start + q_block_size, T)
        T_q_block = q_end - q_start

        q_block = q_f[:, :, q_start:q_end, :]  # (B, H, T_q_block, D)

        # Initialize running accumulators for this Q tile
        running_max = torch.full(
            (B, H, T_q_block, 1), float("-inf"), dtype=torch.float32, device=device
        )
        running_sum = torch.zeros((B, H, T_q_block, 1), dtype=torch.float32, device=device)
        running_out = torch.zeros((B, H, T_q_block, D), dtype=torch.float32, device=device)

        # Q position indices for causal masking: absolute positions in sequence
        q_positions = torch.arange(q_start, q_end, device=device)  # (T_q_block,)

        # Iterate over KV tiles
        for kv_start in range(0, T, kv_block_size):
            kv_end = min(kv_start + kv_block_size, T)

            # Causal masking: skip KV tiles that are entirely in the future
            if config.causal and kv_start > q_end - 1:
                # All keys in this tile are strictly after all queries in Q tile
                break

            k_block = k_f[:, :, kv_start:kv_end, :]  # (B, H, T_kv_block, D)
            v_block = v_f[:, :, kv_start:kv_end, :]  # (B, H, T_kv_block, D)

            # scores: (B, H, T_q_block, T_kv_block)
            scores = scale * torch.matmul(q_block, k_block.transpose(-2, -1))

            if config.causal:
                # q_pos: (T_q_block, 1), kv_pos: (1, T_kv_block)
                q_pos = q_positions.view(-1, 1)
                kv_pos = torch.arange(kv_start, kv_end, device=device).view(1, -1)
                # Mask future positions: mask[i, j] = True where kv_j > q_i
                causal_mask = kv_pos > q_pos  # (T_q_block, T_kv_block)
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            running_max, running_sum, running_out = online_softmax_update(
                running_max, running_sum, running_out, scores, v_block
            )

        # Normalize and write back: avoid division by zero for all-masked rows
        safe_sum = running_sum.clamp(min=1e-12)
        output[:, :, q_start:q_end, :] = running_out / safe_sum

    return output.to(query.dtype)


class TiledAttention(nn.Module):
    """Standard multi-head attention using tiled flash-style computation.

    Projects input into Q, K, V, applies tiled flash attention across heads,
    then projects back to d_model. Drop-in replacement for standard MHA that
    avoids O(T^2) memory by computing attention tile-by-tile.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads. Must divide d_model evenly.
        config: FlashAttnConfig controlling tiling parameters.
    """

    def __init__(self, d_model: int, n_heads: int, config: FlashAttnConfig) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.config = config

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, T, d_model).

        Returns:
            Output tensor, shape (B, T, d_model).
        """
        B, T, _ = x.shape

        # Project and reshape to (B, H, T, D)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Tiled flash attention: (B, H, T, D)
        out = flash_attention_forward(q, k, v, self.config)

        # Reshape back: (B, H, T, D) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


def compare_with_standard_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute both standard (full matrix) and tiled flash attention for comparison.

    Standard attention materializes the full (T, T) attention matrix using
    F.scaled_dot_product_attention; tiled attention uses flash_attention_forward.
    Both should produce numerically close results (atol=1e-4).

    Args:
        query: Query tensor, shape (B, H, T, D).
        key: Key tensor, shape (B, H, T, D).
        value: Value tensor, shape (B, H, T, D).
        causal: If True, apply causal masking to both implementations.

    Returns:
        Tuple of (standard_output, tiled_output), each shape (B, H, T, D).
    """
    # Standard attention using PyTorch's built-in SDPA
    standard_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )

    # Tiled flash attention
    config = FlashAttnConfig(
        block_size=32,
        kv_block_size=32,
        causal=causal,
        dropout_p=0.0,
        scale=None,
    )
    tiled_output = flash_attention_forward(query, key, value, config)

    return standard_output, tiled_output


def compute_memory_usage(
    seq_len: int,
    n_heads: int,
    head_dim: int,
    block_size: int,
) -> dict[str, int]:
    """Estimate memory in bytes for tiled vs standard attention.

    Standard attention requires an O(T^2) attention matrix per head.
    Tiled attention only materializes one tile at a time: O(T * block_size).

    Args:
        seq_len: Sequence length T.
        n_heads: Number of attention heads H.
        head_dim: Dimension per attention head D.
        block_size: Tile/block size used in tiled attention.

    Returns:
        Dictionary with keys:
            'standard_bytes': Bytes for full (H, T, T) attention matrix (float32).
            'tiled_bytes': Bytes for single (H, T_q_block, T_kv_block) tile (float32).
            'reduction_factor': standard_bytes / tiled_bytes (float).
    """
    bytes_per_element = 4  # float32

    # Standard: full attention matrix (n_heads, T, T)
    standard_bytes = n_heads * seq_len * seq_len * bytes_per_element

    # Tiled: only one (Q_block x KV_block) tile per head at a time
    # Q tile: block_size rows, KV tile: block_size cols
    tiled_bytes = n_heads * block_size * block_size * bytes_per_element

    reduction_factor = standard_bytes / tiled_bytes if tiled_bytes > 0 else 1.0

    return {
        "standard_bytes": standard_bytes,
        "tiled_bytes": tiled_bytes,
        "reduction_factor": reduction_factor,
    }
