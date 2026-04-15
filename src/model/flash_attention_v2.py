"""Flash Attention V2: CPU-friendly tiled attention simulation with online softmax.

This module demonstrates the algorithmic approach of Flash Attention V2 using
pure PyTorch (no custom CUDA kernels). It iterates over Q in block_q chunks and
K/V in block_k chunks, accumulating with numerically stable online softmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlashAttnV2Config:
    """Configuration for Flash Attention V2 simulation.

    Attributes:
        d_model: Total model dimension.
        n_heads: Number of attention heads.
        block_size_q: Block size for Q tiles.
        block_size_k: Block size for K/V tiles.
        causal: If True, apply causal masking (no future token attention).
        scale: Optional scale factor; defaults to 1/sqrt(d_head).
    """

    d_model: int = 64
    n_heads: int = 4
    block_size_q: int = 16
    block_size_k: int = 16
    causal: bool = True
    scale: Optional[float] = None


def tiled_attention_v2(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    block_q: int,
    block_k: int,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tensor:
    """Tiled attention with online (numerically stable) softmax accumulation.

    Iterates over Q in block_q-sized chunks; for each Q chunk, iterates over
    K/V in block_k-sized chunks and accumulates the output using the online
    softmax correction technique from Flash Attention V2.

    Args:
        Q: Query tensor of shape (B, H, T, d_head).
        K: Key tensor of shape (B, H, T, d_head).
        V: Value tensor of shape (B, H, T, d_head).
        block_q: Block (tile) size along the query sequence dimension.
        block_k: Block (tile) size along the key/value sequence dimension.
        causal: If True, future K positions are masked to -inf.
        scale: Scaling factor applied to QK^T. Defaults to 1/sqrt(d_head).

    Returns:
        Output tensor of shape (B, H, T, d_head).
    """
    B, H, T, d_head = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    # Output accumulator and online softmax state
    out = torch.zeros_like(Q)                            # (B, H, T, d_head)
    # Running max per query position (used for numerical stability)
    running_max = torch.full((B, H, T, 1), float("-inf"), dtype=Q.dtype, device=Q.device)
    # Running sum of exp weights
    running_sum = torch.zeros(B, H, T, 1, dtype=Q.dtype, device=Q.device)

    num_q_blocks = math.ceil(T / block_q)
    num_k_blocks = math.ceil(T / block_k)

    for qi in range(num_q_blocks):
        q_start = qi * block_q
        q_end = min(q_start + block_q, T)
        Qi = Q[:, :, q_start:q_end, :]           # (B, H, bq, d_head)
        bq_actual = q_end - q_start

        # Per-block online softmax state
        m_i = torch.full((B, H, bq_actual, 1), float("-inf"), dtype=Q.dtype, device=Q.device)
        l_i = torch.zeros(B, H, bq_actual, 1, dtype=Q.dtype, device=Q.device)
        o_i = torch.zeros(B, H, bq_actual, d_head, dtype=Q.dtype, device=Q.device)

        for ki in range(num_k_blocks):
            k_start = ki * block_k
            k_end = min(k_start + block_k, T)
            Kj = K[:, :, k_start:k_end, :]       # (B, H, bk, d_head)
            Vj = V[:, :, k_start:k_end, :]       # (B, H, bk, d_head)

            # Compute scaled dot-product scores: (B, H, bq, bk)
            scores = torch.matmul(Qi, Kj.transpose(-1, -2)) * scale

            # Apply causal mask: for position q, mask positions k > q
            if causal:
                # Absolute positions for queries and keys in this block
                q_positions = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)  # (bq, 1)
                k_positions = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)  # (1, bk)
                mask = k_positions > q_positions  # (bq, bk), True where future
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Online softmax update
            # New block max
            m_new = torch.maximum(m_i, scores.max(dim=-1, keepdim=True).values)

            # Corrected running sum and output
            # Correction factor for previous accumulation
            alpha = torch.exp(m_i - m_new)          # (B, H, bq, 1)
            # Weights for new block
            exp_scores = torch.exp(scores - m_new)  # (B, H, bq, bk)
            l_new = alpha * l_i + exp_scores.sum(dim=-1, keepdim=True)

            # Update output accumulator
            o_i = alpha * o_i + torch.matmul(exp_scores, Vj)

            m_i = m_new
            l_i = l_new

        # Normalize
        o_i = o_i / (l_i + 1e-9)
        out[:, :, q_start:q_end, :] = o_i

    return out


def standard_attention_v2(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tensor:
    """Reference standard scaled dot-product attention.

    Args:
        Q: Query tensor of shape (B, H, T, d_head).
        K: Key tensor of shape (B, H, T, d_head).
        V: Value tensor of shape (B, H, T, d_head).
        causal: If True, apply causal mask.
        scale: Scaling factor. Defaults to 1/sqrt(d_head).

    Returns:
        Output tensor of shape (B, H, T, d_head).
    """
    B, H, T, d_head = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale   # (B, H, T, T)

    if causal:
        # Upper-triangular mask (future positions)
        mask = torch.triu(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)             # (B, H, T, T)
    return torch.matmul(attn_weights, V)                     # (B, H, T, d_head)


class FlashAttentionV2(nn.Module):
    """Flash Attention V2 module.

    Projects input to Q, K, V via linear layers, applies tiled attention,
    and projects back to d_model.

    Args:
        config: FlashAttnV2Config instance.
    """

    def __init__(self, config: FlashAttnV2Config) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.config = config
        self.d_head = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Flash Attention V2.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, d_model = x.shape
        H = self.config.n_heads
        d_head = self.d_head

        # Project and reshape to (B, H, T, d_head)
        Q = self.q_proj(x).view(B, T, H, d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d_head).transpose(1, 2)

        # Apply tiled attention
        out = tiled_attention_v2(
            Q, K, V,
            block_q=self.config.block_size_q,
            block_k=self.config.block_size_k,
            causal=self.config.causal,
            scale=self.config.scale,
        )

        # Merge heads and project: (B, H, T, d_head) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        return self.out_proj(out)


class FlashAttnV2Block(nn.Module):
    """Transformer block with pre-norm and Flash Attention V2 with residual.

    Applies: x + FlashAttentionV2(LayerNorm(x))

    Args:
        config: FlashAttnV2Config instance.
    """

    def __init__(self, config: FlashAttnV2Config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = FlashAttentionV2(config)

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-norm attention with residual connection.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return x + self.attn(self.norm(x))
