"""
Ring Attention (Liu et al. 2023) — Oracle / Simulated variant.

Reference: "Ring Attention with Blockwise Transformers for Near-Infinite Context"
           arXiv:2310.01889

This module simulates distributed ring attention on a single device.
The sequence is partitioned into chunks (one per simulated "device").  In
the real ring algorithm each device holds one Q-chunk and passes its K/V
around a ring of N devices, accumulating partial attention outputs until it
has seen all K/V.  Here we shortcut the communication and let every Q-chunk
attend directly to the full K and V tensors — the "oracle" result that the
ring converges to after N rounds.

Key advantage over full attention: memory footprint per simulated device is
O(chunk_size × T) rather than O(T²), so very long sequences can be processed
in chunks without materialising a full T×T score matrix.

Classes
-------
RingAttentionConfig  — dataclass of hyperparameters
ChunkAttention       — scaled dot-product for one (Q-chunk, full-K, full-V) triple
RingAttentionSimulated — full module; projects, chunks Q, calls ChunkAttention
RingAttentionLayer   — pre-norm wrapper: LayerNorm + RingAttentionSimulated + FFN
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RingAttentionConfig:
    """Hyperparameters for RingAttentionSimulated."""

    d_model: int
    n_heads: int
    head_dim: int
    chunk_size: int = 64   # tokens per simulated device / ring slot
    causal: bool = True


# ---------------------------------------------------------------------------
# ChunkAttention
# ---------------------------------------------------------------------------

class ChunkAttention(nn.Module):
    """Scaled dot-product attention for a single Q-chunk against full K/V.

    Args:
        n_heads:  Number of attention heads.
        head_dim: Dimension per head.
    """

    def __init__(self, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

    def forward(
        self,
        q: Tensor,                # (B, nh, Tq, dh)
        k: Tensor,                # (B, nh, Tk, dh)
        v: Tensor,                # (B, nh, Tk, dh)
        causal: bool = False,
        q_chunk_start: int = 0,   # absolute start position of this Q-chunk
    ) -> Tensor:
        """Compute attention for a Q-chunk attending to full K/V.

        Args:
            q:              Query chunk, shape (B, nh, Tq, dh).
            k:              Full key tensor, shape (B, nh, Tk, dh).
            v:              Full value tensor, shape (B, nh, Tk, dh).
            causal:         If True, apply causal masking.
            q_chunk_start:  Absolute sequence index of q[:, :, 0, :].
                            Used to build the correct causal mask.

        Returns:
            Attended output of shape (B, nh, Tq, dh).
        """
        B, nh, Tq, dh = q.shape
        Tk = k.shape[2]

        # Raw scores: (B, nh, Tq, Tk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if causal:
            # q_positions: (Tq,)  absolute position of each query token
            # k_positions: (Tk,)  absolute position of each key   token
            q_pos = torch.arange(Tq, device=q.device) + q_chunk_start  # (Tq,)
            k_pos = torch.arange(Tk, device=q.device)                   # (Tk,)
            # mask[qi, ki] = True  where k_pos[ki] > q_pos[qi]  → future → -inf
            mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)              # (Tq, Tk)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # Replace NaN from all-masked rows (softmax of all-inf) with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# RingAttentionSimulated
# ---------------------------------------------------------------------------

class RingAttentionSimulated(nn.Module):
    """Oracle ring attention (single-device simulation).

    Splits the projected Q into chunks of ``config.chunk_size`` tokens and
    computes, for each chunk, scaled dot-product attention against the *full*
    K and V tensors.  This is mathematically equivalent to what the ring
    algorithm produces after all N rounds of KV communication have completed.

    Memory per chunk: O(chunk_size × T) — the dominant attention score matrix.
    Compare with full attention: O(T²).

    Args:
        config: :class:`RingAttentionConfig` instance.
    """

    def __init__(self, config: RingAttentionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        inner = config.n_heads * config.head_dim

        self.W_q = nn.Linear(d, inner, bias=False)
        self.W_k = nn.Linear(d, inner, bias=False)
        self.W_v = nn.Linear(d, inner, bias=False)
        self.W_o = nn.Linear(inner, d, bias=False)

        self.chunk_attn = ChunkAttention(config.n_heads, config.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, nh*dh) → (B, nh, T, dh)."""
        B, T, _ = x.shape
        x = x.view(B, T, self.config.n_heads, self.config.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, nh, T, dh) → (B, T, nh*dh)."""
        B, nh, T, dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, nh * dh)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Compute ring attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        cfg = self.config

        # Project and reshape: (B, nh, T, dh)
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        chunk_size = cfg.chunk_size
        output_chunks: list[Tensor] = []

        # Process Q in chunks; K and V are always the full sequence
        start = 0
        while start < T:
            end = min(start + chunk_size, T)
            q_chunk = Q[:, :, start:end, :]          # (B, nh, Cq, dh)

            out_chunk = self.chunk_attn(
                q=q_chunk,
                k=K,
                v=V,
                causal=cfg.causal,
                q_chunk_start=start,
            )                                         # (B, nh, Cq, dh)
            output_chunks.append(out_chunk)
            start = end

        # Reassemble: (B, nh, T, dh) → (B, T, nh*dh)
        O = torch.cat(output_chunks, dim=2)           # (B, nh, T, dh)
        out = self._merge_heads(O)                    # (B, T, nh*dh)
        return self.W_o(out)                          # (B, T, d_model)


# ---------------------------------------------------------------------------
# RingAttentionLayer  (pre-norm wrapper)
# ---------------------------------------------------------------------------

class RingAttentionLayer(nn.Module):
    """Pre-norm transformer layer using RingAttentionSimulated.

    Architecture::

        x → LayerNorm → RingAttentionSimulated → + x
          → LayerNorm → FFN(d_model, d_ff, d_model) → + x

    Args:
        config: :class:`RingAttentionConfig` instance.
        d_ff:   Hidden dimension of the position-wise FFN.
    """

    def __init__(self, config: RingAttentionConfig, d_ff: int) -> None:
        super().__init__()
        d = config.d_model
        self.norm1 = nn.LayerNorm(d)
        self.attn = RingAttentionSimulated(config)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm forward pass.

        Args:
            x: Input of shape (B, T, d_model).

        Returns:
            Output of shape (B, T, d_model).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
