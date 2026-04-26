"""Parallel-block transformer layer (PaLM / GPT-J / Falcon style).

Attention and FFN run IN PARALLEL from the same normalized input rather than
sequentially, which improves training throughput by permitting parallel
execution of the two branches and by fusing the QKV / FFN projections.

    out = x + attn(LN(x)) + ffn(LN(x))

This module is self-contained (GQA-aware MHA + SwiGLU FFN + LayerNorm) and has
no coupling to the frozen sequential transformer implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ParallelAttentionBlock(nn.Module):
    """Parallel-block transformer layer with GQA-aware attention and SwiGLU FFN.

    Args:
        d_model:    Model hidden dimension (must equal ``n_heads * head_dim``).
        n_heads:    Number of query heads.
        head_dim:   Per-head dimension.
        n_kv_heads: Number of key/value heads (GQA). ``n_heads`` must be a
                    multiple of ``n_kv_heads``.
        d_ff:       SwiGLU inner dimension.
        dropout:    Dropout probability applied to attention weights and the
                    residual branch outputs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # ---- Config validation -------------------------------------------------
        if d_model <= 0 or n_heads <= 0 or head_dim <= 0 or n_kv_heads <= 0 or d_ff <= 0:
            raise ValueError(
                f"All dimensions must be positive; got d_model={d_model}, "
                f"n_heads={n_heads}, head_dim={head_dim}, n_kv_heads={n_kv_heads}, "
                f"d_ff={d_ff}"
            )
        if n_heads * head_dim != d_model:
            raise ValueError(
                f"n_heads * head_dim ({n_heads} * {head_dim} = {n_heads * head_dim}) "
                f"must equal d_model ({d_model})"
            )
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0); got {dropout}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.n_groups = n_heads // n_kv_heads
        self._attn_scale = 1.0 / math.sqrt(head_dim)

        # Shared pre-norm. Both branches consume the same normalized activation.
        self.norm = nn.LayerNorm(d_model)

        # Attention projections (GQA: Q gets n_heads, K/V get n_kv_heads).
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # SwiGLU FFN: gate * SiLU(up) -> down
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ---- Branches ----------------------------------------------------------
    def _attention(self, x: Tensor) -> Tensor:
        """GQA-aware causal self-attention on pre-normalized input."""
        B, S, _ = x.shape
        h = self.n_heads
        kh = self.n_kv_heads
        hd = self.head_dim

        q = self.q_proj(x).view(B, S, h, hd).transpose(1, 2)  # (B, h,  S, hd)
        k = self.k_proj(x).view(B, S, kh, hd).transpose(1, 2)  # (B, kh, S, hd)
        v = self.v_proj(x).view(B, S, kh, hd).transpose(1, 2)  # (B, kh, S, hd)

        # Broadcast KV heads to Q heads for GQA via repeat_interleave.
        if self.n_groups != 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Causal self-attention.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self._attn_scale  # (B, h, S, S)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        if self.attn_dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout_p)

        out = torch.matmul(attn, v)  # (B, h, S, hd)
        out = out.transpose(1, 2).contiguous().view(B, S, h * hd)
        return self.o_proj(out)

    def _ffn(self, x: Tensor) -> Tensor:
        """SwiGLU FFN."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    # ---- Forward -----------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Parallel block: ``out = x + attn(LN(x)) + ffn(LN(x))``.

        Args:
            x: Tensor of shape ``(B, S, D)``.

        Returns:
            Tensor of shape ``(B, S, D)`` with the same dtype as ``x``.
        """
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input shape (B, S, {self.d_model}); got {tuple(x.shape)}")

        normed = self.norm(x)
        attn_out = self._attention(normed)
        ffn_out = self._ffn(normed)
        return x + self.resid_dropout(attn_out) + self.resid_dropout(ffn_out)
