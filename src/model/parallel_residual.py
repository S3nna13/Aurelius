"""Parallel residual block: attention and FFN run in parallel, outputs summed."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _make_norm(d_model: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    return nn.LayerNorm(d_model)


@dataclass
class ParallelConfig:
    d_model: int
    n_heads: int
    head_dim: int
    d_ff: int
    dropout: float = 0.0


class ParallelBlock(nn.Module):
    """Pre-norm block where attention and FFN run in parallel.

    Given normed input x_n:
        out = x + attn(x_n) + ffn(x_n)
    """

    def __init__(self, config: ParallelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        h = config.n_heads
        hd = config.head_dim

        self.norm = _make_norm(d)

        # Attention projections
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, h * hd, bias=False)
        self.v_proj = nn.Linear(d, h * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)

        # FFN: W1 (gate), W2 (down) with SiLU activation
        self.W1 = nn.Linear(d, config.d_ff, bias=False)
        self.W2 = nn.Linear(config.d_ff, d, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        self._scale = 1.0 / math.sqrt(hd)

    def _self_attention(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        h = self.config.n_heads
        hd = self.config.head_dim

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)  # (B, h, T, hd)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        # Causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        attn = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, h, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, h * hd)
        return self.o_proj(out)

    def _ffn(self, x: Tensor) -> Tensor:
        return self.W2(F.silu(self.W1(x)))

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm(x)
        attn_out = self._self_attention(normed)
        ffn_out = self._ffn(normed)
        return x + attn_out + ffn_out
