"""Multi-query, grouped-query, and cross-attention variants.

Standalone attention variants with benchmarking utilities.
All variants are self-contained and do not depend on AureliusTransformer internals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AttentionVariantConfig:
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 1          # 1 = MQA, n_heads = MHA, else = GQA
    head_dim: int = 32
    dropout: float = 0.0
    use_rope: bool = False        # simplified sinusoidal position bias instead


# ---------------------------------------------------------------------------
# Core attention function
# ---------------------------------------------------------------------------

def mha_attention(
    q: Tensor,               # (B, H, T, head_dim)
    k: Tensor,               # (B, H_kv, T, head_dim)
    v: Tensor,               # (B, H_kv, T, head_dim)
    mask: Tensor | None = None,  # (T, T) causal mask
    dropout: float = 0.0,
) -> Tensor:
    """Scaled dot-product attention with optional KV head expansion (for GQA/MQA).

    Expands K/V heads to match Q heads if H_kv < H.
    Returns (B, H, T, head_dim).
    """
    B, H, T, head_dim = q.shape
    H_kv = k.shape[1]

    # Expand K/V heads to match Q heads (GQA/MQA -> MHA equivalent)
    if H_kv < H:
        groups = H // H_kv
        k = k.repeat_interleave(groups, dim=1)  # (B, H, T, head_dim)
        v = v.repeat_interleave(groups, dim=1)  # (B, H, T, head_dim)

    scale = math.sqrt(head_dim)
    # (B, H, T, T)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if mask is not None:
        # mask shape (T, T): 1 = keep, 0 = mask out
        # Broadcast to (B, H, T, T)
        mask = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout)

    output = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
    return output


# ---------------------------------------------------------------------------
# Multi-Head Attention (MHA)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention (n_kv_heads == n_heads)."""

    def __init__(self, config: AttentionVariantConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x (B, T, d_model) -> (B, T, d_model)"""
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = mha_attention(q, k, v, mask=mask, dropout=self.dropout if self.training else 0.0)
        # (B, H, T, hd) -> (B, T, H*hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Multi-Query Attention (MQA)
# ---------------------------------------------------------------------------

class MultiQueryAttention(nn.Module):
    """Multi-query attention: n_kv_heads == 1 (shared K/V across all Q heads)."""

    def __init__(self, config: AttentionVariantConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        # Single K and V head
        self.k_proj = nn.Linear(config.d_model, config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)              # (B, 1, T, hd)
        v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)              # (B, 1, T, hd)

        out = mha_attention(q, k, v, mask=mask, dropout=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Grouped-query attention: 1 < n_kv_heads < n_heads."""

    def __init__(self, config: AttentionVariantConfig) -> None:
        super().__init__()
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({config.n_kv_heads})"
            )
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # (B, H, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, H_kv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        out = mha_attention(q, k, v, mask=mask, dropout=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Cross-attention: Q from decoder, K/V from encoder."""

    def __init__(self, config: AttentionVariantConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, query: Tensor, context: Tensor, mask: Tensor | None = None) -> Tensor:
        """query (B, T_q, d_model), context (B, T_kv, d_model) -> (B, T_q, d_model)"""
        B, T_q, _ = query.shape
        T_kv = context.shape[1]

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)       # (B, H, T_q, hd)
        k = self.k_proj(context).view(B, T_kv, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, H_kv, T_kv, hd)
        v = self.v_proj(context).view(B, T_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # For cross-attention, expand kv heads if needed
        H = self.n_heads
        H_kv = self.n_kv_heads
        if H_kv < H:
            groups = H // H_kv
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T_q, T_kv)

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, H, T_q, hd)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Benchmarking / analysis utilities
# ---------------------------------------------------------------------------

def count_kv_parameters(n_heads: int, n_kv_heads: int, head_dim: int, d_model: int) -> int:
    """Count K and V projection parameters.

    Returns 2 * n_kv_heads * head_dim * d_model.
    """
    return 2 * n_kv_heads * head_dim * d_model


def attention_memory_ratio(n_heads: int, n_kv_heads: int) -> float:
    """KV cache memory ratio relative to MHA: n_kv_heads / n_heads.

    MQA (n_kv_heads=1) -> 1/n_heads
    MHA (n_kv_heads=n_heads) -> 1.0
    Returns float in (0, 1].
    """
    return n_kv_heads / n_heads
