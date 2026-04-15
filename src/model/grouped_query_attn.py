"""Grouped Query Attention (GQA) and Multi-Query Attention (MQA).

GQA reduces KV cache size by sharing K/V heads across multiple Q heads.
MQA is the extreme case with a single K/V head shared by all Q heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GQAConfig:
    """Configuration for Grouped Query Attention.

    Attributes:
        d_model: Model (embedding) dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (must divide n_heads evenly).
        d_head: Per-head dimension. Defaults to d_model // n_heads.
        causal: Whether to apply a causal (autoregressive) mask.
        dropout_p: Attention dropout probability (applied during training).
    """

    d_model: int = 64
    n_heads: int = 8
    n_kv_heads: int = 2
    d_head: Optional[int] = None
    causal: bool = True
    dropout_p: float = 0.0

    def __post_init__(self) -> None:
        if self.d_head is None:
            self.d_head = self.d_model // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})."
            )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat each KV head n_rep times to match the number of Q heads.

    Args:
        x: Tensor of shape (B, T, n_kv_heads, d_head).
        n_rep: Repetition factor — n_heads // n_kv_heads.

    Returns:
        Tensor of shape (B, T, n_kv_heads * n_rep, d_head), achieved via
        expand + reshape so no data is copied.
    """
    if n_rep == 1:
        return x
    B, T, n_kv_heads, d_head = x.shape
    # (B, T, n_kv_heads, 1, d_head) → expand → (B, T, n_kv_heads, n_rep, d_head)
    x = x[:, :, :, None, :].expand(B, T, n_kv_heads, n_rep, d_head)
    return x.reshape(B, T, n_kv_heads * n_rep, d_head)


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention.

    Projects queries to (n_heads * d_head) and keys/values to
    (n_kv_heads * d_head), then expands K/V via repeat_kv before computing
    scaled dot-product attention.

    Args:
        config: GQAConfig instance.
    """

    def __init__(self, config: GQAConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        d_head = config.d_head
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.scale = math.sqrt(d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute grouped query attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        cfg = self.config
        B, T, _ = x.shape

        # Project
        q = self.q_proj(x)  # (B, T, n_heads * d_head)
        k = self.k_proj(x)  # (B, T, n_kv_heads * d_head)
        v = self.v_proj(x)  # (B, T, n_kv_heads * d_head)

        # Reshape to (B, T, heads, d_head)
        q = q.view(B, T, cfg.n_heads, cfg.d_head)
        k = k.view(B, T, cfg.n_kv_heads, cfg.d_head)
        v = v.view(B, T, cfg.n_kv_heads, cfg.d_head)

        # Expand K/V heads to match Q heads
        k = repeat_kv(k, self.n_rep)  # (B, T, n_heads, d_head)
        v = repeat_kv(v, self.n_rep)

        # Transpose to (B, n_heads, T, d_head) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        # Build causal mask if needed
        attn_mask = None
        if cfg.causal:
            attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()

        dropout_p = cfg.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # mask supplied explicitly
        )

        # (B, n_heads, T, d_head) → (B, T, n_heads * d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, cfg.n_heads * cfg.d_head)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention — GQA with n_kv_heads=1.

    A single K/V head is shared by all Q heads. This maximises KV cache
    savings at the cost of some model quality.

    Args:
        d_model: Model dimension.
        n_heads: Number of query heads.
        d_head: Per-head dimension (default: d_model // n_heads).
        causal: Whether to apply a causal mask.
        dropout_p: Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        d_head: Optional[int] = None,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        cfg = GQAConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,
            d_head=d_head,
            causal=causal,
            dropout_p=dropout_p,
        )
        self.attn = GroupedQueryAttention(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class GQABlock(nn.Module):
    """Transformer block with pre-LayerNorm + GroupedQueryAttention + residual.

    Args:
        config: GQAConfig instance.
    """

    def __init__(self, config: GQAConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention with residual connection.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return x + self.attn(self.norm(x))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_kv_cache_params(config: GQAConfig) -> Dict[str, int]:
    """Return KV cache parameter statistics for a GQA configuration.

    Returns a dictionary with:
        - ``n_kv_params``: Total number of K+V parameters per position
          (n_kv_heads * d_head * 2).
        - ``n_q_params``: Total number of Q parameters per position
          (n_heads * d_head).
        - ``kv_reduction_ratio_vs_mha``: How many times smaller the KV cache
          is compared to full MHA (n_heads // n_kv_heads).
    """
    d_head = config.d_head  # already resolved in __post_init__
    return {
        "n_kv_params": config.n_kv_heads * d_head * 2,
        "n_q_params": config.n_heads * d_head,
        "kv_reduction_ratio_vs_mha": config.n_heads // config.n_kv_heads,
    }
