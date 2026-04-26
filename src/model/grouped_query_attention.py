"""Grouped Query Attention (GQA) — Ainslie et al. 2023.

GQA uses fewer KV heads than Q heads. Each KV head is shared by
n_q_per_kv = n_q_heads // n_kv_heads query heads, reducing KV-cache
memory and compute while retaining most of the quality of full MHA.

Reference: https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GQAConfig:
    """Configuration for Grouped Query Attention.

    Attributes:
        d_model:    Model (embedding) dimension.
        n_q_heads:  Number of query heads.
        n_kv_heads: Number of key/value heads.  Must divide n_q_heads evenly.
        head_dim:   Per-head dimension.
        dropout:    Attention dropout probability (applied during training).
        causal:     Whether to apply a causal (lower-triangular) mask.
    """

    d_model: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    dropout: float = 0.0
    causal: bool = True

    def __post_init__(self) -> None:
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({self.n_q_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})."
            )

    @property
    def n_q_per_kv(self) -> int:
        """Number of query heads that share each KV head."""
        return self.n_q_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# Core GQA module
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (Ainslie et al. 2023).

    Args:
        config: GQAConfig instance.
    """

    def __init__(self, config: GQAConfig) -> None:
        super().__init__()
        self.config = config

        q_dim = config.n_q_heads * config.head_dim
        kv_dim = config.n_kv_heads * config.head_dim

        self.W_q = nn.Linear(config.d_model, q_dim, bias=False)
        self.W_k = nn.Linear(config.d_model, kv_dim, bias=False)
        self.W_v = nn.Linear(config.d_model, kv_dim, bias=False)
        self.W_o = nn.Linear(q_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self._scale = math.sqrt(config.head_dim)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: Tensor, n_heads: int) -> Tensor:
        """(B, T, n_heads*head_dim) → (B, n_heads, T, head_dim)."""
        B, T, _ = x.shape
        return x.view(B, T, n_heads, self.config.head_dim).transpose(1, 2)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute grouped-query attention.

        Args:
            x:    Input tensor of shape (B, T, d_model).
            mask: Optional additive attention mask of shape (B, T, T) or
                  (1, T, T). Values of ``-inf`` block the corresponding
                  positions.  When *None* and ``config.causal=True`` a
                  causal lower-triangular mask is built automatically.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        cfg = self.config
        B, T, _ = x.shape

        # Project
        q = self._split_heads(self.W_q(x), cfg.n_q_heads)  # (B, Hq, T, D)
        k = self._split_heads(self.W_k(x), cfg.n_kv_heads)  # (B, Hkv, T, D)
        v = self._split_heads(self.W_v(x), cfg.n_kv_heads)  # (B, Hkv, T, D)

        # Expand KV heads to match Q heads via repeat_interleave
        # (B, Hkv, T, D) → (B, Hq, T, D)
        if cfg.n_q_per_kv > 1:
            k = k.repeat_interleave(cfg.n_q_per_kv, dim=1)
            v = v.repeat_interleave(cfg.n_q_per_kv, dim=1)

        # Scaled dot-product attention scores
        # (B, Hq, T, T)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self._scale

        # Build or apply causal mask
        if cfg.causal:
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            causal_additive = torch.zeros(T, T, device=x.device, dtype=attn.dtype)
            causal_additive = causal_additive.masked_fill(~causal_mask, float("-inf"))
            attn = attn + causal_additive.unsqueeze(0).unsqueeze(0)

        # Optional extra mask (e.g. padding)
        if mask is not None:
            attn = attn + mask.unsqueeze(1) if mask.dim() == 3 else attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum: (B, Hq, T, D)
        out = torch.matmul(attn, v)

        # Merge heads: (B, T, Hq*D)
        out = out.transpose(1, 2).contiguous().view(B, T, cfg.n_q_heads * cfg.head_dim)

        return self.W_o(out)


# ---------------------------------------------------------------------------
# Full pre-norm transformer layer with GQA
# ---------------------------------------------------------------------------


class GQALayer(nn.Module):
    """Pre-norm transformer layer using GQA and a SiLU FFN.

    Layout:
        x → LayerNorm → GQA → residual → LayerNorm → FFN(SiLU) → residual

    Args:
        config: GQAConfig for the attention sub-layer.
        d_ff:   Hidden dimension of the feed-forward network.
    """

    def __init__(self, config: GQAConfig, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        # Two-layer FFN with SiLU activation
        self.ff_in = nn.Linear(config.d_model, d_ff, bias=False)
        self.ff_out = nn.Linear(d_ff, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one GQA transformer layer with pre-norm residuals.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        # Attention sub-layer
        x = x + self.attn(self.norm1(x))
        # FFN sub-layer with SiLU
        x = x + self.ff_out(F.silu(self.ff_in(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Standard MHA baseline (n_kv_heads == n_q_heads)
# ---------------------------------------------------------------------------


class MultiHeadAttentionBaseline(nn.Module):
    """Standard Multi-Head Attention — equivalent to GQA with n_kv_heads == n_q_heads.

    Provided as a comparison baseline.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of attention heads (both Q and KV).
        head_dim: Per-head dimension.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        cfg = GQAConfig(
            d_model=d_model,
            n_q_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=head_dim,
        )
        self._gqa = GroupedQueryAttention(cfg)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute standard MHA attention.

        Args:
            x:    Input tensor of shape (B, T, d_model).
            mask: Optional additive attention mask.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return self._gqa(x, mask=mask)
