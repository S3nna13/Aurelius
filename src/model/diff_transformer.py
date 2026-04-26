"""Differential Transformer Attention (Microsoft, 2024).

Core idea: split each head's Q and K into two halves and compute a
differential attention map:

    Attn = softmax(Q1 K1ᵀ / √d) − λ · softmax(Q2 K2ᵀ / √d)

where λ is a per-head learnable scalar (initialised ~0.8, clamped [0,1]).
The subtraction cancels common noise between the two attention maps,
producing sharper, less noisy attention.

Reference:
    Tianhao Ding et al., "Differential Transformer", Microsoft Research 2024.
    https://arxiv.org/abs/2410.05258
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DiffAttnConfig:
    """Configuration for DifferentialAttention and its surrounding blocks.

    Args:
        d_model:      Token embedding / hidden dimension.
        n_heads:      Number of *differential* heads (each internally uses two
                      standard softmax heads, so the effective number of
                      attention maps is 2 * n_heads).
        head_dim:     Dimension of each single softmax head.  The combined
                      Q1/Q2 (or K1/K2) projection per head therefore has
                      dimension 2 * head_dim.
        dropout:      Dropout probability applied to the attention output.
        lambda_init:  Initial value for the per-head λ scalar (clamped to
                      [0, 1] during the forward pass).
        d_ff:         FFN hidden dimension.  Defaults to 4 * d_model.
        n_layers:     Number of DiffTransformerBlock layers in a stack.
        rms_norm_eps: Epsilon for RMSNorm.
    """

    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    dropout: float = 0.0
    lambda_init: float = 0.8
    d_ff: int = 0  # 0 → set to 4 * d_model in __post_init__
    n_layers: int = 6
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm used internally by this module."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms.to(x.dtype)) * self.weight.to(x.dtype)


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block used inside DiffTransformerBlock."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Differential Attention
# ---------------------------------------------------------------------------


class DifferentialAttention(nn.Module):
    """Multi-head Differential Attention.

    Each head computes:
        A1 = softmax(Q1 K1ᵀ / √head_dim)
        A2 = softmax(Q2 K2ᵀ / √head_dim)
        out_head = (A1 − λ · A2) V

    where Q1, Q2 are the two halves of the Q projection for that head,
    and λ is a per-head learnable parameter clamped to [0, 1].

    The per-head outputs are concatenated and projected through W_o.
    """

    def __init__(self, config: DiffAttnConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim  # dimension of *each* softmax head
        self.scale = math.sqrt(self.head_dim)

        # Each differential head needs Q1, Q2 (2 * head_dim) and K1, K2 (2 * head_dim)
        # and V (head_dim).  We project everything in a single fused linear for
        # efficiency: output size per head = 2*head_dim (Q) + 2*head_dim (K) + head_dim (V)
        # = 5 * head_dim.  Total = n_heads * 5 * head_dim.
        qkv_dim = config.n_heads * 5 * config.head_dim
        self.qkv_proj = nn.Linear(config.d_model, qkv_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        # Per-head learnable λ — shape (n_heads,)
        self.lambda_param = nn.Parameter(torch.full((config.n_heads,), config.lambda_init))

        self.attn_drop = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lambda_values(self) -> torch.Tensor:
        """λ clamped to [0, 1] — used in forward and exposed for tests."""
        return self.lambda_param.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute differential attention.

        Args:
            x:    (B, T, d_model)
            mask: Optional attention bias of shape (B, 1, T, T) or (1, 1, T, T).
                  Typically a causal mask with 0 at valid positions and -inf
                  (or very large negative) at masked positions.

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H = self.n_heads
        D = self.head_dim  # dimension of each softmax head

        # ---- Project -------------------------------------------------------
        # qkv: (B, T, H * 5 * D)
        qkv = self.qkv_proj(x)

        # Reshape to (B, T, H, 5*D) then split along last dim
        qkv = qkv.view(B, T, H, 5 * D)

        # Q1, Q2: first 2D slice; K1, K2: next 2D; V: last D
        q1 = qkv[..., :D]  # (B, T, H, D)
        q2 = qkv[..., D : 2 * D]  # (B, T, H, D)
        k1 = qkv[..., 2 * D : 3 * D]  # (B, T, H, D)
        k2 = qkv[..., 3 * D : 4 * D]  # (B, T, H, D)
        v = qkv[..., 4 * D :]  # (B, T, H, D)

        # Transpose to (B, H, T, D) for batch-matmul
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v = v.transpose(1, 2)

        # ---- Attention scores -----------------------------------------------
        # (B, H, T, T)
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / self.scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores1 = scores1 + mask
            scores2 = scores2 + mask

        attn1 = F.softmax(scores1, dim=-1)  # (B, H, T, T)
        attn2 = F.softmax(scores2, dim=-1)  # (B, H, T, T)

        # ---- Differential combination ---------------------------------------
        # λ: (H,) → (1, H, 1, 1) for broadcasting
        lam = self.lambda_values.view(1, H, 1, 1)

        # attn_diff: (B, H, T, T)
        attn_diff = self.attn_drop(attn1 - lam * attn2)

        # ---- Value aggregation ---------------------------------------------
        # out_heads: (B, H, T, D)
        out_heads = torch.matmul(attn_diff, v)

        # ---- Merge heads and project out -----------------------------------
        # (B, T, H * D)
        out = out_heads.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(out)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class DiffTransformerBlock(nn.Module):
    """Single Differential Transformer block.

    Structure (pre-norm):
        x = x + DifferentialAttention(RMSNorm(x))
        x = x + SwiGLUFFN(RMSNorm(x))
    """

    def __init__(self, config: DiffAttnConfig) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = DifferentialAttention(config)
        self.ffn_norm = _RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = _SwiGLUFFN(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: Optional (B, 1, T, T) attention bias.

        Returns:
            (B, T, d_model)
        """
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Layer Stack
# ---------------------------------------------------------------------------


class DiffTransformerLayer(nn.Module):
    """Stack of N DiffTransformerBlocks.

    This is the reusable backbone component.  It does *not* include token
    embeddings or a language-model head so it can be composed freely.
    """

    def __init__(self, config: DiffAttnConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DiffTransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = _RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: Optional (B, 1, T, T) attention bias.

        Returns:
            (B, T, d_model)
        """
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)
