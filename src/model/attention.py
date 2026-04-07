"""Grouped-Query Attention with Rotary Position Embeddings.

Supports Flash Attention via PyTorch's scaled_dot_product_attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500_000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the complex-valued RoPE frequency tensor.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    # outer product: (seq_len, head_dim // 2)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensors.

    Args:
        x: (batch, seq_len, n_heads, head_dim) — real-valued.
        freqs_cis: (seq_len, head_dim // 2) — complex-valued.

    Returns:
        Tensor of the same shape as x with RoPE applied.
    """
    # Reshape to pairs of consecutive dims -> view as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Broadcast freqs over batch and heads: (1, seq_len, 1, head_dim//2)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(x.dtype)


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with grouped key/value heads and RoPE.

    Uses PyTorch's scaled_dot_product_attention which automatically dispatches
    to Flash Attention 2, xFormers memory-efficient attention, or the math
    fallback depending on hardware and input characteristics.
    """

    def __init__(self, config: AureliusConfig, apply_rope: bool = True) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor
        self.apply_rope = apply_rope

        # Projections — no bias anywhere
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.attn_dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            freqs_cis: (seq_len, head_dim // 2) — precomputed RoPE frequencies.
            mask: Optional attention mask (broadcastable to (B, H, S, S)).

        Returns:
            (batch, seq_len, d_model)
        """
        B, S, _ = x.shape

        # Project
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys
        if self.apply_rope:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        # Expand KV heads to match Q heads for GQA
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, S, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, S, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, S, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (dispatches to Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=mask is None,  # use built-in causal mask when no explicit mask
        )

        # Reshape back: (B, n_heads, S, head_dim) -> (B, S, d_model)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class DifferentialAttention(nn.Module):
    """Differential Attention (DIFF Transformer, ICLR 2025).

    Computes two softmax attention maps and takes their weighted difference,
    cancelling attention noise and improving focus on relevant tokens.

    Reference: Ye et al., "Differential Transformer", ICLR 2025.
    arXiv: 2410.05258
    """

    def __init__(self, config: AureliusConfig, apply_rope: bool = True) -> None:
        super().__init__()
        assert config.head_dim % 2 == 0, "head_dim must be even for DifferentialAttention"
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.half_dim = config.head_dim // 2
        self.n_rep = config.n_heads // config.n_kv_heads
        self.apply_rope = apply_rope

        # Same projections as GQA — shapes are identical
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        # Per-head learnable lambda (scalar per head, initialized to lambda_init)
        # λ = exp(λ_param) / (exp(λ_param) + 1)  ≈ sigmoid
        lambda_init = config.diff_attn_lambda_init
        self.lambda_param = nn.Parameter(
            torch.full((config.n_heads,), math.log(lambda_init / (1.0 - lambda_init)))
        )
        # Scale factor applied after differencing (1 - lambda_init) per paper
        self.out_scale = 1.0 - lambda_init
        self.attn_dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim // 2 * 2)  # full head_dim for V

        if self.apply_rope:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        # GQA expand
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)

        # Split Q and K into two halves along head_dim
        q1, q2 = q.chunk(2, dim=-1)   # each: (B, S, n_heads, half_dim)
        k1, k2 = k.chunk(2, dim=-1)   # each: (B, S, n_heads, half_dim)

        # Transpose to (B, n_heads, S, half_dim)
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v_t = v.transpose(1, 2)  # (B, n_heads, S, head_dim)

        drop_p = self.attn_dropout if self.training else 0.0

        # Two attention maps — half_dim scaling happens inside SDPA
        a1 = F.scaled_dot_product_attention(q1, k1, v_t, attn_mask=mask, dropout_p=drop_p, is_causal=mask is None)
        a2 = F.scaled_dot_product_attention(q2, k2, v_t, attn_mask=mask, dropout_p=drop_p, is_causal=mask is None)

        # Per-head learnable lambda: sigmoid of param, shape (n_heads, 1, 1) for broadcasting
        lam = torch.sigmoid(self.lambda_param).to(x.dtype).view(1, self.n_heads, 1, 1)

        # Differential combination: (A1 - λ*A2) * (1 - λ_init)
        out = (a1 - lam * a2) * self.out_scale

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)
