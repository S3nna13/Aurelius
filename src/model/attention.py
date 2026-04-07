"""Grouped-Query Attention with Rotary Position Embeddings.

Supports Flash Attention via PyTorch's scaled_dot_product_attention.
"""

from __future__ import annotations

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

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor

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


