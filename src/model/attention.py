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


def yarn_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500_000.0,
    scale: float = 4.0,
    original_max_seq_len: int = 8192,
    beta_low: float = 1.0,
    beta_high: float = 32.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute RoPE frequency tensor with YaRN context extension.

    YaRN (Yet another RoPE extensioN) applies dimension-dependent frequency
    scaling, enabling context extension beyond the original training length.

    Args:
        head_dim: Attention head dimension.
        max_seq_len: New (extended) maximum sequence length.
        theta: RoPE theta base.
        scale: Context extension factor (e.g., 4 = 4x longer context).
        original_max_seq_len: Original training context length (before extension).
        beta_low: Lower ramp boundary (wavelengths relative to original_max_seq_len).
        beta_high: Upper ramp boundary.
        device: Target device.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    # Number of frequency pairs
    half_dim = head_dim // 2

    # Dimension indices: 0, 2, 4, ..., head_dim-2  →  k = 0, 1, ..., half_dim-1
    k = torch.arange(0, half_dim, device=device).float()

    # Base frequencies: 1 / theta^(2k/head_dim)
    base_freqs = 1.0 / (theta ** (2 * k / head_dim))

    # Wavelength of each dimension relative to original context:
    # wavelength_k = 2π / base_freqs[k]
    # relative_wavelength = wavelength_k / original_max_seq_len
    wavelengths = 2 * torch.pi / base_freqs  # (half_dim,)
    relative = wavelengths / original_max_seq_len  # (half_dim,)

    # Ramp function γ(k): 0 for low-freq (extrapolate), 1 for high-freq (interpolate)
    # γ = clamp((relative - beta_high) / (beta_low - beta_high), 0, 1)
    gamma = ((relative - beta_high) / (beta_low - beta_high)).clamp(0.0, 1.0)

    # Apply YaRN scaling:
    # - Interpolation (high γ): divide frequency by scale (positions scaled by 1/s)
    # - Extrapolation (low γ): keep frequency unchanged
    # Blend: scaled_freq = (1 - γ) * base_freq + γ * (base_freq / scale)
    # Equivalently: scaled_freq = base_freq * (1 - γ * (1 - 1/scale))
    scaled_freqs = base_freqs * (1.0 - gamma * (1.0 - 1.0 / scale))

    # Compute positions for the extended context
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: (max_seq_len, half_dim)
    angles = torch.outer(positions, scaled_freqs)

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
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            freqs_cis: (seq_len, head_dim // 2) — precomputed RoPE frequencies for
                the NEW tokens only (the transformer passes the correctly offset slice).
            mask: Optional attention mask (broadcastable to (B, H, S_total, S_total)).
            past_kv: Optional cached (k, v) tensors from previous steps, each of shape
                (batch, past_seq_len, n_kv_heads, head_dim). Stored pre-GQA-expansion.

        Returns:
            (output, (k_cache, v_cache)) where:
              - output has shape (batch, seq_len, d_model)
              - k_cache and v_cache have shape (batch, past_seq_len + seq_len, n_kv_heads, head_dim)
        """
        B, S, _ = x.shape

        if past_kv is not None and S > 1:
            raise ValueError(
                "KV cache only supports single-token decode (S=1). "
                "For multi-token generation with cache, pass past_kv=None and use a causal mask."
            )

        # Project new tokens
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k_new = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v_new = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and new keys (freqs_cis covers only the new S positions)
        q = apply_rope(q, freqs_cis)
        k_new = apply_rope(k_new, freqs_cis)

        # Concat with cached KV (pre-GQA-expansion). Cached tokens already have RoPE applied.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k_new], dim=1)
            v = torch.cat([past_v, v_new], dim=1)
        else:
            k = k_new
            v = v_new

        # Store cache at n_kv_heads size (pre-expansion)
        k_cache = k
        v_cache = v

        S_total = k.shape[1]

        # Expand KV heads to match Q heads for GQA
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, S_total, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, S_total, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, S_total, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, S_total, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, S, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # is_causal=True only during full prefill with no explicit mask or cache
        is_causal = mask is None and past_kv is None

        # Scaled dot-product attention (dispatches to Flash Attention when available)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back: (B, n_heads, S, head_dim) -> (B, S, d_model)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), (k_cache, v_cache)
