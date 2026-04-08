"""Gemma-3-style interleaved local/global attention.

Implements a 5:1 interleaving ratio where every 6th layer (layer_idx % 6 == 5)
uses full causal (global) attention with a high RoPE theta, while all other
layers use sliding-window (local) attention with a 1024-token window and a
lower RoPE theta.

Both layer types use the same GQA structure (n_heads=16, n_kv_heads=8,
head_dim=128) as the rest of the Aurelius model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AureliusConfig
from .attention import apply_rope, precompute_rope_frequencies

# RoPE theta values for each attention type (Gemma-3 convention)
LOCAL_ROPE_THETA: float = 10_000.0
GLOBAL_ROPE_THETA: float = 1_000_000.0

# Sliding-window size for local attention
LOCAL_WINDOW_SIZE: int = 1_024


def is_global_layer(layer_idx: int) -> bool:
    """Return True if *layer_idx* should use global (full-context) attention.

    Every 6th layer — i.e. layer_idx % 6 == 5 — is a global layer.  All other
    layers are local (sliding-window) layers.

    Args:
        layer_idx: Zero-based index of the transformer layer.

    Returns:
        True for global layers, False for local layers.
    """
    return layer_idx % 6 == 5


def _build_local_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Build a banded causal mask for sliding-window attention.

    Token *i* may attend to token *j* only when ``i - window_size < j <= i``.
    Forbidden positions receive ``-inf``; allowed positions receive ``0.0``.

    Args:
        seq_len: Sequence length.
        window_size: Number of past tokens each position can attend to.
        device: Target device.

    Returns:
        Float tensor of shape ``(seq_len, seq_len)``.
    """
    # Lower-triangular band: allowed positions are 1, forbidden are 0.
    ones = torch.ones(seq_len, seq_len, device=device)
    causal_band = torch.tril(ones) - torch.tril(ones, diagonal=-(window_size + 1))
    # Convert: allowed → 0.0, forbidden → -inf
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask[causal_band == 0] = float("-inf")
    return mask  # (seq_len, seq_len)


class LocalAttentionLayer(nn.Module):
    """Sliding-window grouped-query attention layer.

    Identical projection structure to ``GroupedQueryAttention``, but the
    attention scores are masked so that each token can only attend to the
    ``window_size`` most-recently-seen tokens (plus itself).  RoPE uses a
    lower theta (10 000) suited for short-range dependencies.

    Args:
        config: Global model hyperparameters.
        window_size: Sliding-window size in tokens (default: 1 024).
    """

    def __init__(self, config: AureliusConfig, window_size: int = LOCAL_WINDOW_SIZE) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.window_size = window_size
        self.rope_theta = LOCAL_ROPE_THETA
        self.attn_dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run local attention over ``x``.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            freqs_cis: Optional precomputed RoPE frequencies
                ``(seq_len, head_dim // 2)``.  When *None*, frequencies are
                computed on the fly using ``LOCAL_ROPE_THETA``.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        B, S, _ = x.shape
        device = x.device

        if freqs_cis is None:
            freqs_cis = precompute_rope_frequencies(
                self.head_dim, S, theta=self.rope_theta, device=device
            )

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # GQA: expand k/v to match number of query heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, S, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, S, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, S, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build sliding-window mask: (S, S) → broadcast over (B, n_heads, S, S)
        mask = _build_local_mask(S, self.window_size, device=device)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class GlobalAttentionLayer(nn.Module):
    """Full-context causal grouped-query attention layer.

    Attends across the entire sequence with no window restriction.  Uses a
    high RoPE theta (1 000 000) to support long-range position encoding.

    Args:
        config: Global model hyperparameters.
    """

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.rope_theta = GLOBAL_ROPE_THETA
        self.attn_dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run full causal attention over ``x``.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            freqs_cis: Optional precomputed RoPE frequencies
                ``(seq_len, head_dim // 2)``.  When *None*, frequencies are
                computed on the fly using ``GLOBAL_ROPE_THETA``.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        B, S, _ = x.shape
        device = x.device

        if freqs_cis is None:
            freqs_cis = precompute_rope_frequencies(
                self.head_dim, S, theta=self.rope_theta, device=device
            )

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # GQA: expand k/v to match number of query heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, S, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, S, self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, S, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Full causal attention — no explicit mask needed
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class InterleavedAttentionLayer(nn.Module):
    """Attention layer that picks local or global attention based on layer index.

    Uses the 5:1 interleaving rule: ``layer_idx % 6 == 5`` → global attention,
    all other indices → local (sliding-window) attention.

    Args:
        config: Global model hyperparameters.
        layer_idx: Zero-based index of this layer in the transformer stack.
        window_size: Sliding-window size passed to the local layer.
    """

    def __init__(
        self,
        config: AureliusConfig,
        layer_idx: int,
        window_size: int = LOCAL_WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._is_global = is_global_layer(layer_idx)

        if self._is_global:
            self.attn = GlobalAttentionLayer(config)
        else:
            self.attn = LocalAttentionLayer(config, window_size=window_size)

    @property
    def rope_theta(self) -> float:
        """RoPE theta used by the underlying attention layer."""
        return self.attn.rope_theta

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — delegates to the wrapped local or global layer.

        Args:
            x: ``(batch, seq_len, d_model)``
            freqs_cis: Optional precomputed RoPE frequencies.

        Returns:
            ``(batch, seq_len, d_model)``
        """
        return self.attn(x, freqs_cis=freqs_cis)
