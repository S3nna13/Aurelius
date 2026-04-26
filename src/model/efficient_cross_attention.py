"""Efficient Cross-Attention Variants.

Three designs for attending to large external contexts:
1. Perceiver-style: compressed latent queries, full context K/V
2. Gated: Flamingo-style tanh gate on cross-attention output
3. Linear: kernel-based O(N+M) cross-attention

References:
    Jaegle et al. 2021 (Perceiver) — https://arxiv.org/abs/2103.03206
    Alayrac et al. 2022 (Flamingo) — https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PerceiverCrossAttention(nn.Module):
    """Perceiver-style cross-attention: latent queries attend to context K/V.

    Complexity is O(M*N) where M is the number of latent tokens (small) and
    N is the number of context tokens (large). Useful for compressing a large
    context into a fixed-size latent representation.

    Args:
        d_latent: Dimension of latent query tokens.
        d_context: Dimension of context tokens (keys/values).
        n_heads: Number of attention heads.
        d_head: Per-head dimension. Defaults to d_latent // n_heads.
    """

    def __init__(
        self,
        d_latent: int,
        d_context: int,
        n_heads: int = 4,
        d_head: int | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_latent // n_heads
        self.scale = self.d_head**-0.5

        inner = n_heads * self.d_head
        self.q_proj = nn.Linear(d_latent, inner)
        self.kv_proj = nn.Linear(d_context, 2 * inner)
        self.out_proj = nn.Linear(inner, d_latent)

    def forward(self, latent: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            latent:  (B, M, d_latent) — latent query tokens
            context: (B, N, d_context) — context key/value tokens

        Returns:
            (B, M, d_latent)
        """
        B, M, _ = latent.shape
        N = context.shape[1]
        H, D = self.n_heads, self.d_head

        q = self.q_proj(latent)  # (B, M, H*D)
        kv = self.kv_proj(context)  # (B, N, 2*H*D)
        k, v = kv.chunk(2, dim=-1)  # each (B, N, H*D)

        # Reshape to (B, H, seq, D)
        q = q.view(B, M, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, M, N)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, M, D)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, M, H * D)
        return self.out_proj(out)


class GatedCrossAttention(nn.Module):
    """Flamingo-style gated cross-attention.

    Applies cross-attention modulated by a learned tanh gate, initialized to
    zero so the module starts as a no-op and gradually learns to use context.

    Args:
        d_model: Dimension of the main stream.
        d_context: Dimension of external context tokens.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, d_context: int, n_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = PerceiverCrossAttention(d_model, d_context, n_heads)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_context)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       (B, T, d_model)   — main stream
            context: (B, N, d_context) — external context

        Returns:
            (B, T, d_model)
        """
        cross_out = self.cross_attn(self.norm_x(x), self.norm_ctx(context))
        return x + torch.tanh(self.gate) * cross_out


class LinearCrossAttention(nn.Module):
    """O(N+M) cross-attention via the kernel trick.

    Uses ELU+1 feature maps so the softmax is approximated by a positive
    kernel, enabling the associativity trick to compute attention in linear
    time and memory.

    Args:
        d_model:   Dimension of query tokens.
        d_context: Dimension of context tokens.
        d_head:    Projected head dimension (default 16).
    """

    def __init__(self, d_model: int, d_context: int, d_head: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head)
        self.k_proj = nn.Linear(d_context, d_head)
        self.v_proj = nn.Linear(d_context, d_head)
        self.out_proj = nn.Linear(d_head, d_model)

    def _kernel(self, x: Tensor) -> Tensor:
        """ELU+1 feature map — non-negative, as in the Linear Transformer."""
        return F.elu(x) + 1

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       (B, M, d_model)   — query tokens
            context: (B, N, d_context) — key/value tokens

        Returns:
            (B, M, d_model)
        """
        Q = self._kernel(self.q_proj(x))  # (B, M, d_head)
        K = self._kernel(self.k_proj(context))  # (B, N, d_head)
        V = self.v_proj(context)  # (B, N, d_head)

        # Linear attention via associativity:
        #   numerator:   Q @ (K^T @ V)    — shape (B, M, d_head)
        #   denominator: Q @ K.sum(dim=1) — shape (B, M, 1)
        KtV = K.transpose(-2, -1) @ V  # (B, d_head, d_head)
        num = Q @ KtV  # (B, M, d_head)

        K_sum = K.sum(dim=1, keepdim=True)  # (B, 1, d_head)
        denom = (Q @ K_sum.transpose(-2, -1)) + 1e-6  # (B, M, 1)

        out = num / denom  # (B, M, d_head)
        return self.out_proj(out)


class CrossAttentionLayer(nn.Module):
    """Drop-in transformer layer: self-attention + cross-attention + FFN.

    Supports three cross-attention variants:
        - 'gated':    Flamingo-style tanh-gated cross-attention
        - 'perceiver': Perceiver cross-attention (no gate)
        - 'linear':   O(N+M) kernel cross-attention

    All sub-layers use pre-norm residual connections.

    Args:
        d_model:   Main stream dimension.
        d_context: External context dimension.
        n_heads:   Number of heads for self- and cross-attention.
        d_ff:      FFN hidden dimension. Defaults to 4*d_model.
        variant:   One of 'gated', 'perceiver', 'linear'.
    """

    def __init__(
        self,
        d_model: int,
        d_context: int,
        n_heads: int = 4,
        d_ff: int | None = None,
        variant: str = "gated",
    ) -> None:
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        if variant == "gated":
            self.cross_attn = GatedCrossAttention(d_model, d_context, n_heads)
        elif variant == "perceiver":
            self.cross_attn = PerceiverCrossAttention(d_model, d_context, n_heads)
        elif variant == "linear":
            self.cross_attn = LinearCrossAttention(d_model, d_context)
        else:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose 'gated', 'perceiver', or 'linear'."
            )

        self.variant = variant

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       (B, T, d_model)   — main stream
            context: (B, N, d_context) — external context

        Returns:
            (B, T, d_model)
        """
        # Self-attention (pre-norm residual)
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + sa_out

        # Cross-attention (pre-norm for perceiver/linear; gated handles its own norms)
        if self.variant == "gated":
            x = self.cross_attn(x, context)
        else:
            x = x + self.cross_attn(self.norm2(x), context)

        # FFN (pre-norm residual)
        x = x + self.ff(self.norm3(x))

        return x
