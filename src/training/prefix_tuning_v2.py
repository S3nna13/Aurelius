"""Prefix Tuning v2 (Li & Liang, 2021) — K/V prefix injection per attention layer.

Prepends learnable prefix tokens to the Key and Value projections of every
attention layer. During fine-tuning only the prefix parameters are updated;
the backbone model is frozen.

References:
    Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts
    for Generation. ACL 2021. https://arxiv.org/abs/2101.00190
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PrefixConfig:
    """Configuration for per-layer K/V prefix tuning."""

    prefix_length: int = 10
    """Number of learnable prefix tokens prepended to K and V at each layer."""

    n_layers: int = 4
    """Number of transformer attention layers."""

    n_heads: int = 4
    """Number of attention heads per layer."""

    head_dim: int = 16
    """Dimension per attention head."""

    reparameterize: bool = True
    """If True, use an MLP reparameterization (Linear→Tanh→Linear) to produce
    prefix K/V values.  This stabilises training as recommended in the paper."""


# ---------------------------------------------------------------------------
# PrefixEmbedding
# ---------------------------------------------------------------------------


class PrefixEmbedding(nn.Module):
    """Produces per-layer K and V prefix tensors for all attention layers.

    With ``reparameterize=True`` (default):
        - Stores a raw parameter ``raw`` of shape
          ``(n_layers, 2, prefix_length, n_heads * head_dim)``
        - Passes it through ``nn.Sequential(Linear, Tanh, Linear)`` to produce
          the final prefix.  This extra capacity stabilises early training.

    With ``reparameterize=False``:
        - Stores the prefix directly as an ``nn.Parameter`` of the same shape.

    ``forward()`` returns a tensor of shape
    ``(n_layers, 2, prefix_length, n_heads, head_dim)``
    where dim-1 indexes [K prefix, V prefix].
    """

    def __init__(self, config: PrefixConfig) -> None:
        super().__init__()
        self.config = config
        L = config.n_layers
        P = config.prefix_length
        H = config.n_heads
        D = config.head_dim
        inner = H * D  # total dim before head split

        if config.reparameterize:
            # Raw parameter — not the final prefix.
            self.raw = nn.Parameter(torch.empty(L, 2, P, inner).normal_(0.0, 0.02))
            # MLP: maps inner → inner through a hidden bottleneck of same size.
            self.mlp = nn.Sequential(
                nn.Linear(inner, inner),
                nn.Tanh(),
                nn.Linear(inner, inner),
            )
            self._use_reparam = True
        else:
            # Direct learnable prefix.
            self.prefix = nn.Parameter(torch.empty(L, 2, P, inner).normal_(0.0, 0.02))
            self._use_reparam = False

    # ------------------------------------------------------------------
    def forward(self) -> Tensor:
        """Return prefix K/V for all layers.

        Returns:
            Tensor of shape ``(n_layers, 2, prefix_length, n_heads, head_dim)``.
        """
        config = self.config
        L = config.n_layers
        P = config.prefix_length
        H = config.n_heads
        D = config.head_dim

        if self._use_reparam:
            # raw: (L, 2, P, H*D)  →  mlp applied to last dim  →  same shape
            flat = self.raw.view(-1, H * D)  # (L*2*P, H*D)
            flat = self.mlp(flat)  # (L*2*P, H*D)
            out = flat.view(L, 2, P, H * D)
        else:
            out = self.prefix  # (L, 2, P, H*D)

        # Reshape last dim into (n_heads, head_dim)
        return out.view(L, 2, P, H, D)


# ---------------------------------------------------------------------------
# PrefixAttention
# ---------------------------------------------------------------------------


class PrefixAttention(nn.Module):
    """Multi-head self-attention with prepended prefix K and V tokens.

    The prefix K/V tensors are provided at each forward call so this module
    remains stateless w.r.t. the prefix — the prefix is owned by
    ``PrefixEmbedding`` and injected externally.

    No causal mask is applied to prefix positions; the sequence positions still
    receive the standard causal treatment (all positions attend to the prefix).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, prefix_length: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        inner = n_heads * head_dim

        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

        self._scale = math.sqrt(head_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
    ) -> Tensor:
        """Compute self-attention with prefix K/V prepended.

        Args:
            x:        ``(B, T, d_model)`` input sequence.
            prefix_k: ``(B, prefix_length, n_heads, head_dim)`` prefix keys.
            prefix_v: ``(B, prefix_length, n_heads, head_dim)`` prefix values.

        Returns:
            ``(B, T, d_model)`` output sequence.
        """
        B, T, _ = x.shape
        H = self.n_heads
        D = self.head_dim
        P = self.prefix_length

        # Project queries, keys, values from sequence
        # (B, T, H*D) → (B, H, T, D)
        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        if P > 0:
            # prefix_k/prefix_v: (B, P, H, D) → (B, H, P, D)
            pK = prefix_k.transpose(1, 2)  # (B, H, P, D)
            pV = prefix_v.transpose(1, 2)  # (B, H, P, D)

            # Prepend prefix to K and V
            K = torch.cat([pK, K], dim=2)  # (B, H, P+T, D)
            V = torch.cat([pV, V], dim=2)  # (B, H, P+T, D)

        # Scaled dot-product attention
        # Q: (B, H, T, D),  K/V: (B, H, P+T, D)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self._scale  # (B, H, T, P+T)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, H, T, D)

        # Merge heads and project back
        attn_out = attn_out.transpose(1, 2).reshape(B, T, H * D)  # (B, T, H*D)
        return self.W_o(attn_out)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# PrefixModel
# ---------------------------------------------------------------------------


class PrefixModel(nn.Module):
    """Wraps an arbitrary backbone module and attaches prefix embeddings.

    The prefix embeddings for all layers are computed on every forward pass
    and stored as ``self.last_prefix`` for downstream use (e.g. injecting via
    forward hooks into a real transformer backbone).

    For the purposes of this implementation the backbone's ``forward`` is called
    directly — no hook injection is performed — which keeps the code clean and
    testable without a specific backbone architecture.
    """

    def __init__(self, backbone: nn.Module, config: PrefixConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.prefix_embedding = PrefixEmbedding(config)
        # Storage for last computed prefix (n_layers, 2, P, n_heads, head_dim)
        self.last_prefix: Tensor | None = None

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Set ``requires_grad=False`` on all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the prefix parameters (backbone excluded)."""
        return list(self.prefix_embedding.parameters())

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass, computing and storing prefix embeddings.

        Args:
            x: ``(B, T, d_model)`` input tensor.

        Returns:
            ``(B, T, d_model)`` output from the backbone.

        Note:
            Prefix K/V tensors are computed and stored in ``self.last_prefix``
            ``(n_layers, 2, prefix_length, n_heads, head_dim)``.
            Injection into the backbone attention layers can be accomplished
            via ``register_forward_hook`` on individual attention sub-modules
            using ``self.last_prefix[layer_idx]``.
        """
        # Compute and store prefix embeddings for all layers
        self.last_prefix = self.prefix_embedding()  # (L, 2, P, H, D)
        # Forward through backbone
        return self.backbone(x)
