"""Q-Former style cross-modal attention for Aurelius multimodal surface.

Inspired by BLIP-2/Q-Former (Salesforce, Apache-2.0, 2301.12597), Gemini 2.5 cross-modal
grounding (Google DeepMind 2025), Whisper cross-attention fusion (OpenAI, MIT),
clean-room reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal Q-Former attention."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    n_query_tokens: int = 32
    dropout: float = 0.0
    feedforward_mult: int = 4


# ---------------------------------------------------------------------------
# CrossModalAttentionLayer
# ---------------------------------------------------------------------------


class CrossModalAttentionLayer(nn.Module):
    """Single Q-Former layer: self-attention over query tokens, then cross-attention to vision.

    Uses pre-norm style (LayerNorm before each sublayer).

    Args:
        d_model: hidden dimension.
        n_heads: number of attention heads.
        dropout: dropout probability (applied in MHA).
        feedforward_mult: multiplier for FFN inner dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        feedforward_mult: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Self-attention over query tokens
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_self = nn.LayerNorm(d_model)

        # Cross-attention: query tokens attend to vision tokens
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_cross = nn.LayerNorm(d_model)

        # Feed-forward network
        ff_dim = d_model * feedforward_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        """Forward pass of one Q-Former layer.

        Args:
            query: (B, Nq, d_model) — learnable query tokens.
            kv:    (B, Nv, d_model) — vision feature tokens (keys and values).

        Returns:
            (B, Nq, d_model) — updated query tokens.
        """
        # Pre-norm self-attention over query tokens
        q_norm = self.ln_self(query)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        query = query + sa_out

        # Pre-norm cross-attention: query attends to vision
        q_norm = self.ln_cross(query)
        ca_out, _ = self.cross_attn(q_norm, kv, kv)
        query = query + ca_out

        # Pre-norm FFN
        q_norm = self.ln_ffn(query)
        query = query + self.ffn(q_norm)

        return query


# ---------------------------------------------------------------------------
# QFormer
# ---------------------------------------------------------------------------


class QFormer(nn.Module):
    """Multi-layer Q-Former: compresses variable-length vision features to fixed query tokens.

    Args:
        d_model: hidden dimension.
        n_heads: number of attention heads.
        n_layers: number of CrossModalAttentionLayer blocks.
        n_query_tokens: number of learnable query token embeddings.
        dropout: dropout probability.
        feedforward_mult: FFN inner-dimension multiplier.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_query_tokens: int,
        dropout: float = 0.0,
        feedforward_mult: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_query_tokens = n_query_tokens

        # Learnable query token embeddings: (1, n_query_tokens, d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, n_query_tokens, d_model))

        self.layers = nn.ModuleList(
            [
                CrossModalAttentionLayer(d_model, n_heads, dropout, feedforward_mult)
                for _ in range(n_layers)
            ]
        )

    @classmethod
    def from_config(cls, cfg: CrossModalConfig) -> QFormer:
        """Construct a QFormer from a CrossModalConfig.

        Args:
            cfg: CrossModalConfig dataclass instance.

        Returns:
            Initialized QFormer module.
        """
        return cls(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            n_query_tokens=cfg.n_query_tokens,
            dropout=cfg.dropout,
            feedforward_mult=cfg.feedforward_mult,
        )

    def forward(self, vision_features: Tensor) -> Tensor:
        """Compress vision features into a fixed-size query token sequence.

        Args:
            vision_features: (B, Nv, d_model) — vision encoder output tokens.

        Returns:
            (B, n_query_tokens, d_model) — compressed query representations.
        """
        B = vision_features.shape[0]

        # Expand learnable query tokens across the batch
        query = self.query_tokens.expand(B, -1, -1)  # (B, n_query_tokens, d_model)

        for layer in self.layers:
            query = layer(query, vision_features)

        return query


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CROSS_MODAL_REGISTRY: dict[str, type[nn.Module]] = {
    "qformer": QFormer,
}

# Register into the shared MODALITY_PROJECTOR_REGISTRY
from src.multimodal.multimodal_registry import register_modality_projector  # noqa: E402

register_modality_projector("QFormer", QFormer)


__all__ = [
    "CrossModalConfig",
    "CrossModalAttentionLayer",
    "QFormer",
    "CROSS_MODAL_REGISTRY",
]
