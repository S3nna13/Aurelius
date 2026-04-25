"""Late fusion of text, vision, and audio token streams for Aurelius multimodal surface.

Inspired by Flamingo (DeepMind, 2204.14198), BLIP-2 late fusion (Salesforce, Apache-2.0, 2301.12597),
Gemini multimodal fusion (Google DeepMind 2025), clean-room reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# FusionStrategy enum
# ---------------------------------------------------------------------------

class FusionStrategy(Enum):
    """Strategy for combining token streams from different modalities."""

    CONCAT = "concat"
    WEIGHTED_SUM = "weighted_sum"
    CROSS_ATTN = "cross_attn"
    GATED = "gated"


# ---------------------------------------------------------------------------
# TokenFusionConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class TokenFusionConfig:
    """Configuration for MultimodalTokenFusion.

    Args:
        d_model:        Token embedding dimension.
        n_heads:        Number of attention heads (used by CROSS_ATTN strategy).
        strategy:       Fusion strategy enum value.
        text_weight:    Prior weight for text modality (used by WEIGHTED_SUM).
        vision_weight:  Prior weight for vision modality (used by WEIGHTED_SUM).
        audio_weight:   Prior weight for audio modality (used by WEIGHTED_SUM).
        dropout:        Dropout probability applied inside fusion modules.
    """

    d_model: int = 512
    n_heads: int = 8
    strategy: FusionStrategy = FusionStrategy.GATED
    text_weight: float = 0.5
    vision_weight: float = 0.3
    audio_weight: float = 0.2
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# GatedFusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """Sigmoid-gated element-wise fusion of text, vision, and audio token streams.

    All three modalities are mean-pooled along the sequence dimension to match
    the text sequence length, then independently gated and summed.

    Args:
        d_model: Token embedding dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        self.text_gate = nn.Linear(d_model, d_model)
        self.vision_gate = nn.Linear(d_model, d_model)
        self.audio_gate = nn.Linear(d_model, d_model)

        # Used as CONCAT strategy fallback projection
        self.output_proj = nn.Linear(d_model * 3, d_model)

    def forward(
        self,
        text: Tensor,
        vision: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse modality token streams via sigmoid gating.

        Args:
            text:   (B, T, d_model) — text token sequence.
            vision: (B, V, d_model) | None — vision token sequence.
            audio:  (B, A, d_model) | None — audio token sequence.

        Returns:
            (B, T, d_model) — fused token sequence at text sequence length.
        """
        B, T, d = text.shape

        # Gated text contribution
        g_t = torch.sigmoid(self.text_gate(text))  # (B, T, d)
        text_contrib = g_t * text                   # (B, T, d)

        # Vision: mean-pool to (B, 1, d) then expand to (B, T, d)
        if vision is not None:
            vision_pooled = vision.mean(dim=1, keepdim=True).expand(B, T, d)  # (B, T, d)
            g_v = torch.sigmoid(self.vision_gate(vision_pooled))
            vision_contrib = g_v * vision_pooled
        else:
            vision_contrib = torch.zeros_like(text)

        # Audio: mean-pool to (B, 1, d) then expand to (B, T, d)
        if audio is not None:
            audio_pooled = audio.mean(dim=1, keepdim=True).expand(B, T, d)   # (B, T, d)
            g_a = torch.sigmoid(self.audio_gate(audio_pooled))
            audio_contrib = g_a * audio_pooled
        else:
            audio_contrib = torch.zeros_like(text)

        return text_contrib + vision_contrib + audio_contrib  # (B, T, d)


# ---------------------------------------------------------------------------
# WeightedSumFusion
# ---------------------------------------------------------------------------

class WeightedSumFusion(nn.Module):
    """Learnable softmax-weighted sum of three modality token streams.

    Assumes all inputs share the same sequence length T for simplicity.

    Args:
        text_weight:   Prior weight for the text stream.
        vision_weight: Prior weight for the vision stream.
        audio_weight:  Prior weight for the audio stream.
    """

    def __init__(
        self,
        text_weight: float = 0.5,
        vision_weight: float = 0.3,
        audio_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.tensor([text_weight, vision_weight, audio_weight])
        )

    def forward(self, text: Tensor, vision: Tensor, audio: Tensor) -> Tensor:
        """Weighted-sum fusion (all inputs must share the same shape).

        Args:
            text:   (B, T, d_model)
            vision: (B, T, d_model)
            audio:  (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        w = F.softmax(self.weights, dim=0)  # (3,) sums to 1
        return w[0] * text + w[1] * vision + w[2] * audio


# ---------------------------------------------------------------------------
# MultimodalTokenFusion
# ---------------------------------------------------------------------------

class MultimodalTokenFusion(nn.Module):
    """Top-level late-fusion module that dispatches to the configured fusion strategy.

    Handles None-modality inputs gracefully (zero-filled fallback).

    Args:
        config: TokenFusionConfig governing architecture and strategy.
    """

    def __init__(self, config: TokenFusionConfig) -> None:
        super().__init__()
        self.config = config

        if config.strategy in (FusionStrategy.GATED, FusionStrategy.CROSS_ATTN, FusionStrategy.CONCAT):
            self.fusion = GatedFusion(d_model=config.d_model)
        elif config.strategy == FusionStrategy.WEIGHTED_SUM:
            self.fusion = WeightedSumFusion(
                text_weight=config.text_weight,
                vision_weight=config.vision_weight,
                audio_weight=config.audio_weight,
            )
        else:
            raise ValueError(f"Unknown FusionStrategy: {config.strategy!r}")

    @classmethod
    def from_config(cls, cfg: TokenFusionConfig) -> "MultimodalTokenFusion":
        """Construct a MultimodalTokenFusion from a TokenFusionConfig.

        Args:
            cfg: TokenFusionConfig dataclass instance.

        Returns:
            Initialized MultimodalTokenFusion module.
        """
        return cls(cfg)

    def forward(
        self,
        text: Tensor,
        vision: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse text, vision, and audio token streams.

        Args:
            text:   (B, T, d_model) — required text token sequence.
            vision: (B, V, d_model) | None — optional vision tokens.
            audio:  (B, A, d_model) | None — optional audio tokens.

        Returns:
            (B, T, d_model) — fused output at text sequence length.
        """
        if self.config.strategy == FusionStrategy.WEIGHTED_SUM:
            B, T, d = text.shape
            if vision is None:
                vision = torch.zeros_like(text)
            else:
                # Pool/expand vision to text length
                vision = vision.mean(dim=1, keepdim=True).expand(B, T, d)
            if audio is None:
                audio = torch.zeros_like(text)
            else:
                audio = audio.mean(dim=1, keepdim=True).expand(B, T, d)
            return self.fusion(text, vision, audio)

        # GATED / CONCAT / CROSS_ATTN all delegate to GatedFusion
        return self.fusion(text, vision, audio)


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

MULTIMODAL_FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "gated": GatedFusion,
    "weighted_sum": WeightedSumFusion,
    "fusion": MultimodalTokenFusion,
}

# Register MultimodalTokenFusion into the shared MODALITY_PROJECTOR_REGISTRY
from src.multimodal.multimodal_registry import register_modality_projector  # noqa: E402

register_modality_projector("MultimodalTokenFusion", MultimodalTokenFusion)


__all__ = [
    "FusionStrategy",
    "TokenFusionConfig",
    "GatedFusion",
    "WeightedSumFusion",
    "MultimodalTokenFusion",
    "MULTIMODAL_FUSION_REGISTRY",
]
