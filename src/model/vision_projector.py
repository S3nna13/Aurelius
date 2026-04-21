"""Vision Projector — ViT hidden dim → LLM hidden dim linear projection with optional
temporal compression pooling.

Implements the vision-to-LLM projection described in Kimi K2.5 §4 (arXiv:2602.02276):
a single linear layer projects ViT patch tokens into the LLM embedding space, with
optional 4× temporal average pooling to reduce sequence length.

Usage:
    cfg = VisionProjectorConfig(vit_hidden=1024, llm_hidden=2048, temporal_pool=True, pool_factor=4)
    proj = VisionProjector(cfg)
    # vision_features: (B, N, 1024) → out: (B, ceil(N/4), 2048)
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
class VisionProjectorConfig:
    """Configuration for the VisionProjector module."""

    vit_hidden: int = 1024      # ViT output feature dimension
    llm_hidden: int = 2048      # LLM input / d_model dimension
    temporal_pool: bool = True  # whether to apply temporal (sequence-length) compression
    pool_factor: int = 4        # average-pool every pool_factor tokens along sequence dim


# ---------------------------------------------------------------------------
# VisionProjector
# ---------------------------------------------------------------------------

class VisionProjector(nn.Module):
    """Linear projector from ViT hidden space to LLM hidden space.

    Optionally applies 1-D average pooling along the sequence dimension (temporal
    compression) before the linear projection, reducing the number of visual tokens
    from N to ceil(N / pool_factor).

    Args:
        config: VisionProjectorConfig dataclass.

    Input:
        vision_features: Tensor of shape (B, N, vit_hidden)

    Output:
        Tensor of shape (B, N_out, llm_hidden) where:
            N_out == N                    if temporal_pool=False
            N_out == ceil(N / pool_factor) if temporal_pool=True
    """

    def __init__(self, config: VisionProjectorConfig) -> None:
        super().__init__()
        self.config = config
        # Bias-free linear projection (bias=False matches spec)
        self.proj = nn.Linear(config.vit_hidden, config.llm_hidden, bias=False)

    def forward(self, vision_features: Tensor) -> Tensor:
        """Project vision features into LLM token space.

        Args:
            vision_features: (B, N, vit_hidden)

        Returns:
            Tensor of shape (B, N_out, llm_hidden).
        """
        # vision_features: [B, N, C]
        if self.config.temporal_pool:
            B, N, C = vision_features.shape
            pool_factor = self.config.pool_factor

            if pool_factor == 1:
                # Identity: no compression
                pooled = vision_features
            else:
                # Pad N so it's divisible by pool_factor (ceil semantics)
                N_out = math.ceil(N / pool_factor)
                pad_len = N_out * pool_factor - N
                if pad_len > 0:
                    # Pad along sequence dim (dim=1) with zeros at the end
                    # vision_features: [B, N, C] → pad last two dims as (0,0, 0,pad_len)
                    x = F.pad(vision_features, (0, 0, 0, pad_len))  # [B, N+pad, C]
                else:
                    x = vision_features  # [B, N, C]

                # avg_pool1d expects [B, C, L] — transpose
                x = x.transpose(1, 2)  # [B, C, N_padded]
                x = F.avg_pool1d(x, kernel_size=pool_factor, stride=pool_factor)  # [B, C, N_out]
                pooled = x.transpose(1, 2)  # [B, N_out, C]
        else:
            pooled = vision_features  # [B, N, C]

        # Linear projection: [B, N_out, vit_hidden] → [B, N_out, llm_hidden]
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

from src.model import MODEL_COMPONENT_REGISTRY  # noqa: E402

MODEL_COMPONENT_REGISTRY["vision_projector"] = VisionProjector


__all__ = [
    "VisionProjectorConfig",
    "VisionProjector",
]
