from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ProjectorConfig:
    vision_d_model: int = 768
    llm_d_model: int = 512
    strategy: str = "mlp2x"
    pixel_shuffle_scale: int = 2


class VisionProjectorV2(nn.Module):
    def __init__(self, config: ProjectorConfig | None = None):
        super().__init__()
        if config is None:
            config = ProjectorConfig()
        self.config = config
        s = config.strategy
        vi = config.vision_d_model
        lo = config.llm_d_model

        if s == "linear":
            self.proj = nn.Linear(vi, lo)
        elif s == "mlp2x":
            hidden = lo * 2
            self.proj = nn.Sequential(
                nn.Linear(vi, hidden),
                nn.GELU(),
                nn.Linear(hidden, lo),
            )
        elif s == "pixel_shuffle":
            scale = config.pixel_shuffle_scale
            self.proj = nn.Linear(vi * scale, lo)
        elif s == "c_abstractor":
            self.proj = nn.Linear(vi, lo)
        else:
            raise ValueError(f"Unknown strategy: {s!r}")

    def forward(self, vision_features: Tensor) -> Tensor:
        s = self.config.strategy
        if s in ("linear", "mlp2x"):
            return self.proj(vision_features)
        elif s == "pixel_shuffle":
            scale = self.config.pixel_shuffle_scale
            B, N, C = vision_features.shape
            pad = (scale - N % scale) % scale
            if pad:
                vision_features = torch.cat(
                    [vision_features, vision_features[:, -pad:, :]], dim=1
                )
            N2 = vision_features.shape[1]
            chunks = N2 // scale
            x = vision_features.reshape(B, chunks, scale * C)
            return self.proj(x)
        elif s == "c_abstractor":
            scale = self.config.pixel_shuffle_scale
            B, N, C = vision_features.shape
            pad = (scale - N % scale) % scale
            if pad:
                vision_features = torch.cat(
                    [vision_features, vision_features[:, -pad:, :]], dim=1
                )
            N2 = vision_features.shape[1]
            chunks = N2 // scale
            x = vision_features.reshape(B, chunks, scale, C).mean(dim=2)
            return self.proj(x)

    @property
    def output_dim(self) -> int:
        return self.config.llm_d_model


VISION_PROJECTOR_V2_REGISTRY: dict[str, type[VisionProjectorV2]] = {
    "v2": VisionProjectorV2,
}
