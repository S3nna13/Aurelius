from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class OCRConfig:
    d_model: int = 256
    max_text_regions: int = 64
    patch_size: int = 8
    stride: int = 4


class TextRegionEncoder(nn.Module):
    def __init__(self, config: OCRConfig):
        super().__init__()
        self.conv = nn.Conv2d(
            3, config.d_model, kernel_size=config.patch_size, stride=config.stride
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class LayoutAttention(nn.Module):
    def __init__(self, config: OCRConfig, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.d_model, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        out, _ = self.attn(queries, keys, keys)
        return self.norm(queries + out)


class OCRModule(nn.Module):
    def __init__(self, config: OCRConfig | None = None):
        super().__init__()
        if config is None:
            config = OCRConfig()
        self.config = config
        self.encoder = TextRegionEncoder(config)
        self.layout_attn = LayoutAttention(config)
        self.text_queries = nn.Parameter(
            torch.zeros(1, config.max_text_regions, config.d_model)
        )
        nn.init.trunc_normal_(self.text_queries, std=0.02)

    def forward(self, image: Tensor) -> Tensor:
        B = image.shape[0]
        patch_features = self.encoder(image)
        queries = self.text_queries.expand(B, -1, -1)
        out = self.layout_attn(queries, patch_features)
        return out

    def extract_features(self, image: Tensor) -> Tensor:
        return self.forward(image)


OCR_MODULE_REGISTRY: dict[str, type[OCRModule]] = {
    "default": OCRModule,
}
