from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    variant: str = "B"


class ViTPatchEmbedding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        hidden = int(config.d_model * config.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig | None = None):
        super().__init__()
        if config is None:
            config = ViTConfig()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embed = ViTPatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, config.d_model))
        self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens

    def encode_cls(self, x: Tensor) -> Tensor:
        return self.forward(x)[:, 0]


VIT_REGISTRY: dict[str, ViTConfig] = {
    "ViT-B": ViTConfig(variant="B"),
    "ViT-L": ViTConfig(d_model=1024, n_heads=16, n_layers=24, variant="L"),
}
