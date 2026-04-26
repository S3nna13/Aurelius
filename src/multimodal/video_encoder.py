"""Temporal video encoder — 3D patch embedding with MoonViT-style spatiotemporal packing.

Inspired by MoonshotAI/Kimi-K2 MoonViT 3D patch packer (2602.02276), Google Gemini 2.5
doc grounding (Tech Report 2025), Apache-2.0, clean-room reimplementation.
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
class VideoEncoderConfig:
    """Configuration for the temporal video encoder.

    Attributes:
        patch_size:      Spatial patch side length in pixels (square).
        temporal_stride: Temporal stride (also used as temporal kernel size).
        d_model:         Hidden dimension for the encoder.
        n_heads:         Number of attention heads (must divide d_model).
        n_layers:        Number of TransformerEncoderLayer blocks.
        max_frames:      Maximum number of video frames supported.
        dropout:         Dropout probability in transformer layers.
    """

    patch_size: int = 14
    temporal_stride: int = 2
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    max_frames: int = 64
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# 3D Patch Embedding
# ---------------------------------------------------------------------------


class Temporal3DPatchEmbed(nn.Module):
    """3D convolutional patch embedding for video inputs.

    Maps (B, C=3, T, H, W) → (B, N, d_model) where
    N = (T // temporal_stride) * (H // patch_size) * (W // patch_size).

    Args:
        config: :class:`VideoEncoderConfig` instance.
    """

    def __init__(self, config: VideoEncoderConfig) -> None:
        super().__init__()
        self.config = config
        ts = config.temporal_stride
        ps = config.patch_size
        self.proj = nn.Conv3d(
            in_channels=3,
            out_channels=config.d_model,
            kernel_size=(ts, ps, ps),
            stride=(ts, ps, ps),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Embed a video tensor into a sequence of patch tokens.

        Args:
            x: (B, 3, T, H, W) float tensor.

        Returns:
            (B, N, d_model) where N = T_p * H_p * W_p.
        """
        # x: (B, 3, T, H, W)
        out = self.proj(x)  # (B, d_model, T_p, H_p, W_p)
        B, D, T_p, H_p, W_p = out.shape
        # Flatten spatial/temporal dims → (B, N, d_model)
        out = out.flatten(2).transpose(1, 2).contiguous()  # (B, N, d_model)
        return out


# ---------------------------------------------------------------------------
# Temporal Position Encoding
# ---------------------------------------------------------------------------


class TemporalPositionEncoding(nn.Module):
    """Learnable 1-D positional embeddings over the flattened patch sequence.

    In principle this is a 3D position embedding (frame × row × col), but for
    architectural simplicity we use a single learnable table over the maximum
    sequence length N_max = max_frames * max_h_patches * max_w_patches.
    For each forward call the table is interpolated/sliced to the actual N.

    Args:
        config: :class:`VideoEncoderConfig` instance.
        max_h_patches: Maximum number of row patches expected (H // patch_size).
        max_w_patches: Maximum number of column patches expected.
    """

    def __init__(
        self,
        config: VideoEncoderConfig,
        max_h_patches: int = 16,
        max_w_patches: int = 16,
    ) -> None:
        super().__init__()
        self.config = config
        max_t = config.max_frames // config.temporal_stride
        max_n = max_t * max_h_patches * max_w_patches
        # Learnable table: (1, max_n, d_model)
        self.embed = nn.Parameter(torch.zeros(1, max_n, config.d_model))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to patch tokens.

        Args:
            x: (B, N, d_model) patch token tensor.

        Returns:
            (B, N, d_model) with position embeddings added.
        """
        N = x.shape[1]
        # Slice to actual sequence length (truncate if exceeds table)
        pos = self.embed[:, :N, :]
        return x + pos


# ---------------------------------------------------------------------------
# Video Encoder
# ---------------------------------------------------------------------------


class VideoEncoder(nn.Module):
    """Temporal video encoder using 3D patch embedding and transformer layers.

    Pipeline: 3D-patch-embed → pos-encode → N × TransformerEncoderLayer → LayerNorm.

    Args:
        config: :class:`VideoEncoderConfig` instance.
    """

    def __init__(self, config: VideoEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.patch_embed = Temporal3DPatchEmbed(config)
        self.pos_encode = TemporalPositionEncoding(config)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_model * 4,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a video tensor.

        Args:
            x: (B, 3, T, H, W) float tensor in [0, 1] or standardised.

        Returns:
            (B, N, d_model) where N = (T//ts) * (H//ps) * (W//ps).
        """
        x = self.patch_embed(x)  # (B, N, d_model)
        x = self.pos_encode(x)  # (B, N, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    @classmethod
    def from_config(cls, cfg: VideoEncoderConfig) -> VideoEncoder:
        """Construct a VideoEncoder from a :class:`VideoEncoderConfig`.

        Args:
            cfg: Configuration dataclass.

        Returns:
            Initialised :class:`VideoEncoder`.
        """
        return cls(cfg)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

VIDEO_ENCODER_REGISTRY: dict[str, type[VideoEncoder]] = {
    "temporal_3d": VideoEncoder,
}

# ---------------------------------------------------------------------------
# Wire into the shared VISION_ENCODER_REGISTRY
# ---------------------------------------------------------------------------
# Import is deferred to avoid circular import with src.multimodal.__init__.
# We guard the import so the module is still usable standalone.

try:
    from src.multimodal.multimodal_registry import register_vision_encoder as _reg_ve

    _reg_ve("VideoEncoder", VideoEncoder)
except Exception:  # pragma: no cover  # noqa: S110
    pass


__all__ = [
    "VideoEncoderConfig",
    "Temporal3DPatchEmbed",
    "TemporalPositionEncoding",
    "VideoEncoder",
    "VIDEO_ENCODER_REGISTRY",
]
