"""Tests for src/multimodal/video_encoder.py --- Temporal3DPatchEmbed and VideoEncoder."""

from __future__ import annotations

import torch

from src.multimodal.video_encoder import (
    VIDEO_ENCODER_REGISTRY,
    Temporal3DPatchEmbed,
    VideoEncoder,
    VideoEncoderConfig,
)

# ---------------------------------------------------------------------------
# Tiny test configuration (d_model=64, n_heads=4, n_layers=2)
# ---------------------------------------------------------------------------

TINY_CFG = VideoEncoderConfig(
    patch_size=14,
    temporal_stride=2,
    d_model=64,
    n_heads=4,
    n_layers=2,
    max_frames=64,
    dropout=0.0,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_default_patch_size():
    cfg = VideoEncoderConfig()
    assert cfg.patch_size == 14


def test_config_default_temporal_stride():
    cfg = VideoEncoderConfig()
    assert cfg.temporal_stride == 2


def test_config_default_d_model():
    cfg = VideoEncoderConfig()
    assert cfg.d_model == 512


def test_config_default_n_heads():
    cfg = VideoEncoderConfig()
    assert cfg.n_heads == 8


def test_config_default_n_layers():
    cfg = VideoEncoderConfig()
    assert cfg.n_layers == 4


def test_config_default_max_frames():
    cfg = VideoEncoderConfig()
    assert cfg.max_frames == 64


def test_config_default_dropout():
    cfg = VideoEncoderConfig()
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# Temporal3DPatchEmbed shape tests
# ---------------------------------------------------------------------------


def test_patch_embed_output_shape():
    """(B=1, C=3, T=4, H=28, W=28), ps=14, ts=2 -> N = 2*2*2 = 8."""
    cfg = TINY_CFG
    embed = Temporal3DPatchEmbed(cfg)
    x = torch.randn(1, 3, 4, 28, 28)
    out = embed(x)
    assert out.shape == (1, 8, cfg.d_model)


def test_patch_embed_output_n_patches():
    """Verify N = (T//ts) * (H//ps) * (W//ps) arithmetic."""
    cfg = VideoEncoderConfig(patch_size=7, temporal_stride=2, d_model=32, n_heads=4, n_layers=1)
    embed = Temporal3DPatchEmbed(cfg)
    B, T, H, W = 2, 4, 14, 14
    x = torch.randn(B, 3, T, H, W)
    out = embed(x)
    expected_n = (T // 2) * (H // 7) * (W // 7)  # 2*2*2 = 8
    assert out.shape == (B, expected_n, 32)


def test_patch_embed_no_nan():
    embed = Temporal3DPatchEmbed(TINY_CFG)
    x = torch.randn(1, 3, 4, 28, 28)
    out = embed(x)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# VideoEncoder tests
# ---------------------------------------------------------------------------


def test_video_encoder_from_config():
    enc = VideoEncoder.from_config(TINY_CFG)
    assert isinstance(enc, VideoEncoder)


def test_video_encoder_forward_shape():
    enc = VideoEncoder.from_config(TINY_CFG)
    enc.eval()
    x = torch.randn(1, 3, 4, 28, 28)
    with torch.no_grad():
        out = enc(x)
    # N = 2*2*2 = 8, d_model = 64
    assert out.shape == (1, 8, 64)


def test_video_encoder_no_nan():
    enc = VideoEncoder.from_config(TINY_CFG)
    enc.eval()
    x = torch.randn(1, 3, 4, 28, 28)
    with torch.no_grad():
        out = enc(x)
    assert not torch.isnan(out).any()


def test_video_encoder_no_inf():
    enc = VideoEncoder.from_config(TINY_CFG)
    enc.eval()
    x = torch.randn(1, 3, 4, 28, 28)
    with torch.no_grad():
        out = enc(x)
    assert not torch.isinf(out).any()


def test_video_encoder_batch_dim():
    """Batch dimension must propagate correctly."""
    enc = VideoEncoder.from_config(TINY_CFG)
    enc.eval()
    x = torch.randn(3, 3, 4, 28, 28)
    with torch.no_grad():
        out = enc(x)
    assert out.shape[0] == 3


def test_video_encoder_d_model_dim():
    enc = VideoEncoder.from_config(TINY_CFG)
    enc.eval()
    x = torch.randn(1, 3, 4, 28, 28)
    with torch.no_grad():
        out = enc(x)
    assert out.shape[-1] == TINY_CFG.d_model


def test_video_encoder_minimum_frames():
    """T == temporal_stride (1 temporal patch) must not crash."""
    cfg = TINY_CFG
    enc = VideoEncoder.from_config(cfg)
    enc.eval()
    T = cfg.temporal_stride  # 2
    x = torch.randn(1, 3, T, 28, 28)
    with torch.no_grad():
        out = enc(x)
    expected_n = 1 * 2 * 2  # (T//ts)=1, (28//14)=2, 2x2 spatial grid
    assert out.shape == (1, expected_n, cfg.d_model)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_video_encoder_registry_has_temporal_3d():
    assert "temporal_3d" in VIDEO_ENCODER_REGISTRY


def test_video_encoder_registry_temporal_3d_is_class():
    cls = VIDEO_ENCODER_REGISTRY["temporal_3d"]
    assert issubclass(cls, VideoEncoder)


def test_video_encoder_registry_temporal_3d_instantiable():
    cls = VIDEO_ENCODER_REGISTRY["temporal_3d"]
    enc = cls.from_config(TINY_CFG)
    assert isinstance(enc, VideoEncoder)


def test_video_encoder_registered_in_vision_encoder_registry():
    """VideoEncoder must be wired into VISION_ENCODER_REGISTRY via the module import."""
    from src.multimodal.multimodal_registry import VISION_ENCODER_REGISTRY

    assert "VideoEncoder" in VISION_ENCODER_REGISTRY
