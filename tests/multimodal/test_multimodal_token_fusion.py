"""Tests for src/multimodal/multimodal_token_fusion.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.multimodal.multimodal_token_fusion import (
    FusionStrategy,
    GatedFusion,
    MULTIMODAL_FUSION_REGISTRY,
    MultimodalTokenFusion,
    TokenFusionConfig,
    WeightedSumFusion,
)


# ---------------------------------------------------------------------------
# Config and enum tests
# ---------------------------------------------------------------------------

def test_token_fusion_config_defaults():
    cfg = TokenFusionConfig()
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.strategy == FusionStrategy.GATED
    assert cfg.text_weight == 0.5
    assert cfg.vision_weight == 0.3
    assert cfg.audio_weight == 0.2
    assert cfg.dropout == 0.0


def test_fusion_strategy_enum_members():
    assert FusionStrategy.CONCAT is FusionStrategy.CONCAT
    assert FusionStrategy.WEIGHTED_SUM is FusionStrategy.WEIGHTED_SUM
    assert FusionStrategy.CROSS_ATTN is FusionStrategy.CROSS_ATTN
    assert FusionStrategy.GATED is FusionStrategy.GATED


def test_fusion_strategy_values():
    values = {s.value for s in FusionStrategy}
    assert "concat" in values
    assert "weighted_sum" in values
    assert "cross_attn" in values
    assert "gated" in values


# ---------------------------------------------------------------------------
# from_config construction
# ---------------------------------------------------------------------------

def _tiny_cfg() -> TokenFusionConfig:
    return TokenFusionConfig(d_model=64, n_heads=4, strategy=FusionStrategy.GATED)


def test_from_config_builds_module():
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    assert isinstance(model, MultimodalTokenFusion)


def test_from_config_weighted_sum_builds():
    cfg = TokenFusionConfig(d_model=64, n_heads=4, strategy=FusionStrategy.WEIGHTED_SUM)
    model = MultimodalTokenFusion.from_config(cfg)
    assert isinstance(model, MultimodalTokenFusion)
    assert isinstance(model.fusion, WeightedSumFusion)


# ---------------------------------------------------------------------------
# Forward shape tests — GATED
# ---------------------------------------------------------------------------

def _inputs(B: int = 1, T: int = 4, V: int = 6, A: int = 8, d: int = 64):
    torch.manual_seed(42)
    text = torch.randn(B, T, d)
    vision = torch.randn(B, V, d)
    audio = torch.randn(B, A, d)
    return text, vision, audio


def test_forward_all_modalities_shape():
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, vision, audio = _inputs()
    out = model(text, vision, audio)
    assert out.shape == (1, 4, 64)


def test_forward_vision_none_shape():
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, _, audio = _inputs()
    out = model(text, vision=None, audio=audio)
    assert out.shape == (1, 4, 64)


def test_forward_audio_none_shape():
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, vision, _ = _inputs()
    out = model(text, vision=vision, audio=None)
    assert out.shape == (1, 4, 64)


def test_forward_both_optional_none_shape():
    """Text-only forward: vision=None and audio=None must return (1, 4, 64)."""
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, _, _ = _inputs()
    out = model(text, vision=None, audio=None)
    assert out.shape == (1, 4, 64)


# ---------------------------------------------------------------------------
# NaN checks
# ---------------------------------------------------------------------------

def test_no_nan_all_modalities():
    torch.manual_seed(42)
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, vision, audio = _inputs()
    out = model(text, vision, audio)
    assert not out.isnan().any(), "NaN detected in output with all modalities"


def test_no_nan_vision_none():
    torch.manual_seed(42)
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, _, audio = _inputs()
    out = model(text, vision=None, audio=audio)
    assert not out.isnan().any()


def test_no_nan_audio_none():
    torch.manual_seed(42)
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, vision, _ = _inputs()
    out = model(text, vision=vision, audio=None)
    assert not out.isnan().any()


def test_no_nan_both_none():
    torch.manual_seed(42)
    model = MultimodalTokenFusion.from_config(_tiny_cfg())
    text, _, _ = _inputs()
    out = model(text, vision=None, audio=None)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_multimodal_fusion_registry_contains_gated():
    assert "gated" in MULTIMODAL_FUSION_REGISTRY


def test_multimodal_fusion_registry_contains_weighted_sum():
    assert "weighted_sum" in MULTIMODAL_FUSION_REGISTRY


def test_multimodal_fusion_registry_contains_fusion():
    assert "fusion" in MULTIMODAL_FUSION_REGISTRY


def test_modality_projector_registry_has_multimodal_token_fusion():
    """Import triggers registration; MODALITY_PROJECTOR_REGISTRY must have the entry."""
    from src.multimodal.multimodal_registry import MODALITY_PROJECTOR_REGISTRY
    assert "MultimodalTokenFusion" in MODALITY_PROJECTOR_REGISTRY


# ---------------------------------------------------------------------------
# WeightedSumFusion weights
# ---------------------------------------------------------------------------

def test_weighted_sum_weights_sum_to_one():
    """After softmax, the three learnable weights must sum to 1.0."""
    import torch.nn.functional as F
    cfg = TokenFusionConfig(
        d_model=64, n_heads=4, strategy=FusionStrategy.WEIGHTED_SUM,
        text_weight=0.5, vision_weight=0.3, audio_weight=0.2,
    )
    model = MultimodalTokenFusion.from_config(cfg)
    assert isinstance(model.fusion, WeightedSumFusion)
    w = F.softmax(model.fusion.weights, dim=0)
    assert abs(w.sum().item() - 1.0) < 1e-6
