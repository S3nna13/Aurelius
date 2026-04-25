import torch
import pytest
from src.multimodal.vision_projector_v2 import (
    ProjectorConfig,
    VisionProjectorV2,
    VISION_PROJECTOR_V2_REGISTRY,
)


def make_features(B=2, N=16, C=64):
    return torch.randn(B, N, C)


def test_linear_strategy_output_shape():
    cfg = ProjectorConfig(vision_d_model=64, llm_d_model=32, strategy="linear")
    proj = VisionProjectorV2(cfg)
    out = proj(make_features(N=16, C=64))
    assert out.shape == (2, 16, 32)


def test_mlp2x_strategy_output_shape():
    cfg = ProjectorConfig(vision_d_model=64, llm_d_model=32, strategy="mlp2x")
    proj = VisionProjectorV2(cfg)
    out = proj(make_features(N=16, C=64))
    assert out.shape == (2, 16, 32)


def test_pixel_shuffle_strategy_output_shape():
    cfg = ProjectorConfig(
        vision_d_model=64, llm_d_model=32, strategy="pixel_shuffle", pixel_shuffle_scale=2
    )
    proj = VisionProjectorV2(cfg)
    out = proj(make_features(N=16, C=64))
    assert out.shape == (2, 8, 32)


def test_c_abstractor_strategy_output_shape():
    cfg = ProjectorConfig(
        vision_d_model=64, llm_d_model=32, strategy="c_abstractor", pixel_shuffle_scale=2
    )
    proj = VisionProjectorV2(cfg)
    out = proj(make_features(N=16, C=64))
    assert out.shape == (2, 8, 32)


def test_output_dim_property():
    cfg = ProjectorConfig(vision_d_model=64, llm_d_model=128, strategy="linear")
    proj = VisionProjectorV2(cfg)
    assert proj.output_dim == 128


def test_default_config():
    proj = VisionProjectorV2()
    assert proj.config.strategy == "mlp2x"
    assert proj.output_dim == 512


def test_registry_key():
    assert "v2" in VISION_PROJECTOR_V2_REGISTRY
    assert VISION_PROJECTOR_V2_REGISTRY["v2"] is VisionProjectorV2


def test_pixel_shuffle_odd_tokens():
    cfg = ProjectorConfig(
        vision_d_model=64, llm_d_model=32, strategy="pixel_shuffle", pixel_shuffle_scale=2
    )
    proj = VisionProjectorV2(cfg)
    out = proj(make_features(N=15, C=64))
    assert out.shape[0] == 2
    assert out.shape[2] == 32
