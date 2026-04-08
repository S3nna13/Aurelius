"""Tests for MultiModalProjector."""
import torch
import pytest
from src.model.multimodal_projector import (
    ProjectorConfig,
    ModalityProjector,
    MultiModalProjector,
    build_projector,
)


# ---------------------------------------------------------------------------
# ProjectorConfig
# ---------------------------------------------------------------------------

def test_projector_config_hidden_dim_auto():
    cfg = ProjectorConfig(input_dim=768, output_dim=2048)
    assert cfg.hidden_dim is None
    proj = ModalityProjector(cfg)
    # hidden_dim should be computed as (768 + 2048) // 2 = 1408
    assert proj.hidden_dim == (768 + 2048) // 2


# ---------------------------------------------------------------------------
# ModalityProjector — output shapes
# ---------------------------------------------------------------------------

def test_modality_projector_output_shape_3d():
    cfg = ProjectorConfig(input_dim=768, output_dim=2048)
    proj = ModalityProjector(cfg)
    x = torch.randn(2, 16, 768)
    out = proj(x)
    assert out.shape == (2, 16, 2048)


def test_modality_projector_output_shape_2d():
    cfg = ProjectorConfig(input_dim=768, output_dim=2048)
    proj = ModalityProjector(cfg)
    x = torch.randn(4, 768)
    out = proj(x)
    assert out.shape == (4, 2048)


# ---------------------------------------------------------------------------
# ModalityProjector — layer counts
# ---------------------------------------------------------------------------

def test_modality_projector_n_layers_1():
    cfg = ProjectorConfig(input_dim=256, output_dim=512, n_layers=1)
    proj = ModalityProjector(cfg)
    # Should have exactly 1 Linear: input_dim → output_dim
    linears = [m for m in proj.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 1
    assert linears[0].in_features == 256
    assert linears[0].out_features == 512


def test_modality_projector_n_layers_3():
    cfg = ProjectorConfig(input_dim=256, output_dim=512, n_layers=3, hidden_dim=384)
    proj = ModalityProjector(cfg)
    linears = [m for m in proj.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    # Verify shape: in→hidden, hidden→hidden, hidden→out
    assert linears[0].in_features == 256
    assert linears[0].out_features == 384
    assert linears[1].in_features == 384
    assert linears[1].out_features == 384
    assert linears[2].in_features == 384
    assert linears[2].out_features == 512


# ---------------------------------------------------------------------------
# ModalityProjector — options
# ---------------------------------------------------------------------------

def test_modality_projector_layer_norm_input():
    cfg = ProjectorConfig(input_dim=128, output_dim=256, layer_norm_input=True)
    proj = ModalityProjector(cfg)
    layer_norms = [m for m in proj.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(layer_norms) >= 1
    # Verify output shape still correct
    x = torch.randn(3, 10, 128)
    out = proj(x)
    assert out.shape == (3, 10, 256)


def test_modality_projector_activation_silu():
    cfg = ProjectorConfig(input_dim=128, output_dim=256, activation="silu", n_layers=2)
    proj = ModalityProjector(cfg)
    x = torch.randn(2, 8, 128)
    out = proj(x)
    assert out.shape == (2, 8, 256)
    assert not torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# MultiModalProjector
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_proj():
    configs = {
        "vision": ProjectorConfig(input_dim=768, output_dim=2048),
        "audio": ProjectorConfig(input_dim=1024, output_dim=2048),
    }
    return MultiModalProjector(configs)


def test_multimodal_projector_project_single(multi_proj):
    x = torch.randn(2, 12, 768)
    out = multi_proj.project("vision", x)
    assert out.shape == (2, 12, 2048)


def test_multimodal_projector_project_all(multi_proj):
    features = {
        "vision": torch.randn(2, 12, 768),
        "audio": torch.randn(2, 8, 1024),
    }
    result = multi_proj.project_all(features)
    assert set(result.keys()) == {"vision", "audio"}
    assert result["vision"].shape == (2, 12, 2048)
    assert result["audio"].shape == (2, 8, 2048)


# ---------------------------------------------------------------------------
# build_multimodal_sequence
# ---------------------------------------------------------------------------

def test_build_multimodal_sequence_shape(multi_proj):
    B, S, D = 2, 20, 2048
    N_vision = 12
    text_embeds = torch.randn(B, S, D)
    modality_features = {"vision": torch.randn(B, N_vision, 768)}
    out = multi_proj.build_multimodal_sequence(text_embeds, modality_features)
    assert out.shape == (B, N_vision + S, D)


def test_build_multimodal_sequence_prepend_order(multi_proj):
    B, S, D = 1, 10, 2048
    N_v, N_a = 6, 4
    text_embeds = torch.randn(B, S, D)
    modality_features = {
        "vision": torch.randn(B, N_v, 768),
        "audio": torch.randn(B, N_a, 1024),
    }

    # vision first then audio
    out_va = multi_proj.build_multimodal_sequence(
        text_embeds, modality_features, prepend_order=["vision", "audio"]
    )
    # audio first then vision
    out_av = multi_proj.build_multimodal_sequence(
        text_embeds, modality_features, prepend_order=["audio", "vision"]
    )

    assert out_va.shape == (B, N_v + N_a + S, D)
    assert out_av.shape == (B, N_v + N_a + S, D)
    # The first N_v tokens of out_va should equal projected vision tokens
    # The first N_a tokens of out_av should equal projected audio tokens
    # (they differ because order differs)
    assert not torch.allclose(out_va, out_av)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_projector_factory():
    proj = build_projector(input_dim=512, output_dim=1024)
    assert isinstance(proj, ModalityProjector)
    x = torch.randn(1, 5, 512)
    out = proj(x)
    assert out.shape == (1, 5, 1024)
