"""Tests for model stitching and representational similarity analysis."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.model_stitching import (
    StitchConfig,
    centered_kernel_alignment,
    procrustes_similarity,
    ActivationCollector,
    StitchedModel,
    compare_model_representations,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def model_a(tiny_config):
    torch.manual_seed(0)
    return AureliusTransformer(tiny_config).eval()


@pytest.fixture(scope="module")
def model_b(tiny_config):
    torch.manual_seed(1)
    return AureliusTransformer(tiny_config).eval()


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# 1. StitchConfig defaults
# ---------------------------------------------------------------------------

def test_stitch_config_defaults():
    cfg = StitchConfig()
    assert cfg.stitch_layer == 1
    assert cfg.use_affine is True
    assert cfg.freeze_bottom is True
    assert cfg.freeze_top is False


# ---------------------------------------------------------------------------
# 2. CKA self-similarity approximately 1.0
# ---------------------------------------------------------------------------

def test_cka_self_similarity():
    torch.manual_seed(0)
    X = torch.randn(16, 8)
    val = centered_kernel_alignment(X, X)
    assert abs(val - 1.0) < 1e-4, f"CKA(X, X) should be ~1.0, got {val}"


# ---------------------------------------------------------------------------
# 3. CKA in [0, 1]
# ---------------------------------------------------------------------------

def test_cka_range():
    torch.manual_seed(1)
    X = torch.randn(16, 8)
    Y = torch.randn(16, 12)
    val = centered_kernel_alignment(X, Y)
    assert 0.0 <= val <= 1.0, f"CKA out of [0, 1]: {val}"


# ---------------------------------------------------------------------------
# 4. CKA(X, random Y) < 1.0
# ---------------------------------------------------------------------------

def test_cka_different_matrices():
    torch.manual_seed(2)
    X = torch.randn(16, 8)
    Y = torch.randn(16, 8)
    val = centered_kernel_alignment(X, Y)
    assert val < 1.0, f"CKA(X, random Y) should be < 1.0, got {val}"


# ---------------------------------------------------------------------------
# 5. Procrustes self-similarity high
# ---------------------------------------------------------------------------

def test_procrustes_self_similarity():
    torch.manual_seed(3)
    X = torch.randn(16, 8)
    val = procrustes_similarity(X, X)
    assert val > 0.9, f"procrustes(X, X) should be high, got {val}"


# ---------------------------------------------------------------------------
# 6. Procrustes in reasonable range
# ---------------------------------------------------------------------------

def test_procrustes_range():
    torch.manual_seed(4)
    X = torch.randn(16, 8)
    Y = torch.randn(16, 8)
    val = procrustes_similarity(X, Y)
    assert -1.0 <= val <= 1.0, f"Procrustes out of expected range: {val}"


# ---------------------------------------------------------------------------
# 7. ActivationCollector returns a dict
# ---------------------------------------------------------------------------

def test_activation_collector_returns_dict(model_a, input_ids):
    with ActivationCollector(model_a) as collector:
        acts = collector.collect(input_ids)
    assert isinstance(acts, dict)
    assert len(acts) > 0, "Expected at least one activation entry"


# ---------------------------------------------------------------------------
# 8. ActivationCollector activation shape (B, D)
# ---------------------------------------------------------------------------

def test_activation_collector_shape(model_a, input_ids, tiny_config):
    with ActivationCollector(model_a) as collector:
        acts = collector.collect(input_ids)

    B = input_ids.shape[0]
    D = tiny_config.d_model
    for name, act in acts.items():
        assert act.shape == (B, D), (
            f"Activation '{name}' shape {act.shape} != ({B}, {D})"
        )


# ---------------------------------------------------------------------------
# 9. StitchedModel forward returns correct logits shape (B, T, V)
# ---------------------------------------------------------------------------

def test_stitched_model_forward_shape(model_a, model_b, tiny_config, input_ids):
    cfg = StitchConfig(stitch_layer=1, use_affine=True, freeze_bottom=True, freeze_top=False)
    stitched = StitchedModel(model_a, model_b, cfg)

    with torch.no_grad():
        logits = stitched(input_ids)

    B, T = input_ids.shape
    V = tiny_config.vocab_size
    assert logits.shape == (B, T, V), f"Expected ({B}, {T}, {V}), got {logits.shape}"


# ---------------------------------------------------------------------------
# 10. compare_model_representations returns dict with cka keys
# ---------------------------------------------------------------------------

def test_compare_model_representations_keys(model_a, model_b, input_ids, tiny_config):
    results = compare_model_representations(model_a, model_b, input_ids)
    assert isinstance(results, dict)
    n_layers = tiny_config.n_layers
    for i in range(n_layers):
        assert f"layer_{i}_cka" in results, f"Missing key 'layer_{i}_cka'"
        assert f"layer_{i}_procrustes" in results, f"Missing key 'layer_{i}_procrustes'"


# ---------------------------------------------------------------------------
# 11. Bottom layers are frozen when freeze_bottom=True
# ---------------------------------------------------------------------------

def test_stitched_model_params_frozen(model_a, model_b):
    cfg = StitchConfig(stitch_layer=1, use_affine=True, freeze_bottom=True, freeze_top=False)
    stitched = StitchedModel(model_a, model_b, cfg)

    for p in stitched.embed.parameters():
        assert not p.requires_grad, "Embedding params should be frozen"
    for p in stitched.bottom_layers.parameters():
        assert not p.requires_grad, "Bottom layer params should be frozen"

    if isinstance(stitched.stitching_layer, nn.Linear):
        for p in stitched.stitching_layer.parameters():
            assert p.requires_grad, "Stitching layer params should be trainable"
