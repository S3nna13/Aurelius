"""Tests for activation steering / representation engineering.

Minimum 15 tests covering the new SteeringConfig / SteeringVector API.
"""
from __future__ import annotations

import torch
import pytest

from src.alignment.activation_steering import (
    SteeringConfig,
    SteeringVector,
    ActivationSteerer,
    SteeringHook,
    collect_activations,
    extract_mean_diff_vector,
    extract_pca_vector,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

B, T, D = 2, 6, TINY_CFG.d_model


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, TINY_CFG.vocab_size, (B, T))


@pytest.fixture(scope="module")
def steering_cfg():
    return SteeringConfig(layer_indices=[0, 1], coefficient=2.0, normalize=True)


# ---------------------------------------------------------------------------
# 1. test_extract_mean_diff_shape
# ---------------------------------------------------------------------------

def test_extract_mean_diff_shape():
    pos = torch.randn(4, D)
    neg = torch.randn(4, D)
    vec = extract_mean_diff_vector(pos, neg, normalize=False)
    assert vec.shape == (D,), f"Expected ({D},), got {vec.shape}"


# ---------------------------------------------------------------------------
# 2. test_extract_mean_diff_normalized
# ---------------------------------------------------------------------------

def test_extract_mean_diff_normalized():
    pos = torch.randn(4, D)
    neg = torch.randn(4, D)
    vec = extract_mean_diff_vector(pos, neg, normalize=True)
    norm = vec.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# 3. test_extract_mean_diff_direction
# ---------------------------------------------------------------------------

def test_extract_mean_diff_direction():
    """mean_diff vector should point from the negative mean toward the positive mean."""
    pos = torch.randn(8, D) + 5.0   # shifted positive cluster
    neg = torch.randn(8, D) - 5.0   # shifted negative cluster
    vec = extract_mean_diff_vector(pos, neg, normalize=False)
    # The dot product with (mean_pos - mean_neg) should be positive
    expected_dir = pos.mean(dim=0) - neg.mean(dim=0)
    dot = (vec * expected_dir).sum().item()
    assert dot > 0, f"Direction should align with pos-neg mean, dot={dot}"


# ---------------------------------------------------------------------------
# 4. test_extract_pca_vector_shape
# ---------------------------------------------------------------------------

def test_extract_pca_vector_shape():
    data = torch.randn(8, D)
    direction, _ = extract_pca_vector(data, normalize=False)
    assert direction.shape == (D,), f"Expected ({D},), got {direction.shape}"


# ---------------------------------------------------------------------------
# 5. test_extract_pca_vector_normalized
# ---------------------------------------------------------------------------

def test_extract_pca_vector_normalized():
    data = torch.randn(8, D)
    direction, _ = extract_pca_vector(data, normalize=True)
    norm = direction.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ---------------------------------------------------------------------------
# 6. test_extract_pca_explained_variance
# ---------------------------------------------------------------------------

def test_extract_pca_explained_variance():
    data = torch.randn(10, D)
    _, ev = extract_pca_vector(data, normalize=True)
    assert 0.0 < ev <= 1.0, f"Explained variance should be in (0, 1], got {ev}"


# ---------------------------------------------------------------------------
# 7. test_collect_activations_shape
# ---------------------------------------------------------------------------

def test_collect_activations_shape(model, input_ids):
    acts = collect_activations(model, input_ids, layer_idx=0)
    assert acts.shape == (B, D), f"Expected ({B}, {D}), got {acts.shape}"


# ---------------------------------------------------------------------------
# 8. test_steering_hook_add_mode
# ---------------------------------------------------------------------------

def test_steering_hook_add_mode(model, input_ids):
    direction = torch.randn(D)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=0, concept="test")
    cfg = SteeringConfig(layer_indices=[0], coefficient=10.0, normalize=True, mode="add")

    model.eval()
    with torch.no_grad():
        _, baseline, _ = model(input_ids)

    with SteeringHook(model, sv, cfg):
        with torch.no_grad():
            _, steered, _ = model(input_ids)

    assert not torch.allclose(baseline, steered), \
        "add-mode steering should change the output"


# ---------------------------------------------------------------------------
# 9. test_steering_hook_project_out
# ---------------------------------------------------------------------------

def test_steering_hook_project_out(model, input_ids):
    direction = torch.randn(D)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=0, concept="test")
    cfg = SteeringConfig(layer_indices=[0], coefficient=1.0, normalize=True,
                         mode="project_out")

    model.eval()
    with torch.no_grad():
        _, baseline, _ = model(input_ids)

    with SteeringHook(model, sv, cfg):
        with torch.no_grad():
            _, steered, _ = model(input_ids)

    assert not torch.allclose(baseline, steered), \
        "project_out-mode steering should change the output"


# ---------------------------------------------------------------------------
# 10. test_steering_hook_context_manager_restores
# ---------------------------------------------------------------------------

def test_steering_hook_context_manager_restores(model, input_ids):
    """After exiting SteeringHook, the model should produce its original output."""
    direction = torch.randn(D)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=0, concept="test")
    cfg = SteeringConfig(layer_indices=[0], coefficient=10.0, normalize=True, mode="add")

    model.eval()
    with torch.no_grad():
        _, before, _ = model(input_ids)

    with SteeringHook(model, sv, cfg):
        pass  # enter and immediately exit

    with torch.no_grad():
        _, after, _ = model(input_ids)

    assert torch.allclose(before, after, atol=1e-6), \
        "Output should be restored after context manager exit"


# ---------------------------------------------------------------------------
# 11. test_steerer_fit_mean_diff_count
# ---------------------------------------------------------------------------

def test_steerer_fit_mean_diff_count(model, input_ids):
    cfg = SteeringConfig(layer_indices=[0, 1], coefficient=1.0)
    steerer = ActivationSteerer(model, cfg)

    pos_ids = input_ids[:1].expand(4, -1)   # (4, T)
    neg_ids = input_ids[1:].expand(4, -1)   # (4, T)

    vectors = steerer.fit_from_pairs(pos_ids, neg_ids, method="mean_diff")
    assert len(vectors) == len(cfg.layer_indices), \
        f"Expected {len(cfg.layer_indices)} vectors, got {len(vectors)}"


# ---------------------------------------------------------------------------
# 12. test_steerer_fit_pca_count
# ---------------------------------------------------------------------------

def test_steerer_fit_pca_count(model, input_ids):
    cfg = SteeringConfig(layer_indices=[0, 1], coefficient=1.0)
    steerer = ActivationSteerer(model, cfg)

    pos_ids = input_ids[:1].expand(4, -1)
    neg_ids = input_ids[1:].expand(4, -1)

    vectors = steerer.fit_from_pairs(pos_ids, neg_ids, method="pca")
    assert len(vectors) == len(cfg.layer_indices), \
        f"Expected {len(cfg.layer_indices)} vectors, got {len(vectors)}"


# ---------------------------------------------------------------------------
# 13. test_steerer_steer_shape
# ---------------------------------------------------------------------------

def test_steerer_steer_shape(model, input_ids):
    cfg = SteeringConfig(layer_indices=[0, 1], coefficient=1.0)
    steerer = ActivationSteerer(model, cfg)

    pos_ids = input_ids[:1].expand(4, -1)
    neg_ids = input_ids[1:].expand(4, -1)
    vectors = steerer.fit_from_pairs(pos_ids, neg_ids, method="mean_diff")

    max_new = 5
    single_input = input_ids[:1]   # (1, T)
    generated = steerer.steer(single_input, vectors, max_new_tokens=max_new)

    assert generated.shape[0] == 1, f"Batch dim should be 1, got {generated.shape[0]}"
    assert generated.shape[1] == max_new, \
        f"Expected {max_new} new tokens, got {generated.shape[1]}"


# ---------------------------------------------------------------------------
# 14. test_measure_concept_activation_scalar
# ---------------------------------------------------------------------------

def test_measure_concept_activation_scalar(model, input_ids):
    direction = torch.randn(D)
    direction = direction / direction.norm()
    sv = SteeringVector(direction=direction, layer_idx=0, concept="test")

    cfg = SteeringConfig(layer_indices=[0])
    steerer = ActivationSteerer(model, cfg)

    result = steerer.measure_concept_activation(input_ids, sv)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# 15. test_steering_vector_normalize
# ---------------------------------------------------------------------------

def test_steering_vector_normalize(model, input_ids):
    """SteeringVector direction should have unit norm when normalize=True."""
    cfg = SteeringConfig(layer_indices=[0], normalize=True)
    steerer = ActivationSteerer(model, cfg)

    pos_ids = input_ids[:1].expand(4, -1)
    neg_ids = input_ids[1:].expand(4, -1)
    vectors = steerer.fit_from_pairs(pos_ids, neg_ids, method="mean_diff")

    for sv in vectors:
        norm = sv.direction.norm().item()
        assert abs(norm - 1.0) < 1e-5, \
            f"Direction at layer {sv.layer_idx} should have unit norm, got {norm}"
