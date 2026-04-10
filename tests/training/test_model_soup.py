"""Tests for model_soup.py: weight-space ensembling and model soups.

Covers ModelSoupConfig, uniform_soup_module, weighted_soup_module,
greedy_soup_module, interpolate_models_module, compute_weight_distance,
task_vector, apply_task_vector, and ModelSoupBuilder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.model_soup import (
    ModelSoupConfig,
    ModelSoupBuilder,
    uniform_soup_module,
    weighted_soup_module,
    greedy_soup_module,
    interpolate_models_module,
    compute_weight_distance,
    task_vector,
    apply_task_vector,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
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


def _make_model(seed: int) -> AureliusTransformer:
    torch.manual_seed(seed)
    return AureliusTransformer(TINY_CFG)


# Pre-build a few models for reuse
_MODEL_A = _make_model(0)
_MODEL_B = _make_model(1)
_MODEL_C = _make_model(2)


# Simple deterministic val_fn: negative L1 norm of first param (stable).
def _val_fn_stable(model: nn.Module) -> float:
    first_param = next(model.parameters())
    return -float(first_param.float().abs().sum().item())


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """ModelSoupConfig should have correct defaults."""
    cfg = ModelSoupConfig()
    assert cfg.method == "uniform"
    assert cfg.normalize_weights is True


# ---------------------------------------------------------------------------
# 2. test_uniform_soup_returns_module
# ---------------------------------------------------------------------------

def test_uniform_soup_returns_module():
    """uniform_soup_module should return an nn.Module."""
    result = uniform_soup_module([_MODEL_A, _MODEL_B])
    assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# 3. test_uniform_soup_identical_models
# ---------------------------------------------------------------------------

def test_uniform_soup_identical_models():
    """Averaging identical models should produce identical weights."""
    # Make a copy with the same weights as _MODEL_A
    model_copy = _make_model(0)  # same seed → same weights
    result = uniform_soup_module([_MODEL_A, model_copy])
    sd_a = _MODEL_A.state_dict()
    sd_r = result.state_dict()
    for key in sd_a:
        assert torch.allclose(sd_r[key].float(), sd_a[key].float(), atol=1e-5), (
            f"Key {key}: uniform soup of identical models should be unchanged"
        )


# ---------------------------------------------------------------------------
# 4. test_uniform_soup_two_different_models
# ---------------------------------------------------------------------------

def test_uniform_soup_two_different_models():
    """uniform_soup_module with 2 different models should produce exact midpoint."""
    result = uniform_soup_module([_MODEL_A, _MODEL_B])
    sd_a = _MODEL_A.state_dict()
    sd_b = _MODEL_B.state_dict()
    sd_r = result.state_dict()
    for key in sd_a:
        expected = (sd_a[key].float() + sd_b[key].float()) / 2.0
        assert torch.allclose(sd_r[key].float(), expected, atol=1e-5), (
            f"Key {key}: uniform soup should be exact midpoint"
        )


# ---------------------------------------------------------------------------
# 5. test_weighted_soup_output_shape_correct
# ---------------------------------------------------------------------------

def test_weighted_soup_output_shape_correct():
    """weighted_soup_module output should have same parameter shapes as inputs."""
    result = weighted_soup_module([_MODEL_A, _MODEL_B], weights=[0.3, 0.7])
    assert isinstance(result, nn.Module)
    sd_a = _MODEL_A.state_dict()
    sd_r = result.state_dict()
    for key in sd_a:
        assert sd_r[key].shape == sd_a[key].shape, (
            f"Key {key}: shape mismatch in weighted_soup output"
        )


# ---------------------------------------------------------------------------
# 6. test_weighted_soup_weight_1_0_equals_first_model
# ---------------------------------------------------------------------------

def test_weighted_soup_weight_1_0_equals_first_model():
    """weighted_soup_module with weights=[1.0, 0.0] should equal first model."""
    result = weighted_soup_module([_MODEL_A, _MODEL_B], weights=[1.0, 0.0])
    sd_a = _MODEL_A.state_dict()
    sd_r = result.state_dict()
    for key in sd_a:
        assert torch.allclose(sd_r[key].float(), sd_a[key].float(), atol=1e-5), (
            f"Key {key}: weight=1.0 on first model should give first model's params"
        )


# ---------------------------------------------------------------------------
# 7. test_greedy_soup_returns_module_and_list
# ---------------------------------------------------------------------------

def test_greedy_soup_returns_module_and_list():
    """greedy_soup_module should return (nn.Module, list)."""
    soup, indices = greedy_soup_module(
        [_MODEL_A, _MODEL_B, _MODEL_C], val_fn=_val_fn_stable
    )
    assert isinstance(soup, nn.Module)
    assert isinstance(indices, list)


# ---------------------------------------------------------------------------
# 8. test_greedy_soup_included_indices_starts_with_zero
# ---------------------------------------------------------------------------

def test_greedy_soup_included_indices_starts_with_zero():
    """greedy_soup_module included_indices must always contain index 0."""
    _, indices = greedy_soup_module(
        [_MODEL_A, _MODEL_B, _MODEL_C], val_fn=_val_fn_stable
    )
    assert len(indices) >= 1
    assert indices[0] == 0


# ---------------------------------------------------------------------------
# 9. test_interpolate_models_alpha_1_equals_model_a
# ---------------------------------------------------------------------------

def test_interpolate_models_alpha_1_equals_model_a():
    """interpolate_models_module with alpha=1.0 should equal model_a."""
    result = interpolate_models_module(_MODEL_A, _MODEL_B, alpha=1.0)
    assert isinstance(result, nn.Module)
    sd_a = _MODEL_A.state_dict()
    sd_r = result.state_dict()
    for key in sd_a:
        assert torch.allclose(sd_r[key].float(), sd_a[key].float(), atol=1e-5), (
            f"Key {key}: alpha=1.0 should return model_a parameters"
        )


# ---------------------------------------------------------------------------
# 10. test_interpolate_models_alpha_0_equals_model_b
# ---------------------------------------------------------------------------

def test_interpolate_models_alpha_0_equals_model_b():
    """interpolate_models_module with alpha=0.0 should equal model_b."""
    result = interpolate_models_module(_MODEL_A, _MODEL_B, alpha=0.0)
    assert isinstance(result, nn.Module)
    sd_b = _MODEL_B.state_dict()
    sd_r = result.state_dict()
    for key in sd_b:
        assert torch.allclose(sd_r[key].float(), sd_b[key].float(), atol=1e-5), (
            f"Key {key}: alpha=0.0 should return model_b parameters"
        )


# ---------------------------------------------------------------------------
# 11. test_compute_weight_distance_identical_models
# ---------------------------------------------------------------------------

def test_compute_weight_distance_identical_models():
    """compute_weight_distance should return 0 for identical models."""
    dist = compute_weight_distance(_MODEL_A, _MODEL_A)
    assert dist == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 12. test_compute_weight_distance_different_models
# ---------------------------------------------------------------------------

def test_compute_weight_distance_different_models():
    """compute_weight_distance should return positive value for different models."""
    dist = compute_weight_distance(_MODEL_A, _MODEL_B)
    assert isinstance(dist, float)
    assert dist > 0.0


# ---------------------------------------------------------------------------
# 13. test_task_vector_returns_dict_with_correct_keys
# ---------------------------------------------------------------------------

def test_task_vector_returns_dict_with_correct_keys():
    """task_vector should return a dict with the same keys as the model."""
    tv = task_vector(_MODEL_A, _MODEL_B)
    assert isinstance(tv, dict)
    expected_keys = set(_MODEL_A.state_dict().keys())
    assert set(tv.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 14. test_apply_task_vector_scale_zero_equals_base
# ---------------------------------------------------------------------------

def test_apply_task_vector_scale_zero_equals_base():
    """apply_task_vector with scale=0 should return a model equal to base."""
    tv = task_vector(_MODEL_A, _MODEL_B)
    result = apply_task_vector(_MODEL_A, tv, scale=0.0)
    assert isinstance(result, nn.Module)
    sd_base = _MODEL_A.state_dict()
    sd_result = result.state_dict()
    for key in sd_base:
        assert torch.allclose(sd_result[key].float(), sd_base[key].float(), atol=1e-5), (
            f"Key {key}: scale=0 should leave base unchanged"
        )


# ---------------------------------------------------------------------------
# 15. test_model_soup_builder_add_increases_length
# ---------------------------------------------------------------------------

def test_model_soup_builder_add_increases_length():
    """ModelSoupBuilder.add should increase __len__ by 1 each call."""
    builder = ModelSoupBuilder(ModelSoupConfig(method="uniform"))
    assert len(builder) == 0
    builder.add(_MODEL_A, score=0.9)
    assert len(builder) == 1
    builder.add(_MODEL_B, score=0.85)
    assert len(builder) == 2


# ---------------------------------------------------------------------------
# 16. test_model_soup_builder_build_uniform_returns_module
# ---------------------------------------------------------------------------

def test_model_soup_builder_build_uniform_returns_module():
    """ModelSoupBuilder.build with uniform method should return an nn.Module."""
    builder = ModelSoupBuilder(ModelSoupConfig(method="uniform"))
    builder.add(_MODEL_A, score=0.9)
    builder.add(_MODEL_B, score=0.85)
    builder.add(_MODEL_C, score=0.88)
    soup = builder.build()
    assert isinstance(soup, nn.Module)
    # Verify parameter keys match
    assert set(soup.state_dict().keys()) == set(_MODEL_A.state_dict().keys())
