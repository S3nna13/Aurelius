"""Tests for model_soup.py: weight-space ensembling and model soups."""
from __future__ import annotations

import torch
import pytest

from src.training.model_soup import (
    ModelSoupConfig,
    ModelSoupEnsemble,
    LearnedSoupMixer,
    compute_loss_barrier,
    compute_weight_divergence,
    greedy_soup,
    interpolate_models,
    uniform_soup,
    weighted_soup,
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


def _random_eval_fn(model) -> float:
    """Deterministic-ish eval: returns random scalar (just tests structure)."""
    return torch.randn(1).item()


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ModelSoupConfig()
    assert cfg.method == "uniform"
    assert cfg.n_models == 3


# ---------------------------------------------------------------------------
# 2. test_uniform_soup_keys
# ---------------------------------------------------------------------------

def test_uniform_soup_keys():
    result = uniform_soup([_MODEL_A, _MODEL_B])
    assert set(result.keys()) == set(_MODEL_A.state_dict().keys())


# ---------------------------------------------------------------------------
# 3. test_uniform_soup_is_average
# ---------------------------------------------------------------------------

def test_uniform_soup_is_average():
    """Two models with known weights → verify average."""
    result = uniform_soup([_MODEL_A, _MODEL_B])
    sd_a = _MODEL_A.state_dict()
    sd_b = _MODEL_B.state_dict()
    for key in sd_a:
        expected = (sd_a[key].float() + sd_b[key].float()) / 2.0
        assert torch.allclose(result[key].float(), expected, atol=1e-5), (
            f"Key {key}: uniform soup should be exact midpoint"
        )


# ---------------------------------------------------------------------------
# 4. test_weighted_soup_weights_sum_to_one
# ---------------------------------------------------------------------------

def test_weighted_soup_weights_sum_to_one():
    """Weights that don't sum to 1 should be normalised; zero weights raise."""
    # Non-unit weights normalised → same as [0.5, 0.5]
    result_unnorm = weighted_soup([_MODEL_A, _MODEL_B], weights=[2.0, 2.0])
    result_norm = weighted_soup([_MODEL_A, _MODEL_B], weights=[0.5, 0.5])
    for key in _MODEL_A.state_dict():
        assert torch.allclose(
            result_unnorm[key].float(), result_norm[key].float(), atol=1e-5
        ), f"Key {key}: normalised weights should match explicit 0.5/0.5"

    # All-zero weights should raise
    with pytest.raises((ValueError, ZeroDivisionError)):
        weighted_soup([_MODEL_A, _MODEL_B], weights=[0.0, 0.0])


# ---------------------------------------------------------------------------
# 5. test_weighted_soup_extreme_alpha
# ---------------------------------------------------------------------------

def test_weighted_soup_extreme_alpha():
    """alpha=1.0 on model_b → same as model_b's weights."""
    result = weighted_soup([_MODEL_A, _MODEL_B], weights=[0.0, 1.0])
    sd_b = _MODEL_B.state_dict()
    for key in sd_b:
        assert torch.allclose(result[key].float(), sd_b[key].float(), atol=1e-5), (
            f"Key {key}: weight=1 on model_b should return model_b"
        )


# ---------------------------------------------------------------------------
# 6. test_interpolate_models_alpha0
# ---------------------------------------------------------------------------

def test_interpolate_models_alpha0():
    result = interpolate_models(_MODEL_A, _MODEL_B, alpha=0.0)
    sd_a = _MODEL_A.state_dict()
    for key in sd_a:
        assert torch.allclose(result[key].float(), sd_a[key].float(), atol=1e-5), (
            f"Key {key}: alpha=0 should return model_a"
        )


# ---------------------------------------------------------------------------
# 7. test_interpolate_models_alpha1
# ---------------------------------------------------------------------------

def test_interpolate_models_alpha1():
    result = interpolate_models(_MODEL_A, _MODEL_B, alpha=1.0)
    sd_b = _MODEL_B.state_dict()
    for key in sd_b:
        assert torch.allclose(result[key].float(), sd_b[key].float(), atol=1e-5), (
            f"Key {key}: alpha=1 should return model_b"
        )


# ---------------------------------------------------------------------------
# 8. test_interpolate_models_midpoint
# ---------------------------------------------------------------------------

def test_interpolate_models_midpoint():
    result = interpolate_models(_MODEL_A, _MODEL_B, alpha=0.5)
    sd_a = _MODEL_A.state_dict()
    sd_b = _MODEL_B.state_dict()
    for key in sd_a:
        expected = (sd_a[key].float() + sd_b[key].float()) / 2.0
        assert torch.allclose(result[key].float(), expected, atol=1e-5), (
            f"Key {key}: alpha=0.5 should be exact midpoint"
        )


# ---------------------------------------------------------------------------
# 9. test_compute_weight_divergence_same
# ---------------------------------------------------------------------------

def test_compute_weight_divergence_same():
    """Identical models → cosine_sim ≈ 1.0, l2 ≈ 0."""
    div = compute_weight_divergence(_MODEL_A, _MODEL_A)
    assert div["mean_l2"] == pytest.approx(0.0, abs=1e-4)
    assert div["max_l2"] == pytest.approx(0.0, abs=1e-4)
    assert div["mean_cosine_sim"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 10. test_compute_weight_divergence_keys
# ---------------------------------------------------------------------------

def test_compute_weight_divergence_keys():
    div = compute_weight_divergence(_MODEL_A, _MODEL_B)
    assert "mean_l2" in div
    assert "max_l2" in div
    assert "mean_cosine_sim" in div


# ---------------------------------------------------------------------------
# 11. test_learned_soup_weights_sum_to_one
# ---------------------------------------------------------------------------

def test_learned_soup_weights_sum_to_one():
    mixer = LearnedSoupMixer(n_models=4)
    weights = mixer.get_weights()
    assert weights.shape == (4,)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 12. test_learned_soup_forward_shape
# ---------------------------------------------------------------------------

def test_learned_soup_forward_shape():
    B, T, V = 2, 8, 256
    mixer = LearnedSoupMixer(n_models=3)
    outputs = [torch.randn(B, T, V) for _ in range(3)]
    result = mixer(outputs)
    assert result.shape == (B, T, V)


# ---------------------------------------------------------------------------
# 13. test_ensemble_predict_shape
# ---------------------------------------------------------------------------

def test_ensemble_predict_shape():
    cfg = ModelSoupConfig()
    ensemble = ModelSoupEnsemble(_MODEL_A, cfg)
    B, T = 2, 8
    input_ids = torch.randint(0, TINY_CFG.vocab_size, (B, T))
    result = ensemble.ensemble_predict([_MODEL_A, _MODEL_B, _MODEL_C], input_ids)
    assert result.shape == (B, T, TINY_CFG.vocab_size)


# ---------------------------------------------------------------------------
# 14. test_greedy_soup_returns_state_dict
# ---------------------------------------------------------------------------

def test_greedy_soup_returns_state_dict():
    result = greedy_soup([_MODEL_A, _MODEL_B, _MODEL_C], eval_fn=_random_eval_fn)
    assert isinstance(result, dict)
    for key, val in result.items():
        assert isinstance(val, torch.Tensor), f"Key {key}: expected Tensor"


# ---------------------------------------------------------------------------
# 15. test_model_soup_create_uniform
# ---------------------------------------------------------------------------

def test_model_soup_create_uniform():
    cfg = ModelSoupConfig(method="uniform")
    ensemble = ModelSoupEnsemble(_MODEL_A, cfg)
    soup_model = ensemble.create_soup([_MODEL_A, _MODEL_B, _MODEL_C])
    assert isinstance(soup_model, torch.nn.Module)
    # Verify it has the same keys
    assert set(soup_model.state_dict().keys()) == set(_MODEL_A.state_dict().keys())


# ---------------------------------------------------------------------------
# 16. test_find_best_interpolation_returns_alpha
# ---------------------------------------------------------------------------

def test_find_best_interpolation_returns_alpha():
    cfg = ModelSoupConfig(eval_metric="loss")
    ensemble = ModelSoupEnsemble(_MODEL_A, cfg)
    best_alpha, best_score = ensemble.find_best_interpolation(
        _MODEL_A, _MODEL_B, eval_fn=_random_eval_fn, n_trials=5
    )
    assert 0.0 <= best_alpha <= 1.0, f"best_alpha {best_alpha} out of [0, 1]"
    assert isinstance(best_score, float)
