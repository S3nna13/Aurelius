"""Tests for model_soup.py: spec-compliant state-dict-based soup API.

Covers SoupConfig, uniform_soup, greedy_soup, learned_soup,
interpolate_models, ModelSoup, compute_weight_distance, and get_soup_stats.

All tests use tiny nn.Linear(4, 4) models to keep tensors small.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.model_soup_v2 import (
    ModelSoup,
    SoupConfig,
    compute_weight_distance,
    greedy_soup,
    interpolate_models,
    learned_soup,
    uniform_soup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_state_dict(seed: int) -> dict:
    """Return a state dict from a freshly initialised nn.Linear(4, 4)."""
    torch.manual_seed(seed)
    return nn.Linear(4, 4).state_dict()


# Pre-build a few state dicts for reuse
_SD_A = _make_linear_state_dict(0)
_SD_B = _make_linear_state_dict(1)
_SD_C = _make_linear_state_dict(2)
_SD_D = _make_linear_state_dict(3)


def _dummy_eval_fn(state_dict: dict) -> float:
    """Deterministic eval: negative mean-abs of the weight tensor."""
    return -float(state_dict["weight"].float().abs().mean().item())


# ---------------------------------------------------------------------------
# 1. SoupConfig defaults
# ---------------------------------------------------------------------------


def test_soup_config_defaults():
    """SoupConfig should have correct defaults."""
    cfg = SoupConfig()
    assert cfg.method == "uniform"
    assert cfg.max_models == 10


# ---------------------------------------------------------------------------
# 2. uniform_soup — output keys match input
# ---------------------------------------------------------------------------


def test_uniform_soup_keys_match():
    """uniform_soup output should have the same keys as input state dicts."""
    result = uniform_soup([_SD_A, _SD_B])
    assert set(result.keys()) == set(_SD_A.keys())


# ---------------------------------------------------------------------------
# 3. uniform_soup — identical models return same weights
# ---------------------------------------------------------------------------


def test_uniform_soup_identical_models_unchanged():
    """uniform_soup of identical state dicts should return the same weights."""
    sd_copy = {k: v.clone() for k, v in _SD_A.items()}
    result = uniform_soup([_SD_A, sd_copy])
    for key in _SD_A:
        assert torch.allclose(result[key].float(), _SD_A[key].float(), atol=1e-6), (
            f"Key {key}: uniform soup of identical models should be unchanged"
        )


# ---------------------------------------------------------------------------
# 4. learned_soup with uniform weights equals uniform_soup
# ---------------------------------------------------------------------------


def test_learned_soup_uniform_weights_equals_uniform_soup():
    """learned_soup with uniform weights must give the same result as uniform_soup."""
    n = 3
    sds = [_SD_A, _SD_B, _SD_C]
    uniform_weights = torch.ones(n) / n
    result_learned = learned_soup(sds, weights=uniform_weights)
    result_uniform = uniform_soup(sds)
    for key in result_uniform:
        assert torch.allclose(
            result_learned[key].float(), result_uniform[key].float(), atol=1e-5
        ), f"Key {key}: learned_soup with uniform weights should match uniform_soup"


# ---------------------------------------------------------------------------
# 5. interpolate_models — alpha=0 returns A
# ---------------------------------------------------------------------------


def test_interpolate_alpha_0_returns_a():
    """interpolate_models at alpha=0 should equal state_dict_a."""
    result = interpolate_models(_SD_A, _SD_B, alpha=0.0)
    for key in _SD_A:
        assert torch.allclose(result[key].float(), _SD_A[key].float(), atol=1e-6), (
            f"Key {key}: alpha=0 should return A"
        )


# ---------------------------------------------------------------------------
# 6. interpolate_models — alpha=1 returns B
# ---------------------------------------------------------------------------


def test_interpolate_alpha_1_returns_b():
    """interpolate_models at alpha=1 should equal state_dict_b."""
    result = interpolate_models(_SD_A, _SD_B, alpha=1.0)
    for key in _SD_B:
        assert torch.allclose(result[key].float(), _SD_B[key].float(), atol=1e-6), (
            f"Key {key}: alpha=1 should return B"
        )


# ---------------------------------------------------------------------------
# 7. interpolate_models — alpha=0.5 is exact midpoint
# ---------------------------------------------------------------------------


def test_interpolate_alpha_half_is_midpoint():
    """interpolate_models at alpha=0.5 should be the exact midpoint of A and B."""
    result = interpolate_models(_SD_A, _SD_B, alpha=0.5)
    for key in _SD_A:
        expected = (_SD_A[key].float() + _SD_B[key].float()) * 0.5
        assert torch.allclose(result[key].float(), expected, atol=1e-6), (
            f"Key {key}: alpha=0.5 should be midpoint"
        )


# ---------------------------------------------------------------------------
# 8. greedy_soup — soup score never worse than first model alone
# ---------------------------------------------------------------------------


def test_greedy_soup_score_non_decreasing():
    """greedy_soup result should have an eval score >= the initial model's score."""
    initial_score = _dummy_eval_fn(_SD_A)
    result = greedy_soup([_SD_A, _SD_B, _SD_C, _SD_D], eval_fn=_dummy_eval_fn)
    final_score = _dummy_eval_fn(result)
    # The greedy algorithm never makes things worse (higher = better by default)
    assert final_score >= initial_score - 1e-6, (
        f"greedy_soup final score {final_score} worse than initial {initial_score}"
    )


# ---------------------------------------------------------------------------
# 9. ModelSoup.add_checkpoint — stores up to max_models
# ---------------------------------------------------------------------------


def test_model_soup_add_checkpoint_up_to_max():
    """ModelSoup should store at most max_models checkpoints (FIFO eviction)."""
    cfg = SoupConfig(method="uniform", max_models=3)
    soup = ModelSoup(cfg)
    for i in range(5):
        soup.add_checkpoint(_make_linear_state_dict(i))
    assert len(soup._checkpoints) == 3, f"Expected 3 checkpoints, got {len(soup._checkpoints)}"


# ---------------------------------------------------------------------------
# 10. ModelSoup.cook — uniform method returns correct keys
# ---------------------------------------------------------------------------


def test_model_soup_cook_uniform_returns_correct_keys():
    """ModelSoup.cook with uniform method should return state dict with correct keys."""
    cfg = SoupConfig(method="uniform")
    soup = ModelSoup(cfg)
    soup.add_checkpoint(_SD_A)
    soup.add_checkpoint(_SD_B)
    result = soup.cook()
    assert set(result.keys()) == set(_SD_A.keys())


# ---------------------------------------------------------------------------
# 11. ModelSoup.cook — greedy method requires eval_fn
# ---------------------------------------------------------------------------


def test_model_soup_cook_greedy_requires_eval_fn():
    """ModelSoup.cook with method='greedy' and no eval_fn should raise ValueError."""
    cfg = SoupConfig(method="greedy")
    soup = ModelSoup(cfg)
    soup.add_checkpoint(_SD_A)
    with pytest.raises(ValueError, match="eval_fn"):
        soup.cook(eval_fn=None)


# ---------------------------------------------------------------------------
# 12. ModelSoup.cook — greedy method returns correct result
# ---------------------------------------------------------------------------


def test_model_soup_cook_greedy_method():
    """ModelSoup.cook with method='greedy' and eval_fn should return valid state dict."""
    cfg = SoupConfig(method="greedy")
    soup = ModelSoup(cfg)
    soup.add_checkpoint(_SD_A)
    soup.add_checkpoint(_SD_B)
    soup.add_checkpoint(_SD_C)
    result = soup.cook(eval_fn=_dummy_eval_fn)
    assert set(result.keys()) == set(_SD_A.keys())
    for key in result:
        assert result[key].shape == _SD_A[key].shape


# ---------------------------------------------------------------------------
# 13. get_soup_stats — returns correct keys
# ---------------------------------------------------------------------------


def test_get_soup_stats_returns_correct_keys():
    """get_soup_stats should return dict with mean_l2_distance, max_l2_distance, n_params."""
    cfg = SoupConfig(method="uniform")
    soup = ModelSoup(cfg)
    soup.add_checkpoint(_SD_A)
    soup.add_checkpoint(_SD_B)
    result = soup.cook()
    stats = soup.get_soup_stats(_SD_A, result)
    assert "mean_l2_distance" in stats, "Missing key: mean_l2_distance"
    assert "max_l2_distance" in stats, "Missing key: max_l2_distance"
    assert "n_params" in stats, "Missing key: n_params"


# ---------------------------------------------------------------------------
# 14. compute_weight_distance — identical state dicts return 0
# ---------------------------------------------------------------------------


def test_compute_weight_distance_identical_is_zero():
    """compute_weight_distance with identical state dicts should return 0."""
    sd_copy = {k: v.clone() for k, v in _SD_A.items()}
    dist = compute_weight_distance(_SD_A, sd_copy)
    assert dist == pytest.approx(0.0, abs=1e-6), f"Expected 0.0 for identical models, got {dist}"


# ---------------------------------------------------------------------------
# 15. ModelSoup truncates at max_models (FIFO order check)
# ---------------------------------------------------------------------------


def test_model_soup_truncates_fifo():
    """ModelSoup should evict the oldest checkpoint when max_models is reached."""
    cfg = SoupConfig(method="uniform", max_models=2)
    soup = ModelSoup(cfg)
    sd0 = _make_linear_state_dict(10)
    sd1 = _make_linear_state_dict(11)
    sd2 = _make_linear_state_dict(12)
    soup.add_checkpoint(sd0)
    soup.add_checkpoint(sd1)
    # Adding sd2 should evict sd0
    soup.add_checkpoint(sd2)
    assert len(soup._checkpoints) == 2
    # The remaining checkpoints should be sd1 and sd2 (not sd0)
    for key in sd0:
        # sd0 should NOT be in the pool — check neither checkpoint equals sd0
        for cp in soup._checkpoints:
            assert (
                not torch.allclose(cp[key].float(), sd0[key].float(), atol=1e-8)
                or torch.allclose(sd1[key].float(), sd0[key].float(), atol=1e-8)
                or torch.allclose(sd2[key].float(), sd0[key].float(), atol=1e-8)
            ), f"Key {key}: evicted checkpoint still present"
        break  # checking one key is sufficient for the structure test


# ---------------------------------------------------------------------------
# 16. learned_soup with None weights equals uniform_soup
# ---------------------------------------------------------------------------


def test_learned_soup_none_weights_equals_uniform():
    """learned_soup with weights=None should produce the same result as uniform_soup."""
    sds = [_SD_A, _SD_B]
    result_learned = learned_soup(sds, weights=None)
    result_uniform = uniform_soup(sds)
    for key in result_uniform:
        assert torch.allclose(
            result_learned[key].float(), result_uniform[key].float(), atol=1e-6
        ), f"Key {key}: learned_soup(None) should match uniform_soup"


# ---------------------------------------------------------------------------
# 17. compute_weight_distance — different models return positive value
# ---------------------------------------------------------------------------


def test_compute_weight_distance_different_models_positive():
    """compute_weight_distance with different models should return a positive float."""
    dist = compute_weight_distance(_SD_A, _SD_B)
    assert isinstance(dist, float)
    assert dist > 0.0, f"Expected positive distance, got {dist}"


# ---------------------------------------------------------------------------
# 18. uniform_soup with 3 models — shape preserved
# ---------------------------------------------------------------------------


def test_uniform_soup_three_models_shape_preserved():
    """uniform_soup with 3 models should preserve all parameter shapes."""
    result = uniform_soup([_SD_A, _SD_B, _SD_C])
    for key in _SD_A:
        assert result[key].shape == _SD_A[key].shape, (
            f"Key {key}: shape mismatch in uniform_soup output"
        )


# ---------------------------------------------------------------------------
# 19. get_soup_stats — n_params is correct for nn.Linear(4, 4)
# ---------------------------------------------------------------------------


def test_get_soup_stats_n_params_correct():
    """get_soup_stats n_params should equal total scalar parameters in nn.Linear(4,4)."""
    # nn.Linear(4, 4): weight (4x4=16) + bias (4) = 20 params
    cfg = SoupConfig(method="uniform")
    soup = ModelSoup(cfg)
    soup.add_checkpoint(_SD_A)
    result = soup.cook()
    stats = soup.get_soup_stats(_SD_A, result)
    assert stats["n_params"] == pytest.approx(20.0), (
        f"Expected 20 params for Linear(4,4), got {stats['n_params']}"
    )


# ---------------------------------------------------------------------------
# 20. greedy_soup — higher_is_better=False (lower is better)
# ---------------------------------------------------------------------------


def test_greedy_soup_lower_is_better():
    """greedy_soup with higher_is_better=False should seek the lower eval score."""

    # eval_fn returning positive L1 mean (lower = better in this mode)
    def _lower_better(sd: dict) -> float:
        return float(sd["weight"].float().abs().mean().item())

    initial_score = _lower_better(_SD_A)
    result = greedy_soup([_SD_A, _SD_B, _SD_C], eval_fn=_lower_better, higher_is_better=False)
    final_score = _lower_better(result)
    # Score should be <= initial (greedy never makes things strictly worse)
    assert final_score <= initial_score + 1e-6, (
        f"greedy_soup (lower_is_better) final score {final_score} > initial {initial_score}"
    )
