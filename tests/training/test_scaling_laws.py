"""
Tests for src/training/scaling_laws.py — Chinchilla / neural scaling law utilities.
"""

import math

from src.training.scaling_laws import (
    ChinchillaPredictor,
    ComputeOptimalScheduler,
    compute_training_budget,
    kaplan_scaling_law,
    recommended_model_size,
)

# ---------------------------------------------------------------------------
# ChinchillaPredictor.loss
# ---------------------------------------------------------------------------


def test_chinchilla_loss_decreases_with_more_params():
    predictor = ChinchillaPredictor()
    n = 1e9
    d = 2e10
    assert predictor.loss(2 * n, d) < predictor.loss(n, d)


def test_chinchilla_loss_decreases_with_more_data():
    predictor = ChinchillaPredictor()
    n = 1e9
    d = 2e10
    assert predictor.loss(n, 2 * d) < predictor.loss(n, d)


# ---------------------------------------------------------------------------
# ChinchillaPredictor.optimal_allocation
# ---------------------------------------------------------------------------


def test_optimal_allocation_keys():
    predictor = ChinchillaPredictor()
    result = predictor.optimal_allocation(1e21)
    for key in ("n_params", "n_tokens", "predicted_loss", "flops"):
        assert key in result, f"Missing key: {key}"


def test_optimal_allocation_respects_budget():
    predictor = ChinchillaPredictor()
    flop_budget = 1e21
    result = predictor.optimal_allocation(flop_budget)
    implied_flops = 6 * result["n_params"] * result["n_tokens"]
    assert math.isclose(implied_flops, flop_budget, rel_tol=0.10), (
        f"6*N*D={implied_flops:.3e} differs from budget {flop_budget:.3e} by >10%"
    )


# ---------------------------------------------------------------------------
# ChinchillaPredictor.isoflop_curve
# ---------------------------------------------------------------------------


def test_isoflop_curve_length():
    predictor = ChinchillaPredictor()
    curve = predictor.isoflop_curve(1e21, n_points=20)
    assert len(curve) == 20


def test_isoflop_curve_all_same_flops():
    predictor = ChinchillaPredictor()
    flop_budget = 1e21
    curve = predictor.isoflop_curve(flop_budget, n_points=15)
    for point in curve:
        implied = 6 * point["n_params"] * point["n_tokens"]
        assert math.isclose(implied, flop_budget, rel_tol=0.01), (
            f"Point flops {implied:.3e} != budget {flop_budget:.3e}"
        )


# ---------------------------------------------------------------------------
# ChinchillaPredictor.extrapolate_loss
# ---------------------------------------------------------------------------


def test_extrapolate_loss_fits_observation():
    predictor = ChinchillaPredictor()
    n, d = 1e9, 2e10
    observed_loss = 2.5  # arbitrary target loss (may differ from formula)

    # Extrapolating at the same (N, D) should return the observed loss exactly
    result = predictor.extrapolate_loss(
        observed_n=n,
        observed_d=d,
        observed_loss=observed_loss,
        target_n=n,
        target_d=d,
    )
    assert math.isclose(result, observed_loss, rel_tol=1e-9), (
        f"Expected {observed_loss}, got {result}"
    )


# ---------------------------------------------------------------------------
# compute_training_budget
# ---------------------------------------------------------------------------


def test_compute_training_budget_positive():
    result = compute_training_budget(
        gpu_flops=312e12,
        n_gpus=8,
        training_days=7,
        efficiency=0.4,
    )
    assert isinstance(result, float)
    assert result > 0


# ---------------------------------------------------------------------------
# kaplan_scaling_law
# ---------------------------------------------------------------------------


def test_kaplan_scaling_law_positive():
    result = kaplan_scaling_law(n_params=1e9, n_tokens=1e10)
    assert isinstance(result, float)
    assert result > 0


# ---------------------------------------------------------------------------
# recommended_model_size
# ---------------------------------------------------------------------------


def test_recommended_model_size_architecture():
    result = recommended_model_size(1e21)
    d_model = result["d_model"]
    n_heads = result["n_heads"]
    assert n_heads >= 1
    assert d_model % n_heads == 0, f"d_model={d_model} is not divisible by n_heads={n_heads}"


# ---------------------------------------------------------------------------
# ComputeOptimalScheduler.should_scale_up
# ---------------------------------------------------------------------------


def test_should_scale_up_returns_dict():
    scheduler = ComputeOptimalScheduler(
        initial_n_params=1e9,
        total_flop_budget=1e21,
    )
    result = scheduler.should_scale_up(
        current_n_params=1e9,
        flops_used_so_far=2e20,
        current_loss=2.8,
    )
    for key in ("scale_up", "optimal_n", "expected_gain"):
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["scale_up"], bool)
    assert isinstance(result["optimal_n"], float)
    assert isinstance(result["expected_gain"], float)
