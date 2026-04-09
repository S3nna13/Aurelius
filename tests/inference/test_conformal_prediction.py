"""Tests for src/inference/conformal_prediction.py."""

import pytest
import torch

from src.inference.conformal_prediction import (
    ConformalConfig,
    ConformalPredictor,
    calibrate_threshold,
    compute_aps_scores,
    compute_lac_scores,
    compute_raps_scores,
    compute_softmax_scores,
    construct_prediction_set,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N, C = 100, 10

torch.manual_seed(42)
_logits = torch.randn(N, C)
_probs = compute_softmax_scores(_logits)
_labels = torch.randint(0, C, (N,))


# ---------------------------------------------------------------------------
# ConformalConfig
# ---------------------------------------------------------------------------


def test_conformal_config_defaults():
    cfg = ConformalConfig()
    assert cfg.alpha == 0.1
    assert cfg.method == "aps"
    assert cfg.raps_lambda == 0.01
    assert cfg.raps_k_reg == 5


# ---------------------------------------------------------------------------
# compute_softmax_scores
# ---------------------------------------------------------------------------


def test_softmax_scores_sum_to_one():
    probs = compute_softmax_scores(_logits)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(N), atol=1e-5), "Rows must sum to 1"


# ---------------------------------------------------------------------------
# compute_lac_scores
# ---------------------------------------------------------------------------


def test_lac_scores_shape():
    scores = compute_lac_scores(_probs, _labels)
    assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"


def test_lac_scores_low_for_high_confidence():
    """If a class has near-1 probability the LAC score should be near 0."""
    probs = torch.zeros(1, C)
    probs[0, 3] = 1.0
    labels = torch.tensor([3])
    score = compute_lac_scores(probs, labels)
    assert score.item() < 1e-5, f"Expected ~0, got {score.item()}"


# ---------------------------------------------------------------------------
# compute_aps_scores
# ---------------------------------------------------------------------------


def test_aps_scores_shape():
    scores = compute_aps_scores(_probs, _labels)
    assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"


def test_aps_scores_in_valid_range():
    """APS scores are cumulative probs so must be in (0, 1]."""
    scores = compute_aps_scores(_probs, _labels)
    assert (scores > 0).all(), "APS scores should be > 0"
    assert (scores <= 1.0 + 1e-5).all(), "APS scores should be <= 1"


# ---------------------------------------------------------------------------
# compute_raps_scores
# ---------------------------------------------------------------------------


def test_raps_scores_shape():
    scores = compute_raps_scores(_probs, _labels, lambda_reg=0.01, k_reg=5)
    assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"


def test_raps_scores_geq_aps_scores():
    """RAPS adds a non-negative regularisation term so must be >= APS."""
    aps = compute_aps_scores(_probs, _labels)
    raps = compute_raps_scores(_probs, _labels, lambda_reg=0.01, k_reg=5)
    assert (raps >= aps - 1e-6).all(), "RAPS scores must be >= APS scores"


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------


def test_calibrate_threshold_returns_float():
    scores = compute_aps_scores(_probs, _labels)
    threshold = calibrate_threshold(scores, alpha=0.1)
    assert isinstance(threshold, float), f"Expected float, got {type(threshold)}"


def test_calibrate_threshold_in_score_range():
    """Threshold should lie within [min_score, max_score]."""
    scores = compute_aps_scores(_probs, _labels)
    threshold = calibrate_threshold(scores, alpha=0.1)
    assert scores.min().item() <= threshold <= scores.max().item() + 1e-6


# ---------------------------------------------------------------------------
# construct_prediction_set
# ---------------------------------------------------------------------------


def test_construct_prediction_set_returns_list_of_lists():
    threshold = 0.5
    result = construct_prediction_set(_probs, threshold, method="aps")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == N, f"Should have {N} prediction sets"
    assert all(isinstance(s, list) for s in result), "Each element should be a list"


def test_construct_prediction_set_high_threshold_includes_all():
    """With threshold >= 1.0 (APS), every class should be included."""
    result = construct_prediction_set(_probs, threshold=1.0, method="aps")
    for i, pred_set in enumerate(result):
        assert len(pred_set) == C, (
            f"Sample {i}: expected {C} classes, got {len(pred_set)}"
        )


# ---------------------------------------------------------------------------
# ConformalPredictor end-to-end
# ---------------------------------------------------------------------------


def test_conformal_predictor_calibrate_and_predict():
    cfg = ConformalConfig(alpha=0.1, method="aps")
    predictor = ConformalPredictor(cfg)

    threshold = predictor.calibrate(_probs, _labels)
    assert isinstance(threshold, float), "calibrate() should return a float"
    assert predictor.threshold == threshold

    test_logits = torch.randn(20, C)
    test_probs = compute_softmax_scores(test_logits)
    prediction_sets = predictor.predict(test_probs)

    assert isinstance(prediction_sets, list)
    assert len(prediction_sets) == 20
    assert all(isinstance(s, list) for s in prediction_sets)


def test_conformal_predictor_coverage_rate_in_range():
    cfg = ConformalConfig(alpha=0.1, method="aps")
    predictor = ConformalPredictor(cfg)
    predictor.calibrate(_probs, _labels)

    test_logits = torch.randn(50, C)
    test_probs = compute_softmax_scores(test_logits)
    test_labels = torch.randint(0, C, (50,))

    prediction_sets = predictor.predict(test_probs)
    rate = predictor.coverage_rate(prediction_sets, test_labels)

    assert isinstance(rate, float), "coverage_rate() should return a float"
    assert 0.0 <= rate <= 1.0, f"Coverage rate must be in [0, 1], got {rate}"


# ---------------------------------------------------------------------------
# Extra: test all three methods work without error
# ---------------------------------------------------------------------------


def test_all_methods_end_to_end():
    for method in ("aps", "raps", "lac"):
        cfg = ConformalConfig(alpha=0.1, method=method)
        predictor = ConformalPredictor(cfg)
        threshold = predictor.calibrate(_probs, _labels)
        assert isinstance(threshold, float), f"{method}: threshold should be float"

        test_probs = compute_softmax_scores(torch.randn(10, C))
        sets = predictor.predict(test_probs)
        assert len(sets) == 10, f"{method}: should produce 10 prediction sets"
