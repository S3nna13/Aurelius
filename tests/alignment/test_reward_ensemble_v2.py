"""Tests for aurelius.alignment.reward_ensemble_v2.

Covers:
 1.  RewardEnsemble forward output shape (B,)
 2.  predict returns (mean, std) both shape (B,)
 3.  std from predict is non-negative
 4.  aggregation='median' computes median
 5.  aggregation='min' computes min
 6.  _trimmed_mean trims correctly
 7.  agreement_score values in [0, 1]
 8.  RewardCalibrator fit then calibrate reduces RMSE vs uncalibrated
 9.  calibration_error is non-negative
10.  calibrate is a linear transformation
11.  RewardAgreementFilter.filter_by_agreement returns correct shapes
12.  high_confidence_pairs returns valid indices (within [0, B))
13.  Works with n_models=1 (no crash, correct shapes)
14.  aggregation='max' computes max
15.  trimmed_mean fallback when ratio trims everything
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from aurelius.alignment.reward_ensemble_v2 import (
    EnsembleConfig,
    RewardAgreementFilter,
    RewardCalibrator,
    RewardEnsemble,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
B = 4
K = 3

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LinearReward(nn.Module):
    """Simple linear reward model: w^T x → scalar per sample."""

    def __init__(self, d: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(d, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)  # (B,)


def make_ensemble(k: int = K, d: int = D_MODEL) -> RewardEnsemble:
    models = [LinearReward(d, seed=i) for i in range(k)]
    cfg = EnsembleConfig(n_models=k)
    return RewardEnsemble(models, cfg)


def make_input(b: int = B, d: int = D_MODEL) -> Tensor:
    torch.manual_seed(42)
    return torch.randn(b, d)


# ---------------------------------------------------------------------------
# 1. forward output shape
# ---------------------------------------------------------------------------


def test_forward_shape():
    """RewardEnsemble.forward must return shape (B,)."""
    ensemble = make_ensemble()
    x = make_input()
    with torch.no_grad():
        out = ensemble(x)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. predict returns (mean, std) both shape (B,)
# ---------------------------------------------------------------------------


def test_predict_shapes():
    """predict must return two tensors of shape (B,)."""
    ensemble = make_ensemble()
    x = make_input()
    with torch.no_grad():
        mean, std = ensemble.predict(x)
    assert mean.shape == (B,), f"mean shape {mean.shape}"
    assert std.shape == (B,), f"std shape {std.shape}"


# ---------------------------------------------------------------------------
# 3. std is non-negative
# ---------------------------------------------------------------------------


def test_predict_std_nonneg():
    """std returned by predict must be >= 0 everywhere."""
    ensemble = make_ensemble()
    x = make_input()
    with torch.no_grad():
        _, std = ensemble.predict(x)
    assert (std >= 0).all(), f"Negative std values: {std}"


# ---------------------------------------------------------------------------
# 4. aggregation='median'
# ---------------------------------------------------------------------------


def test_aggregation_median():
    """aggregation='median' must compute the column-wise median."""
    cfg = EnsembleConfig(n_models=3, aggregation="median")
    models = [LinearReward(D_MODEL, seed=i) for i in range(3)]
    ensemble = RewardEnsemble(models, cfg)

    # Override internal _stack_outputs with known values to verify
    known = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 6.0]])  # (3, 2)
    result = ensemble._aggregate(known)
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(result, expected, atol=1e-5), f"{result} != {expected}"


# ---------------------------------------------------------------------------
# 5. aggregation='min'
# ---------------------------------------------------------------------------


def test_aggregation_min():
    """aggregation='min' must return column-wise minimum."""
    cfg = EnsembleConfig(n_models=3, aggregation="min")
    models = [LinearReward(D_MODEL, seed=i) for i in range(3)]
    ensemble = RewardEnsemble(models, cfg)

    known = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 6.0]])  # (3, 2)
    result = ensemble._aggregate(known)
    expected = torch.tensor([1.0, 2.0])
    assert torch.allclose(result, expected, atol=1e-5), f"{result} != {expected}"


# ---------------------------------------------------------------------------
# 6. _trimmed_mean trims correctly
# ---------------------------------------------------------------------------


def test_trimmed_mean_correctness():
    """_trimmed_mean with ratio=1/3 should drop 1 from each end of K=5 rows."""
    ensemble = make_ensemble()

    # 5 models, 2 samples: sorted columns are [1,2,3,4,5] and [10,20,30,40,50]
    rewards = torch.tensor(
        [
            [3.0, 30.0],
            [1.0, 10.0],
            [5.0, 50.0],
            [2.0, 20.0],
            [4.0, 40.0],
        ]
    )  # (5, 2)

    # ratio=0.2 → n_trim=int(5*0.2)=1 → keep rows 1..3 → [2,3,4] and [20,30,40]
    result = ensemble._trimmed_mean(rewards, ratio=0.2)
    expected = torch.tensor([3.0, 30.0])
    assert torch.allclose(result, expected, atol=1e-5), f"{result} != {expected}"


# ---------------------------------------------------------------------------
# 7. agreement_score in [0, 1]
# ---------------------------------------------------------------------------


def test_agreement_score_range():
    """agreement_score must return values in [0, 1]."""
    ensemble = make_ensemble()
    x = make_input()
    with torch.no_grad():
        scores = ensemble.agreement_score(x)
    assert scores.shape == (B,), f"shape {scores.shape}"
    assert (scores >= 0.0).all() and (scores <= 1.0 + 1e-6).all(), (
        f"Out-of-range agreement scores: {scores}"
    )


# ---------------------------------------------------------------------------
# 8. RewardCalibrator fit+calibrate improves RMSE
# ---------------------------------------------------------------------------


def test_calibrator_improves_rmse():
    """Fitting calibrator on training data should reduce RMSE vs raw scores."""
    torch.manual_seed(7)
    N = 50
    # Human scores are a scaled/shifted version of raw scores
    raw = torch.randn(N)
    human = 2.5 * raw - 1.3

    cal = RewardCalibrator()
    cal.fit(raw, human)

    rmse_calibrated = cal.calibration_error(raw, human)
    rmse_raw = torch.sqrt(((raw - human) ** 2).mean()).item()

    assert rmse_calibrated < rmse_raw, (
        f"Calibrated RMSE ({rmse_calibrated:.4f}) should be < raw RMSE ({rmse_raw:.4f})"
    )


# ---------------------------------------------------------------------------
# 9. calibration_error is non-negative
# ---------------------------------------------------------------------------


def test_calibration_error_nonneg():
    """calibration_error must return a non-negative float."""
    cal = RewardCalibrator()
    torch.manual_seed(1)
    raw = torch.randn(20)
    targets = torch.randn(20)
    cal.fit(raw, targets)
    err = cal.calibration_error(raw, targets)
    assert err >= 0.0, f"Negative calibration error: {err}"
    assert isinstance(err, float), f"Expected float, got {type(err)}"


# ---------------------------------------------------------------------------
# 10. calibrate is linear
# ---------------------------------------------------------------------------


def test_calibrate_is_linear():
    """calibrate must apply a * raw + b, verifiable via known a,b."""
    cal = RewardCalibrator()
    torch.manual_seed(3)
    raw = torch.randn(30)
    human = 3.0 * raw + 0.5

    cal.fit(raw, human)
    calibrated = cal.calibrate(raw)
    expected = 3.0 * raw + 0.5

    # Fitted a,b should recover exact scaling for noiseless data
    assert torch.allclose(calibrated, expected, atol=1e-3), (
        f"Max deviation: {(calibrated - expected).abs().max():.5f}"
    )


# ---------------------------------------------------------------------------
# 11. filter_by_agreement shapes
# ---------------------------------------------------------------------------


def test_filter_by_agreement_shapes():
    """filter_by_agreement must return (kept_x, kept_mask) with correct shapes."""
    ensemble = make_ensemble()
    # Set a high threshold so most samples pass
    cfg = EnsembleConfig(n_models=K, uncertainty_threshold=1e6)
    filt = RewardAgreementFilter(cfg)
    x = make_input()

    kept_x, kept_mask = filt.filter_by_agreement(x, ensemble)

    assert kept_mask.shape == (B,), f"mask shape {kept_mask.shape}"
    assert kept_x.shape[1] == D_MODEL, f"kept_x dim1 {kept_x.shape[1]}"
    assert kept_x.shape[0] == kept_mask.sum().item(), (
        f"kept_x rows {kept_x.shape[0]} != mask sum {kept_mask.sum()}"
    )


# ---------------------------------------------------------------------------
# 12. high_confidence_pairs returns valid indices
# ---------------------------------------------------------------------------


def test_high_confidence_pairs_valid_indices():
    """high_confidence_pairs indices must be in [0, B)."""
    ensemble = make_ensemble()
    cfg = EnsembleConfig(n_models=K, uncertainty_threshold=1e6)
    filt = RewardAgreementFilter(cfg)

    torch.manual_seed(10)
    x_w = torch.randn(B, D_MODEL) + 2.0  # higher reward expected
    x_l = torch.randn(B, D_MODEL) - 2.0

    indices = filt.high_confidence_pairs(x_w, x_l, ensemble, margin=-1e9)
    assert indices.ndim == 1, f"Expected 1-D tensor, got shape {indices.shape}"
    if indices.numel() > 0:
        assert indices.min() >= 0, f"Negative index: {indices.min()}"
        assert indices.max() < B, f"Index out of range: {indices.max()}"


# ---------------------------------------------------------------------------
# 13. Works with n_models=1
# ---------------------------------------------------------------------------


def test_single_model():
    """RewardEnsemble with n_models=1 must not crash and return correct shapes."""
    models = [LinearReward(D_MODEL, seed=0)]
    cfg = EnsembleConfig(n_models=1)
    ensemble = RewardEnsemble(models, cfg)
    x = make_input()

    with torch.no_grad():
        out = ensemble(x)
        mean, std = ensemble.predict(x)

    assert out.shape == (B,)
    assert mean.shape == (B,)
    assert std.shape == (B,)
    assert (std == 0).all(), "std should be zero for single model"


# ---------------------------------------------------------------------------
# 14. aggregation='max'
# ---------------------------------------------------------------------------


def test_aggregation_max():
    """aggregation='max' must return the column-wise maximum."""
    cfg = EnsembleConfig(n_models=3, aggregation="max")
    models = [LinearReward(D_MODEL, seed=i) for i in range(3)]
    ensemble = RewardEnsemble(models, cfg)

    known = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 6.0]])  # (3, 2)
    result = ensemble._aggregate(known)
    expected = torch.tensor([5.0, 6.0])
    assert torch.allclose(result, expected, atol=1e-5), f"{result} != {expected}"


# ---------------------------------------------------------------------------
# 15. trimmed_mean fallback when ratio trims everything
# ---------------------------------------------------------------------------


def test_trimmed_mean_fallback():
    """When trimming would remove all values, _trimmed_mean falls back to mean."""
    ensemble = make_ensemble(k=2)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

    # ratio=0.6 → n_trim=1 → 2*1 >= 2, fallback to full mean
    result = ensemble._trimmed_mean(rewards, ratio=0.6)
    expected = rewards.mean(dim=0)
    assert torch.allclose(result, expected, atol=1e-5), f"{result} != {expected}"
