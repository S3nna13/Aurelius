"""Tests for src/eval/platt_scaling.py — pure PyTorch, no scipy/sklearn.

Test inventory (15 tests):
 1.  TemperatureScaler: T > 0 after fitting
 2.  TemperatureScaler: calibrated probs sum to 1 (softmax output)
 3.  TemperatureScaler: T=1 is identity (calibrate ≈ softmax(logits))
 4.  TemperatureScaler: perfectly calibrated data → T near 1.0
 5.  TemperatureScaler: overconfident model → T > 1.0 after calibration
 6.  PlattScaler: output in (0, 1) for any finite input
 7.  PlattScaler: monotonically increasing in input scores
 8.  IsotonicCalibrator: output is non-decreasing (PAVA guarantee)
 9.  IsotonicCalibrator: output in [0, 1] when labels are 0/1
10.  ECE: in [0, 1]
11.  ECE: ~0.0 for perfectly calibrated model
12.  ECE: high for maximally miscalibrated model
13.  Determinism: repeated fit gives identical results
14.  All calibrators: output shape matches input shape
15.  No NaN/Inf on normal inputs
"""

from __future__ import annotations

import torch
import pytest

from src.eval.platt_scaling import (
    TemperatureScaler,
    PlattScaler,
    IsotonicCalibrator,
    CalibrationMetrics,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N = 64
C = 8


def _make_logits(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, C)


def _make_binary(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scores, labels) for binary calibration tasks."""
    torch.manual_seed(seed)
    scores = torch.randn(N)
    labels = (torch.rand(N) > 0.5).float()
    return scores, labels


def _softmax_labels(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax labels for a logit tensor."""
    return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# 1. TemperatureScaler: T > 0 after fitting
# ---------------------------------------------------------------------------


def test_temperature_scaler_positive_temperature():
    logits = _make_logits(0)
    labels = _softmax_labels(logits)
    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.temperature > 0.0, f"T must be > 0, got {scaler.temperature}"


# ---------------------------------------------------------------------------
# 2. TemperatureScaler: calibrated probs sum to 1
# ---------------------------------------------------------------------------


def test_temperature_scaler_probs_sum_to_one():
    logits = _make_logits(1)
    labels = _softmax_labels(logits)
    scaler = TemperatureScaler().fit(logits, labels)
    probs = scaler.calibrate(logits)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), (
        f"Row sums should be 1.0, max deviation: {(row_sums - 1.0).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 3. TemperatureScaler: T=1 is identity
# ---------------------------------------------------------------------------


def test_temperature_scaler_t1_is_identity():
    """With T manually set to 1.0, calibrate should equal softmax(logits)."""
    logits = _make_logits(2)
    scaler = TemperatureScaler()
    # Do not fit — default T is 1.0.
    probs_calibrated = scaler.calibrate(logits)
    probs_softmax = torch.softmax(logits.float(), dim=-1)
    assert torch.allclose(probs_calibrated, probs_softmax, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. TemperatureScaler: perfectly calibrated data → T near 1.0
# ---------------------------------------------------------------------------


def test_temperature_scaler_near_one_when_well_calibrated():
    """Logits generated as scaled = logits / T_true should recover T_true after fit."""
    torch.manual_seed(42)
    T_true = 1.0
    n = 400
    # Generate logits whose "true" temperature is 1.0: sample labels from the
    # softmax distribution so that the model is in calibration at T=1.
    raw = torch.randn(n, C)
    probs_true = torch.softmax(raw / T_true, dim=-1)
    # Sample labels from the true distribution (not just argmax).
    labels = torch.distributions.Categorical(probs=probs_true).sample()
    scaler = TemperatureScaler().fit(raw, labels, lr=0.02, n_iters=1000)
    # The fitted T should be in [0.5, 2.0] for data generated at T=1.
    assert 0.5 < scaler.temperature < 2.0, (
        f"Expected T near 1.0 for well-calibrated data, got {scaler.temperature}"
    )


# ---------------------------------------------------------------------------
# 5. TemperatureScaler: overconfident model → T > 1.0 after calibration
# ---------------------------------------------------------------------------


def test_temperature_scaler_overconfident_gives_t_gt_one():
    """Highly peaked logits that mismatch labels → T > 1 to soften predictions."""
    torch.manual_seed(7)
    n = 200
    # Create logits where the argmax class is mostly wrong (overconfident, wrong class).
    logits = torch.zeros(n, C)
    logits[:, 0] = 10.0  # very confident about class 0
    # Labels are uniformly random (model is mostly wrong and overconfident).
    labels = torch.randint(1, C, (n,))  # never class 0
    scaler = TemperatureScaler().fit(logits, labels, lr=0.05, n_iters=500)
    assert scaler.temperature > 1.0, (
        f"Overconfident model should get T > 1.0, got {scaler.temperature}"
    )


# ---------------------------------------------------------------------------
# 6. PlattScaler: output in (0, 1) for any finite input
# ---------------------------------------------------------------------------


def test_platt_scaler_output_in_open_unit_interval():
    scores, labels = _make_binary(3)
    scaler = PlattScaler().fit(scores, labels)
    probs = scaler.calibrate(scores)
    assert (probs > 0.0).all(), "Platt outputs must be > 0"
    assert (probs < 1.0).all(), "Platt outputs must be < 1"


# ---------------------------------------------------------------------------
# 7. PlattScaler: monotonically increasing in input scores
# ---------------------------------------------------------------------------


def test_platt_scaler_monotonic():
    """σ(A*f + B) is monotone if A > 0 and scores are sorted."""
    scores, labels = _make_binary(4)
    scaler = PlattScaler().fit(scores, labels)
    test_scores = torch.linspace(-5.0, 5.0, 100)
    probs = scaler.calibrate(test_scores)
    diffs = probs[1:] - probs[:-1]
    # Because A can be negative (if labels are flipped), we check for
    # strict monotonicity in either direction.
    assert (diffs >= -1e-6).all() or (diffs <= 1e-6).all(), (
        "PlattScaler output must be monotone in input scores"
    )


# ---------------------------------------------------------------------------
# 8. IsotonicCalibrator: output is non-decreasing (PAVA guarantee)
# ---------------------------------------------------------------------------


def test_isotonic_calibrator_nondecreasing():
    scores, labels = _make_binary(5)
    cal = IsotonicCalibrator().fit(scores, labels)
    test_scores = torch.sort(scores).values
    probs = cal.calibrate(test_scores)
    diffs = probs[1:] - probs[:-1]
    assert (diffs >= -1e-6).all(), (
        f"Isotonic output must be non-decreasing; found min diff {diffs.min().item()}"
    )


# ---------------------------------------------------------------------------
# 9. IsotonicCalibrator: output in [0, 1] when labels are 0/1
# ---------------------------------------------------------------------------


def test_isotonic_calibrator_output_in_unit_interval():
    scores, labels = _make_binary(6)
    cal = IsotonicCalibrator().fit(scores, labels)
    probs = cal.calibrate(scores)
    assert (probs >= 0.0).all(), "Isotonic probs must be >= 0"
    assert (probs <= 1.0).all(), "Isotonic probs must be <= 1"


# ---------------------------------------------------------------------------
# 10. ECE: in [0, 1]
# ---------------------------------------------------------------------------


def test_ece_in_unit_interval():
    torch.manual_seed(8)
    probs = torch.rand(N)
    labels = (torch.rand(N) > 0.5).float()
    ece = CalibrationMetrics.ece(probs, labels)
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0, 1], got {ece}"


# ---------------------------------------------------------------------------
# 11. ECE: ~0.0 for perfectly calibrated model
# ---------------------------------------------------------------------------


def test_ece_zero_for_perfectly_calibrated():
    """If every bin has acc == conf, ECE should be exactly 0."""
    n_bins = 10
    # Build synthetic data where each sample's confidence equals the label
    # directly — i.e. the model is perfectly calibrated.
    # Use bin centres as both confidence and "accuracy".
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    # Repeat each centre many times for stable bin statistics.
    reps = 20
    probs = centers.repeat_interleave(reps)
    # Labels drawn to match the confidence: for centre c, flip a Bernoulli(c).
    torch.manual_seed(99)
    labels = torch.bernoulli(probs)
    # With enough samples the ECE should be close to 0.
    ece = CalibrationMetrics.ece(probs, labels, n_bins=n_bins)
    assert ece < 0.15, f"Expected ECE ≈ 0 for well-calibrated data, got {ece}"


# ---------------------------------------------------------------------------
# 12. ECE: high for maximally miscalibrated model
# ---------------------------------------------------------------------------


def test_ece_high_for_miscalibrated():
    """Predicting 1.0 confidence when always wrong → high ECE."""
    n = 100
    probs = torch.ones(n)           # model says 100 % confident
    labels = torch.zeros(n)         # but is always wrong
    ece = CalibrationMetrics.ece(probs, labels)
    assert ece > 0.5, f"Expected high ECE for miscalibrated model, got {ece}"


# ---------------------------------------------------------------------------
# 13. Determinism: repeated fit gives identical results
# ---------------------------------------------------------------------------


def test_determinism_temperature_scaler():
    logits = _make_logits(10)
    labels = _softmax_labels(logits)
    t1 = TemperatureScaler().fit(logits, labels, lr=0.01, n_iters=100).temperature
    t2 = TemperatureScaler().fit(logits, labels, lr=0.01, n_iters=100).temperature
    assert t1 == t2, f"TemperatureScaler must be deterministic: {t1} vs {t2}"


def test_determinism_platt_scaler():
    scores, labels = _make_binary(11)
    s1 = PlattScaler().fit(scores, labels, lr=0.01, n_iters=100)
    s2 = PlattScaler().fit(scores, labels, lr=0.01, n_iters=100)
    assert s1._A == s2._A and s1._B == s2._B


def test_determinism_isotonic():
    scores, labels = _make_binary(12)
    c1 = IsotonicCalibrator().fit(scores, labels)
    c2 = IsotonicCalibrator().fit(scores, labels)
    assert torch.equal(c1._y_fit, c2._y_fit)


# ---------------------------------------------------------------------------
# 14. Output shape matches input shape
# ---------------------------------------------------------------------------


def test_output_shapes():
    logits = _make_logits(13)
    labels = _softmax_labels(logits)
    scores_1d = logits[:, 0]
    bin_labels = (torch.rand(N) > 0.5).float()

    ts = TemperatureScaler().fit(logits, labels)
    assert ts.calibrate(logits).shape == (N, C)

    ps = PlattScaler().fit(scores_1d, bin_labels)
    assert ps.calibrate(scores_1d).shape == (N,)

    ic = IsotonicCalibrator().fit(scores_1d, bin_labels)
    assert ic.calibrate(scores_1d).shape == (N,)


# ---------------------------------------------------------------------------
# 15. No NaN/Inf on normal inputs
# ---------------------------------------------------------------------------


def test_no_nan_inf_temperature_scaler():
    logits = _make_logits(14)
    labels = _softmax_labels(logits)
    scaler = TemperatureScaler().fit(logits, labels)
    probs = scaler.calibrate(logits)
    assert not torch.isnan(probs).any(), "NaN in TemperatureScaler output"
    assert not torch.isinf(probs).any(), "Inf in TemperatureScaler output"


def test_no_nan_inf_platt_scaler():
    scores, labels = _make_binary(15)
    scaler = PlattScaler().fit(scores, labels)
    probs = scaler.calibrate(scores)
    assert not torch.isnan(probs).any(), "NaN in PlattScaler output"
    assert not torch.isinf(probs).any(), "Inf in PlattScaler output"


def test_no_nan_inf_isotonic():
    scores, labels = _make_binary(16)
    cal = IsotonicCalibrator().fit(scores, labels)
    probs = cal.calibrate(scores)
    assert not torch.isnan(probs).any(), "NaN in IsotonicCalibrator output"
    assert not torch.isinf(probs).any(), "Inf in IsotonicCalibrator output"
