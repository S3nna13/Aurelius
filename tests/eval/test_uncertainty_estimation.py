"""
Tests for src/eval/uncertainty_estimation.py

15 tests using tiny configs:
  d_model=16, vocab=16, seq_len=8, batch=2, n_samples=5, n_models=3
Model: nn.Embedding(16,16) + nn.Dropout(0.1) + nn.Linear(16,16)
Every test runs a real forward (and where relevant backward) pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.uncertainty_estimation import (
    DeepEnsemble,
    EntropyThresholder,
    MCDropoutEstimator,
    TemperatureCalibration,
    UncertaintyBenchmark,
)

# -------------------------------------------------------------------------
# Tiny constants
# -------------------------------------------------------------------------
B, T, V = 2, 8, 16
D = 16
N_SAMPLES = 5
N_MODELS = 3


# -------------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------------


def make_model() -> nn.Module:
    """Small model: Embedding -> Dropout -> Linear, output (B, T, V)."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(V, D)
            self.drop = nn.Dropout(0.1)
            self.proj = nn.Linear(D, V)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T) int
            h = self.embed(x)  # (B, T, D)
            h = self.drop(h)  # (B, T, D)
            return self.proj(h)  # (B, T, V)

    return TinyModel()


def make_input() -> torch.Tensor:
    return torch.randint(0, V, (B, T))


def make_labels() -> torch.Tensor:
    return torch.randint(0, V, (B, T))


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return make_model()


@pytest.fixture()
def mc(model):
    return MCDropoutEstimator(model, n_samples=N_SAMPLES, dropout_rate=0.1)


@pytest.fixture()
def ensemble():
    torch.manual_seed(42)
    models = [make_model() for _ in range(N_MODELS)]
    return DeepEnsemble(models)


# -------------------------------------------------------------------------
# 1. MCDropoutEstimator.predict: shapes correct
# -------------------------------------------------------------------------


def test_mc_predict_shapes(mc):
    x = make_input()
    mean_logits, var_logits, entropy = mc.predict(x)
    assert mean_logits.shape == (B, T, V), f"mean_logits shape {mean_logits.shape}"
    assert var_logits.shape == (B, T, V), f"var_logits shape {var_logits.shape}"
    assert entropy.shape == (B, T), f"entropy shape {entropy.shape}"


# -------------------------------------------------------------------------
# 2. MCDropoutEstimator: variance > 0 (dropout causes different passes)
# -------------------------------------------------------------------------


def test_mc_variance_positive(mc):
    x = make_input()
    _, var_logits, _ = mc.predict(x)
    assert var_logits.mean().item() > 0, "Expected positive variance with dropout"


# -------------------------------------------------------------------------
# 3. MCDropoutEstimator.predictive_entropy: shape (B, T) and >= 0
# -------------------------------------------------------------------------


def test_mc_predictive_entropy_shape_nonneg(mc):
    x = make_input()
    mc.enable_dropout()
    samples = []
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            samples.append(mc.model(x))
    logits_samples = torch.stack(samples, dim=0)  # (n_s, B, T, V)

    pe = mc.predictive_entropy(logits_samples)
    assert pe.shape == (B, T), f"shape {pe.shape}"
    assert (pe >= 0).all(), "Predictive entropy must be >= 0"


# -------------------------------------------------------------------------
# 4. MCDropoutEstimator.mutual_information: shape (B,T) and <= predictive_entropy
# -------------------------------------------------------------------------


def test_mc_mutual_information(mc):
    x = make_input()
    mc.enable_dropout()
    samples = []
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            samples.append(mc.model(x))
    logits_samples = torch.stack(samples, dim=0)

    pe = mc.predictive_entropy(logits_samples)
    mi = mc.mutual_information(logits_samples)

    assert mi.shape == (B, T), f"MI shape {mi.shape}"
    assert (mi <= pe + 1e-5).all(), "MI must be <= predictive entropy"
    assert (mi >= 0).all(), "MI must be non-negative"


# -------------------------------------------------------------------------
# 5. DeepEnsemble.forward: mean_logits (B,T,V), uncertainty (B,T)
# -------------------------------------------------------------------------


def test_ensemble_forward_shapes(ensemble):
    x = make_input()
    mean_logits, uncertainty = ensemble.forward(x)
    assert mean_logits.shape == (B, T, V), f"mean_logits shape {mean_logits.shape}"
    assert uncertainty.shape == (B, T), f"uncertainty shape {uncertainty.shape}"


# -------------------------------------------------------------------------
# 6. DeepEnsemble: uncertainty > 0 when ensemble members differ
# -------------------------------------------------------------------------


def test_ensemble_uncertainty_positive(ensemble):
    x = make_input()
    _, uncertainty = ensemble.forward(x)
    # Different random init => members produce different outputs
    assert uncertainty.mean().item() >= 0
    # At least some positions should have non-zero uncertainty
    assert uncertainty.max().item() > 0, "Expected non-zero uncertainty in ensemble"


# -------------------------------------------------------------------------
# 7. DeepEnsemble.calibrated_uncertainty: ECE in [0,1], MCE in [0,1]
# -------------------------------------------------------------------------


def test_ensemble_calibrated_uncertainty(ensemble):
    x = make_input()
    labels = make_labels()
    mean_logits, _ = ensemble.forward(x)
    ece, mce = ensemble.calibrated_uncertainty(mean_logits, labels)
    assert 0.0 <= ece <= 1.0, f"ECE={ece}"
    assert 0.0 <= mce <= 1.0, f"MCE={mce}"


# -------------------------------------------------------------------------
# 8. EntropyThresholder.is_uncertain: bool mask, correct threshold comparison
# -------------------------------------------------------------------------


def test_entropy_thresholder_is_uncertain():
    thresh = 1.0
    thresholder = EntropyThresholder(threshold=thresh)
    entropy = torch.tensor([[0.5, 1.5], [1.0, 2.0]])  # (2, 2)
    mask = thresholder.is_uncertain(entropy)
    assert mask.dtype == torch.bool
    expected = entropy > thresh
    assert (mask == expected).all(), f"Mask mismatch: {mask} vs {expected}"


# -------------------------------------------------------------------------
# 9. EntropyThresholder.filter_predictions: 4 keys, coverage in [0,1]
# -------------------------------------------------------------------------


def test_entropy_thresholder_filter_predictions(model):
    model.eval()
    x = make_input()
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)

    thresholder = EntropyThresholder(threshold=1.5)
    result = thresholder.filter_predictions(logits, entropy)

    for key in ("confident", "uncertain", "mean_confident_entropy", "coverage"):
        assert key in result, f"Missing key: {key}"

    cov = result["coverage"]
    assert 0.0 <= cov <= 1.0, f"Coverage={cov}"
    # confident + uncertain should cover everything
    assert (result["confident"] | result["uncertain"]).all()


# -------------------------------------------------------------------------
# 10. TemperatureCalibration.fit: returns float in [0.1, 10.0]
# -------------------------------------------------------------------------


def test_temperature_calibration_fit(model):
    model.eval()
    x = make_input()
    labels = make_labels()
    with torch.no_grad():
        logits = model(x)

    tc = TemperatureCalibration()
    t = tc.fit(logits, labels, n_iters=100)
    assert isinstance(t, float), f"Expected float, got {type(t)}"
    assert 0.1 <= t <= 10.0, f"Temperature {t} out of [0.1, 10.0]"


# -------------------------------------------------------------------------
# 11. TemperatureCalibration.calibrate: same shape as input
# -------------------------------------------------------------------------


def test_temperature_calibration_calibrate(model):
    model.eval()
    x = make_input()
    with torch.no_grad():
        logits = model(x)

    tc = TemperatureCalibration()
    calibrated = tc.calibrate(logits, temperature=2.0)
    assert calibrated.shape == logits.shape, f"Shape mismatch {calibrated.shape}"
    # Values should be halved
    assert torch.allclose(calibrated, logits / 2.0)


# -------------------------------------------------------------------------
# 12. TemperatureCalibration.expected_calibration_error: in [0, 1]
# -------------------------------------------------------------------------


def test_temperature_calibration_ece(model):
    model.eval()
    x = make_input()
    labels = make_labels()
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=-1)

    tc = TemperatureCalibration()
    ece = tc.expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0, f"ECE={ece}"


# -------------------------------------------------------------------------
# 13. UncertaintyBenchmark.auroc_uncertainty: in [0,1], 1.0 for perfect
# -------------------------------------------------------------------------


def test_auroc_uncertainty_range_and_perfect():
    bench = UncertaintyBenchmark()

    # Random case
    scores = torch.rand(20)
    is_wrong = torch.randint(0, 2, (20,)).bool()
    auroc = bench.auroc_uncertainty(scores, is_wrong)
    assert 0.0 <= auroc <= 1.0, f"AUROC={auroc}"

    # Perfect: wrong examples have strictly higher uncertainty
    is_wrong_perfect = torch.tensor([True] * 5 + [False] * 5)
    scores_perfect = torch.tensor([1.0] * 5 + [0.0] * 5)
    auroc_perfect = bench.auroc_uncertainty(scores_perfect, is_wrong_perfect)
    assert abs(auroc_perfect - 1.0) < 1e-4, f"Perfect AUROC={auroc_perfect}"


# -------------------------------------------------------------------------
# 14. UncertaintyBenchmark.brier_score: in [0,1], 0 for perfect predictions
# -------------------------------------------------------------------------


def test_brier_score_range_and_perfect():
    bench = UncertaintyBenchmark()

    # Random case
    probs = F.softmax(torch.randn(B, T, V), dim=-1)
    labels = make_labels()
    bs = bench.brier_score(probs, labels)
    assert 0.0 <= bs <= 1.0, f"Brier={bs}"

    # Perfect: one-hot on the correct class
    labels_perfect = torch.zeros(B * T, dtype=torch.long)
    probs_perfect = torch.zeros(B * T, V)
    probs_perfect.scatter_(1, labels_perfect.unsqueeze(1), 1.0)
    bs_perfect = bench.brier_score(probs_perfect, labels_perfect)
    assert abs(bs_perfect) < 1e-6, f"Perfect Brier={bs_perfect}"


# -------------------------------------------------------------------------
# 15. eval() mode gives lower variance than MCDropout (train) mode
# -------------------------------------------------------------------------


def test_eval_mode_gives_lower_variance(model):
    """model.eval() disables dropout => near-zero variance across passes."""
    torch.manual_seed(7)
    x = make_input()

    # MC Dropout (train mode) — should have positive variance
    mc_est = MCDropoutEstimator(model, n_samples=N_SAMPLES)
    _, var_train, _ = mc_est.predict(x)

    # Now set model to eval and collect samples manually
    model.eval()
    samples_eval = []
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            samples_eval.append(model(x))
    logits_eval = torch.stack(samples_eval, dim=0)
    var_eval = logits_eval.var(dim=0)

    assert var_train.mean().item() > var_eval.mean().item(), (
        f"Expected train var ({var_train.mean():.6f}) > eval var ({var_eval.mean():.6f})"
    )
