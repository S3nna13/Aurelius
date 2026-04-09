"""Tests for src/eval/calibration.py."""

from __future__ import annotations

import torch
import pytest

from src.eval.calibration import (
    CalibrationConfig,
    TemperatureScaler,
    PlattScaler,
    ModelCalibrator,
    compute_ece,
    compute_mce,
    compute_brier_score,
    reliability_diagram_data,
    fit_temperature_scaling,
    fit_platt_scaling,
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


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture
def default_cfg():
    return CalibrationConfig()


# ---------------------------------------------------------------------------
# 1. CalibrationConfig defaults
# ---------------------------------------------------------------------------

def test_calibration_config_defaults():
    cfg = CalibrationConfig()
    assert cfg.n_bins == 10
    assert cfg.temperature_init == 1.5


# ---------------------------------------------------------------------------
# 2. compute_ece — perfect calibration
# ---------------------------------------------------------------------------

def test_compute_ece_perfect():
    """A model that is exactly as confident as its accuracy should have ECE=0."""
    torch.manual_seed(0)
    # 100 samples: confident correct ones and uncertain wrong ones arranged so that
    # within each bin accuracy == confidence.
    # Simplest: all correct with prob=1.0 → single bin at top, acc=1, conf=1 → ECE=0
    probs = torch.ones(100)
    labels = torch.ones(100)
    ece = compute_ece(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. compute_ece — result in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_ece_range():
    torch.manual_seed(1)
    probs = torch.rand(200)
    labels = (torch.rand(200) > 0.5).float()
    ece = compute_ece(probs, labels, n_bins=10)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# 4. compute_mce — result in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_mce_range():
    torch.manual_seed(2)
    probs = torch.rand(200)
    labels = (torch.rand(200) > 0.5).float()
    mce = compute_mce(probs, labels, n_bins=10)
    assert 0.0 <= mce <= 1.0


# ---------------------------------------------------------------------------
# 5. compute_brier_score — perfect predictions
# ---------------------------------------------------------------------------

def test_compute_brier_perfect():
    """When predicted prob equals label exactly (all 1.0 and correct), brier=0."""
    probs = torch.ones(50)
    labels = torch.ones(50)
    assert compute_brier_score(probs, labels) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. compute_brier_score — result in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_brier_range():
    torch.manual_seed(3)
    probs = torch.rand(200)
    labels = (torch.rand(200) > 0.5).float()
    score = compute_brier_score(probs, labels)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 7. reliability_diagram_data — correct keys
# ---------------------------------------------------------------------------

def test_reliability_diagram_keys():
    probs = torch.rand(100)
    labels = (torch.rand(100) > 0.5).float()
    data = reliability_diagram_data(probs, labels, n_bins=10)
    assert set(data.keys()) == {"bin_centers", "bin_accs", "bin_confs", "bin_sizes"}


# ---------------------------------------------------------------------------
# 8. reliability_diagram_data — correct number of bins
# ---------------------------------------------------------------------------

def test_reliability_diagram_bin_count():
    probs = torch.rand(100)
    labels = (torch.rand(100) > 0.5).float()
    n_bins = 10
    data = reliability_diagram_data(probs, labels, n_bins=n_bins)
    assert len(data["bin_centers"]) == n_bins


# ---------------------------------------------------------------------------
# 9. TemperatureScaler — output shape matches input
# ---------------------------------------------------------------------------

def test_temperature_scaler_shape():
    scaler = TemperatureScaler(temperature=1.0)
    logits = torch.randn(32, 128)   # (N, C)
    probs = scaler(logits)
    assert probs.shape == logits.shape


# ---------------------------------------------------------------------------
# 10. TemperatureScaler — probabilities sum to 1
# ---------------------------------------------------------------------------

def test_temperature_scaler_sums_to_one():
    scaler = TemperatureScaler(temperature=1.2)
    logits = torch.randn(16, 64)
    probs = scaler(logits)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# 11. TemperatureScaler — high temp gives flatter distribution
# ---------------------------------------------------------------------------

def test_temperature_scaler_high_temp_flatter():
    """Temperature T=10 should yield a flatter (higher entropy) distribution than T=0.1."""
    torch.manual_seed(7)
    logits = torch.randn(1, 50)

    scaler_hot = TemperatureScaler(temperature=10.0)
    scaler_cold = TemperatureScaler(temperature=0.1)

    probs_hot = scaler_hot(logits)
    probs_cold = scaler_cold(logits)

    # Entropy: higher for flatter distribution
    entropy_hot = -(probs_hot * probs_hot.log().clamp(min=-1e9)).sum(dim=-1).item()
    entropy_cold = -(probs_cold * probs_cold.log().clamp(min=-1e9)).sum(dim=-1).item()

    assert entropy_hot > entropy_cold


# ---------------------------------------------------------------------------
# 12. PlattScaler — output in (0, 1)
# ---------------------------------------------------------------------------

def test_platt_scaler_output_range():
    scaler = PlattScaler()
    scores = torch.randn(50)
    probs = scaler(scores)
    assert (probs > 0).all()
    assert (probs < 1).all()


# ---------------------------------------------------------------------------
# 13. fit_temperature_scaling — returns TemperatureScaler instance
# ---------------------------------------------------------------------------

def test_fit_temperature_scaling_returns_scaler():
    torch.manual_seed(10)
    logits = torch.randn(100, 20)
    labels = torch.randint(0, 20, (100,))
    cfg = CalibrationConfig(n_iter=5)
    result = fit_temperature_scaling(logits, labels, cfg)
    assert isinstance(result, TemperatureScaler)


# ---------------------------------------------------------------------------
# 14. fit_platt_scaling — returns PlattScaler instance
# ---------------------------------------------------------------------------

def test_fit_platt_scaling_returns_scaler():
    torch.manual_seed(11)
    scores = torch.randn(100)
    labels = (torch.rand(100) > 0.5).float()
    cfg = CalibrationConfig(n_iter=10)
    result = fit_platt_scaling(scores, labels, cfg)
    assert isinstance(result, PlattScaler)


# ---------------------------------------------------------------------------
# 15. ModelCalibrator.calibrate — runs without error, returns TemperatureScaler
# ---------------------------------------------------------------------------

def test_model_calibrator_calibrate_runs(tiny_model):
    torch.manual_seed(20)
    cfg = CalibrationConfig(n_iter=5)
    calibrator = ModelCalibrator(tiny_model, cfg)

    input_ids = torch.randint(0, 256, (2, 16))
    target_ids = torch.randint(0, 256, (2, 16))

    scaler = calibrator.calibrate(input_ids, target_ids)
    assert isinstance(scaler, TemperatureScaler)


# ---------------------------------------------------------------------------
# 16. ModelCalibrator.evaluate_calibration — returns dict with correct keys
# ---------------------------------------------------------------------------

def test_model_calibrator_evaluate_keys(tiny_model):
    torch.manual_seed(21)
    cfg = CalibrationConfig(n_iter=5)
    calibrator = ModelCalibrator(tiny_model, cfg)

    input_ids = torch.randint(0, 256, (2, 16))
    target_ids = torch.randint(0, 256, (2, 16))

    result = calibrator.evaluate_calibration(input_ids, target_ids)
    expected_keys = {"ece_before", "ece_after", "brier_before", "brier_after", "temperature"}
    assert set(result.keys()) == expected_keys
