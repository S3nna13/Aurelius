"""Tests for src/eval/activation_outlier.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.eval.activation_outlier import (
    OutlierReport,
    ModelOutlierSummary,
    detect_outliers_iqr,
    detect_outliers_zscore,
    compute_per_channel_stats,
    suggest_quantization_scale,
    ActivationOutlierDetector,
    visualize_outlier_distribution,
)


# ---------------------------------------------------------------------------
# Shared small model fixture
# input_dim=32, hidden=16, output=8
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_model():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )
    model.eval()
    return model


@pytest.fixture
def small_input():
    torch.manual_seed(7)
    return torch.randn(2, 32)


# ---------------------------------------------------------------------------
# 1. detect_outliers_iqr returns (mask, scores) of shape (d,)
# ---------------------------------------------------------------------------

def test_iqr_output_shapes():
    torch.manual_seed(0)
    d = 64
    activations = torch.randn(10, d)
    mask, scores = detect_outliers_iqr(activations)
    assert mask.shape == (d,), f"Expected mask shape ({d},), got {mask.shape}"
    assert scores.shape == (d,), f"Expected scores shape ({d},), got {scores.shape}"
    assert mask.dtype == torch.bool
    assert scores.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. detect_outliers_iqr: no outliers in uniform data
# ---------------------------------------------------------------------------

def test_iqr_no_outliers_uniform():
    # All channels exactly equal -- IQR is 0, no outliers
    activations = torch.ones(20, 32)
    mask, scores = detect_outliers_iqr(activations, threshold=3.0)
    assert mask.sum().item() == 0, "Expected no outliers for uniform data"


# ---------------------------------------------------------------------------
# 3. detect_outliers_iqr: detects obvious spike (one channel 100x larger)
# ---------------------------------------------------------------------------

def test_iqr_detects_spike():
    torch.manual_seed(1)
    d = 64
    activations = torch.rand(20, d) * 0.1   # small random values
    # Inject a huge spike in channel 5 across all samples
    activations[:, 5] = 100.0
    mask, scores = detect_outliers_iqr(activations, threshold=3.0)
    assert mask[5].item() is True, "Channel 5 should be detected as outlier"
    assert scores[5].item() > 0.0


# ---------------------------------------------------------------------------
# 4. detect_outliers_zscore returns (mask, scores) of shape (d,)
# ---------------------------------------------------------------------------

def test_zscore_output_shapes():
    torch.manual_seed(2)
    d = 48
    activations = torch.randn(d)
    mask, scores = detect_outliers_zscore(activations)
    assert mask.shape == (d,), f"Expected mask shape ({d},), got {mask.shape}"
    assert scores.shape == (d,), f"Expected scores shape ({d},), got {scores.shape}"
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 5. detect_outliers_zscore: detects spike
# ---------------------------------------------------------------------------

def test_zscore_detects_spike():
    torch.manual_seed(3)
    d = 64
    activations = torch.randn(d)   # near-normal, std ~ 1
    activations[10] = 500.0        # extreme outlier
    mask, scores = detect_outliers_zscore(activations, threshold=3.0)
    assert mask[10].item() is True, "Channel 10 should be a z-score outlier"
    assert scores[10].item() > 3.0


# ---------------------------------------------------------------------------
# 6. compute_per_channel_stats returns dict with required keys
# ---------------------------------------------------------------------------

def test_per_channel_stats_keys():
    t = torch.randn(10, 32)
    stats = compute_per_channel_stats(t)
    required = {"mean", "std", "max", "absmax"}
    assert required == set(stats.keys()), f"Missing keys: {required - set(stats.keys())}"


# ---------------------------------------------------------------------------
# 7. compute_per_channel_stats shapes are (d,) for (N, d) input
# ---------------------------------------------------------------------------

def test_per_channel_stats_shape_2d():
    N, d = 15, 32
    t = torch.randn(N, d)
    stats = compute_per_channel_stats(t)
    for key, val in stats.items():
        assert val.shape == (d,), f"Key '{key}' shape {val.shape} != ({d},)"


# ---------------------------------------------------------------------------
# 8. compute_per_channel_stats shapes are (d,) for (B, T, d) input
# ---------------------------------------------------------------------------

def test_per_channel_stats_shape_3d():
    B, T, d = 3, 5, 16
    t = torch.randn(B, T, d)
    stats = compute_per_channel_stats(t)
    for key, val in stats.items():
        assert val.shape == (d,), f"Key '{key}' shape {val.shape} != ({d},)"


# ---------------------------------------------------------------------------
# 9. suggest_quantization_scale returns tensor of shape (d,)
# ---------------------------------------------------------------------------

def test_quantization_scale_shape():
    d = 32
    absmax = torch.rand(d) * 10.0
    scale = suggest_quantization_scale(absmax, n_bits=8)
    assert scale.shape == (d,), f"Expected scale shape ({d},), got {scale.shape}"


# ---------------------------------------------------------------------------
# 10. suggest_quantization_scale: outlier dims have larger scale
# ---------------------------------------------------------------------------

def test_quantization_scale_outlier_larger():
    d = 16
    absmax = torch.ones(d)
    outlier_mask = torch.zeros(d, dtype=torch.bool)
    outlier_mask[0] = True   # channel 0 is outlier
    scale = suggest_quantization_scale(absmax, n_bits=8, outlier_dims=outlier_mask, outlier_scale_factor=4.0)
    assert scale[0].item() > scale[1].item(), (
        "Outlier dim scale should be larger than normal dim scale"
    )
    # Specifically: outlier should be 4x the base
    base = 1.0 / (2 ** 7 - 1)
    assert scale[0].item() == pytest.approx(base * 4.0, rel=1e-4)


# ---------------------------------------------------------------------------
# 11. ActivationOutlierDetector instantiates and registers hooks
# ---------------------------------------------------------------------------

def test_detector_instantiates(mock_model):
    detector = ActivationOutlierDetector(mock_model)
    assert len(detector._hooks) > 0, "Expected hooks to be registered"
    assert len(detector._target_names) > 0, "Expected target module names"
    detector.remove_hooks()


# ---------------------------------------------------------------------------
# 12. ActivationOutlierDetector.collect runs without error
# ---------------------------------------------------------------------------

def test_detector_collect_no_error(mock_model, small_input):
    detector = ActivationOutlierDetector(mock_model)
    detector.collect(small_input)   # should not raise
    detector.remove_hooks()


# ---------------------------------------------------------------------------
# 13. n_samples_collected increments after collect()
# ---------------------------------------------------------------------------

def test_detector_n_samples(mock_model, small_input):
    detector = ActivationOutlierDetector(mock_model)
    assert detector.n_samples_collected == 0
    detector.collect(small_input)
    assert detector.n_samples_collected == 1
    detector.collect(small_input)
    assert detector.n_samples_collected == 2
    detector.remove_hooks()


# ---------------------------------------------------------------------------
# 14. ActivationOutlierDetector.analyze returns ModelOutlierSummary
# ---------------------------------------------------------------------------

def test_detector_analyze_returns_summary(mock_model, small_input):
    detector = ActivationOutlierDetector(mock_model)
    detector.collect(small_input)
    summary = detector.analyze()
    assert isinstance(summary, ModelOutlierSummary)
    assert isinstance(summary.per_layer_reports, dict)
    assert isinstance(summary.most_problematic_layers, list)
    detector.remove_hooks()


# ---------------------------------------------------------------------------
# 15. ModelOutlierSummary.global_outlier_ratio in [0, 1]
# ---------------------------------------------------------------------------

def test_global_outlier_ratio_range(mock_model, small_input):
    detector = ActivationOutlierDetector(mock_model)
    detector.collect(small_input)
    summary = detector.analyze()
    assert 0.0 <= summary.global_outlier_ratio <= 1.0, (
        f"global_outlier_ratio {summary.global_outlier_ratio} out of [0, 1]"
    )
    detector.remove_hooks()


# ---------------------------------------------------------------------------
# 16. visualize_outlier_distribution returns dict with 'outlier_indices' key
# ---------------------------------------------------------------------------

def test_visualize_returns_outlier_indices():
    torch.manual_seed(42)
    d = 32
    activations = torch.randn(d)
    activations[5] = 50.0
    mask, _ = detect_outliers_zscore(activations)
    result = visualize_outlier_distribution(activations, mask)
    assert "outlier_indices" in result, "'outlier_indices' key missing from result"
    assert isinstance(result["outlier_indices"], list)
    assert "normal_mean" in result
    assert "normal_std" in result
    assert "outlier_values" in result
    assert "histogram_bins" in result
    assert "histogram_counts" in result
