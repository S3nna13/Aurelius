"""
Tests for src/interpretability/neuron_analysis.py

Tiny configuration:
    D=8, B=2, T=4, N=B*T=8, K=3
"""

import pytest
import torch

from src.interpretability.neuron_analysis import (
    NeuronAnalyzer,
    NeuronConfig,
    compute_activation_statistics,
    compute_neuron_correlation,
    find_dead_neurons,
    polysemanticity_score,
    top_activating_examples,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------
D = 8
B = 2
T = 4
N = B * T  # 8
K = 3

torch.manual_seed(42)


def _make_acts() -> torch.Tensor:
    """Random (B, T, D) activation tensor."""
    return torch.randn(B, T, D)


def _make_flat() -> torch.Tensor:
    """Random (N, D) flat activation tensor."""
    return torch.randn(N, D)


# ---------------------------------------------------------------------------
# 1. NeuronConfig defaults
# ---------------------------------------------------------------------------


def test_neuron_config_defaults():
    cfg = NeuronConfig()
    assert cfg.n_top_examples == 10
    assert cfg.activation_threshold == 0.0
    assert cfg.dead_neuron_threshold == 1e-3


# ---------------------------------------------------------------------------
# 2. compute_activation_statistics returns all 5 keys
# ---------------------------------------------------------------------------


def test_compute_activation_statistics_keys():
    acts = _make_acts()
    stats = compute_activation_statistics(acts)
    for key in ("mean", "std", "max", "min", "sparsity"):
        assert key in stats, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 3. Each stat has shape (D,)
# ---------------------------------------------------------------------------


def test_compute_activation_statistics_shapes():
    acts = _make_acts()
    stats = compute_activation_statistics(acts)
    for key, val in stats.items():
        assert val.shape == (D,), f"Key '{key}' has shape {val.shape}, expected ({D},)"


# ---------------------------------------------------------------------------
# 4. Sparsity values are in [0, 1]
# ---------------------------------------------------------------------------


def test_compute_activation_statistics_sparsity_range():
    acts = _make_acts()
    sparsity = compute_activation_statistics(acts)["sparsity"]
    assert (sparsity >= 0.0).all() and (sparsity <= 1.0).all(), (
        f"Sparsity out of range: min={sparsity.min()}, max={sparsity.max()}"
    )


# ---------------------------------------------------------------------------
# 5. find_dead_neurons returns shape (D,) bool tensor
# ---------------------------------------------------------------------------


def test_find_dead_neurons_shape_and_dtype():
    acts = _make_acts()
    dead = find_dead_neurons(acts)
    assert dead.shape == (D,), f"Expected ({D},), got {dead.shape}"
    assert dead.dtype == torch.bool, f"Expected bool dtype, got {dead.dtype}"


# ---------------------------------------------------------------------------
# 6. find_dead_neurons correctly identifies an all-zero neuron
# ---------------------------------------------------------------------------


def test_find_dead_neurons_finds_zero_neuron():
    acts = torch.randn(B, T, D)
    # Force neuron index 2 to be exactly zero everywhere
    acts[:, :, 2] = 0.0
    dead = find_dead_neurons(acts, threshold=1e-3)
    assert dead[2].item() is True, "Neuron 2 (all-zero) should be marked dead"
    # All other neurons should NOT be dead (random values >> 1e-3 with high prob)
    for i in range(D):
        if i != 2:
            assert not dead[i].item(), f"Neuron {i} should not be dead"


# ---------------------------------------------------------------------------
# 7. compute_neuron_correlation returns shape (D,)
# ---------------------------------------------------------------------------


def test_compute_neuron_correlation_shape():
    flat_a = _make_flat()
    flat_b = _make_flat()
    corr = compute_neuron_correlation(flat_a, flat_b)
    assert corr.shape == (D,), f"Expected ({D},), got {corr.shape}"


# ---------------------------------------------------------------------------
# 8. Correlation of identical activations is 1.0
# ---------------------------------------------------------------------------


def test_compute_neuron_correlation_identical_is_one():
    flat = _make_flat()
    corr = compute_neuron_correlation(flat, flat)
    # All non-zero-variance neurons should have corr ≈ 1.0
    # With random data, all neurons should have non-zero variance
    assert torch.allclose(corr, torch.ones(D), atol=1e-5), (
        f"Identical activations should have correlation 1.0, got {corr}"
    )


# ---------------------------------------------------------------------------
# 9. top_activating_examples returns shape (D, K)
# ---------------------------------------------------------------------------


def test_top_activating_examples_shape():
    flat = _make_flat()
    top = top_activating_examples(flat, K)
    assert top.shape == (D, K), f"Expected ({D},{K}), got {top.shape}"


# ---------------------------------------------------------------------------
# 10. top_activating_examples indices are valid (in [0, N))
# ---------------------------------------------------------------------------


def test_top_activating_examples_valid_indices():
    flat = _make_flat()
    top = top_activating_examples(flat, K)
    assert (top >= 0).all() and (top < N).all(), (
        f"Indices must be in [0, {N}), got min={top.min()}, max={top.max()}"
    )


# ---------------------------------------------------------------------------
# 11. polysemanticity_score returns shape (D,)
# ---------------------------------------------------------------------------


def test_polysemanticity_score_shape():
    flat = _make_flat()
    score = polysemanticity_score(flat, k=K)
    assert score.shape == (D,), f"Expected ({D},), got {score.shape}"


# ---------------------------------------------------------------------------
# 12. polysemanticity_score values are in [0, 1]
# ---------------------------------------------------------------------------


def test_polysemanticity_score_range():
    flat = _make_flat()
    score = polysemanticity_score(flat, k=K)
    assert (score >= 0.0).all() and (score <= 1.0).all(), (
        f"Score out of [0,1]: min={score.min()}, max={score.max()}"
    )


# ---------------------------------------------------------------------------
# 13. NeuronAnalyzer.analyze returns required keys
# ---------------------------------------------------------------------------


def test_neuron_analyzer_analyze_keys():
    cfg = NeuronConfig(n_top_examples=K)
    analyzer = NeuronAnalyzer(cfg)
    acts = _make_acts()
    result = analyzer.analyze(acts)
    for key in ("stats", "dead_mask", "top_examples"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 14. NeuronAnalyzer.analyze stats have correct sub-keys and shapes
# ---------------------------------------------------------------------------


def test_neuron_analyzer_analyze_stats_shapes():
    cfg = NeuronConfig(n_top_examples=K)
    analyzer = NeuronAnalyzer(cfg)
    acts = _make_acts()
    result = analyzer.analyze(acts)
    stats = result["stats"]
    for key in ("mean", "std", "max", "min", "sparsity"):
        assert key in stats
        assert stats[key].shape == (D,), f"stats['{key}'] has wrong shape {stats[key].shape}"


# ---------------------------------------------------------------------------
# 15. NeuronAnalyzer.compare_layers returns "correlation" key
# ---------------------------------------------------------------------------


def test_compare_layers_returns_correlation():
    cfg = NeuronConfig()
    analyzer = NeuronAnalyzer(cfg)
    layer_a = _make_acts()
    layer_b = _make_acts()
    result = analyzer.compare_layers(layer_a, layer_b)
    assert "correlation" in result, "compare_layers must return 'correlation' key"


# ---------------------------------------------------------------------------
# 16. compare_layers correlation has shape (D,)
# ---------------------------------------------------------------------------


def test_compare_layers_correlation_shape():
    cfg = NeuronConfig()
    analyzer = NeuronAnalyzer(cfg)
    layer_a = _make_acts()
    layer_b = _make_acts()
    result = analyzer.compare_layers(layer_a, layer_b)
    assert result["correlation"].shape == (D,), (
        f"correlation shape {result['correlation'].shape}, expected ({D},)"
    )


# ---------------------------------------------------------------------------
# 17. compare_layers returns mean_diff and std_diff with correct shapes
# ---------------------------------------------------------------------------


def test_compare_layers_mean_std_diff_shapes():
    cfg = NeuronConfig()
    analyzer = NeuronAnalyzer(cfg)
    layer_a = _make_acts()
    layer_b = _make_acts()
    result = analyzer.compare_layers(layer_a, layer_b)
    for key in ("mean_diff", "std_diff"):
        assert key in result, f"Missing key: {key}"
        assert result[key].shape == (D,), f"{key} shape {result[key].shape}, expected ({D},)"


# ---------------------------------------------------------------------------
# 18. dead_mask from analyze has correct shape and dtype
# ---------------------------------------------------------------------------


def test_neuron_analyzer_dead_mask_shape_dtype():
    cfg = NeuronConfig(n_top_examples=K)
    analyzer = NeuronAnalyzer(cfg)
    acts = _make_acts()
    result = analyzer.analyze(acts)
    dead_mask = result["dead_mask"]
    assert dead_mask.shape == (D,), f"dead_mask shape {dead_mask.shape}, expected ({D},)"
    assert dead_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 19. top_examples from analyze has correct shape
# ---------------------------------------------------------------------------


def test_neuron_analyzer_top_examples_shape():
    cfg = NeuronConfig(n_top_examples=K)
    analyzer = NeuronAnalyzer(cfg)
    acts = _make_acts()
    result = analyzer.analyze(acts)
    top = result["top_examples"]
    assert top.shape == (D, K), f"top_examples shape {top.shape}, expected ({D},{K})"


# ---------------------------------------------------------------------------
# 20. compute_neuron_correlation: zero-variance neuron → 0 correlation
# ---------------------------------------------------------------------------


def test_compute_neuron_correlation_zero_variance():
    flat_a = _make_flat()
    # Make neuron 0 constant in flat_b (zero variance)
    flat_b = _make_flat()
    flat_b[:, 0] = 5.0
    corr = compute_neuron_correlation(flat_a, flat_b)
    assert corr[0].item() == pytest.approx(0.0, abs=1e-6), (
        f"Zero-variance neuron should have correlation 0, got {corr[0].item()}"
    )
