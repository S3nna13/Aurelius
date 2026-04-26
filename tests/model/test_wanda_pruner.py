"""Tests for WandaScorer and WandaPruner (Wanda activation-aware pruning).

Covers:
  - Config defaults
  - 2-D and 3-D activation accumulation
  - Activation norm positivity
  - Score shape and value ordering
  - Unstructured pruning sparsity and correctness
  - Semi-structured (N:M) pattern and shape
  - Reset / state clearing
  - Sparsity helper
  - Full pipeline (accumulate → score → prune)
  - Multi-accumulate
  - Integration test with a realistic weight matrix
"""

from __future__ import annotations

import pytest
import torch

from src.model.wanda_pruner import WandaConfig, WandaPruner, WandaScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> WandaConfig:
    return WandaConfig()


@pytest.fixture
def scorer(default_config: WandaConfig) -> WandaScorer:
    return WandaScorer(default_config)


@pytest.fixture
def pruner(default_config: WandaConfig) -> WandaPruner:
    return WandaPruner(default_config)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = WandaConfig()
    assert cfg.sparsity_ratio == 0.5
    assert cfg.semi_structured is False
    assert cfg.n == 2
    assert cfg.m == 4
    assert cfg.calibration_samples == 128


# ---------------------------------------------------------------------------
# 2. Accumulate 2-D activations
# ---------------------------------------------------------------------------


def test_accumulate_2d(scorer: WandaScorer):
    """accumulate([B, in]) should produce norms of shape [in_features]."""
    in_features = 32
    X = torch.randn(8, in_features)
    scorer.accumulate(X)
    norms = scorer.get_activation_norms()
    assert norms.shape == (in_features,)


# ---------------------------------------------------------------------------
# 3. Accumulate 3-D activations
# ---------------------------------------------------------------------------


def test_accumulate_3d(scorer: WandaScorer):
    """accumulate([B, T, in]) should produce norms of shape [in_features]."""
    in_features = 16
    X = torch.randn(4, 10, in_features)
    scorer.accumulate(X)
    norms = scorer.get_activation_norms()
    assert norms.shape == (in_features,)


# ---------------------------------------------------------------------------
# 4. Activation norms are non-negative
# ---------------------------------------------------------------------------


def test_activation_norms_positive(scorer: WandaScorer):
    X = torch.randn(16, 64)
    scorer.accumulate(X)
    norms = scorer.get_activation_norms()
    assert (norms >= 0).all(), "Activation norms must be non-negative"


# ---------------------------------------------------------------------------
# 5. Score shape
# ---------------------------------------------------------------------------


def test_score_shape(scorer: WandaScorer):
    out_f, in_f = 32, 64
    scorer.accumulate(torch.randn(8, in_f))
    W = torch.randn(out_f, in_f)
    scores = scorer.score(W)
    assert scores.shape == W.shape


# ---------------------------------------------------------------------------
# 6. Score values — high-activation columns get higher scores
# ---------------------------------------------------------------------------


def test_score_values(scorer: WandaScorer):
    """Columns with larger activations should yield larger scores for equal weights."""
    torch.manual_seed(42)
    in_f = 8
    # Craft activations so column 0 has much larger norm than others
    X = torch.zeros(100, in_f)
    X[:, 0] = 10.0  # large activation in column 0
    X[:, 1:] = 0.01  # tiny activations elsewhere
    scorer.accumulate(X)

    # Weight matrix with identical absolute values per row
    W = torch.ones(4, in_f)
    scores = scorer.score(W)

    # Every row should have its max score at column 0
    for row in range(scores.shape[0]):
        assert scores[row, 0] == scores[row].max(), (
            f"Row {row}: expected column 0 to have max score"
        )


# ---------------------------------------------------------------------------
# 7. Unstructured pruning — actual zero fraction matches sparsity_ratio
# ---------------------------------------------------------------------------


def test_prune_unstructured_sparsity():
    torch.manual_seed(0)
    config = WandaConfig(sparsity_ratio=0.5)
    scorer = WandaScorer(config)
    pruner = WandaPruner(config)

    in_f, out_f = 64, 128
    scorer.accumulate(torch.randn(32, in_f))
    W = torch.randn(out_f, in_f)
    W_pruned = pruner.prune(W, scorer)

    actual_sparsity = pruner.sparsity(W_pruned)
    assert abs(actual_sparsity - 0.5) < 0.01, f"Expected sparsity ≈ 0.5, got {actual_sparsity:.4f}"


# ---------------------------------------------------------------------------
# 8. Unstructured pruning keeps highest-scored weights
# ---------------------------------------------------------------------------


def test_prune_unstructured_keeps_high_scores():
    torch.manual_seed(7)
    config = WandaConfig(sparsity_ratio=0.5)
    scorer = WandaScorer(config)
    pruner = WandaPruner(config)

    in_f, out_f = 16, 8
    # Column 0 gets large activation norm → high scores for any weight there
    X = torch.zeros(64, in_f)
    X[:, 0] = 100.0
    X[:, 1:] = 0.001
    scorer.accumulate(X)

    # All weights equal magnitude so only activation norm decides
    W = torch.ones(out_f, in_f)
    scorer.score(W)

    W_pruned = pruner.prune(W, scorer)

    # Column 0 has the largest scores; its weights should all survive (not zeroed)
    assert (W_pruned[:, 0] != 0).all(), "High-score column 0 weights should all survive pruning"


# ---------------------------------------------------------------------------
# 9. Semi-structured pruning pattern — every group of M has exactly (M-N) zeros
# ---------------------------------------------------------------------------


def test_prune_semi_structured_pattern():
    torch.manual_seed(3)
    n, m = 2, 4
    config = WandaConfig(semi_structured=True, n=n, m=m)
    scorer = WandaScorer(config)
    pruner = WandaPruner(config)

    out_f, in_f = 8, 16  # in_f must be divisible by m
    scorer.accumulate(torch.randn(16, in_f))
    W = torch.randn(out_f, in_f)
    W_pruned = pruner.prune(W, scorer)

    # Reshape into groups of M along in_features
    W_grouped = W_pruned.reshape(out_f, in_f // m, m)
    zeros_per_group = (W_grouped == 0).sum(dim=-1)  # [out_f, in_f//m]

    expected_zeros = m - n  # = 2
    assert (zeros_per_group == expected_zeros).all(), (
        f"Every group of {m} should have exactly {expected_zeros} zeros.\nGot: {zeros_per_group}"
    )


# ---------------------------------------------------------------------------
# 10. Semi-structured pruning shape unchanged
# ---------------------------------------------------------------------------


def test_prune_semi_structured_shape():
    config = WandaConfig(semi_structured=True, n=2, m=4)
    scorer = WandaScorer(config)
    pruner = WandaPruner(config)

    out_f, in_f = 4, 8
    scorer.accumulate(torch.randn(8, in_f))
    W = torch.randn(out_f, in_f)
    W_pruned = pruner.prune(W, scorer)

    assert W_pruned.shape == W.shape


# ---------------------------------------------------------------------------
# 11. Reset clears accumulated state
# ---------------------------------------------------------------------------


def test_reset_clears_state(scorer: WandaScorer):
    scorer.accumulate(torch.randn(8, 32))
    scorer.reset()
    with pytest.raises(RuntimeError):
        scorer.get_activation_norms()


# ---------------------------------------------------------------------------
# 12. Sparsity helper returns correct fraction
# ---------------------------------------------------------------------------


def test_sparsity_zero_weight(pruner: WandaPruner):
    W = torch.zeros(4, 8)
    assert pruner.sparsity(W) == 1.0

    W2 = torch.ones(4, 8)
    assert pruner.sparsity(W2) == 0.0

    W3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert pruner.sparsity(W3) == 0.5


# ---------------------------------------------------------------------------
# 13. Full pipeline — accumulate → score → prune, check sparsity
# ---------------------------------------------------------------------------


def test_prune_full_pipeline():
    torch.manual_seed(99)
    ratio = 0.6
    config = WandaConfig(sparsity_ratio=ratio)
    scorer = WandaScorer(config)
    pruner = WandaPruner(config)

    in_f, out_f = 50, 100
    scorer.accumulate(torch.randn(64, in_f))
    W = torch.randn(out_f, in_f)
    W_pruned = pruner.prune(W, scorer)

    actual = pruner.sparsity(W_pruned)
    assert abs(actual - ratio) < 0.02, (
        f"Full pipeline: expected sparsity ≈ {ratio}, got {actual:.4f}"
    )


# ---------------------------------------------------------------------------
# 14. Multi-accumulate — two calls, norms reflect combined data
# ---------------------------------------------------------------------------


def test_multi_accumulate():
    """Two accumulate calls should yield the same norms as a single combined call."""
    torch.manual_seed(5)
    in_f = 32
    X1 = torch.randn(8, in_f)
    X2 = torch.randn(8, in_f)

    # Two separate accumulations
    s_two = WandaScorer(WandaConfig())
    s_two.accumulate(X1)
    s_two.accumulate(X2)
    norms_two = s_two.get_activation_norms()

    # Single combined accumulation
    s_one = WandaScorer(WandaConfig())
    s_one.accumulate(torch.cat([X1, X2], dim=0))
    norms_one = s_one.get_activation_norms()

    assert torch.allclose(norms_two, norms_one, atol=1e-5), (
        "Multi-accumulate should equal single-call accumulate on concatenated data"
    )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_integration_wanda_pruner():
    """End-to-end: accumulate → prune a [128, 64] weight matrix, verify sparsity."""
    torch.manual_seed(2024)
    sparsity_ratio = 0.5
    config = WandaConfig(
        sparsity_ratio=sparsity_ratio,
        semi_structured=False,
        calibration_samples=128,
    )

    scorer = WandaScorer(config)
    pruner = WandaPruner(config)

    out_features, in_features = 128, 64

    # Simulate 128 calibration samples split across 8 batches of 16
    for _ in range(8):
        X = torch.randn(16, in_features)
        scorer.accumulate(X)

    W = torch.randn(out_features, in_features)

    # Score and prune
    W_pruned = pruner.prune(W, scorer)

    # Shape preserved
    assert W_pruned.shape == (out_features, in_features)

    # Sparsity matches target within 1 %
    actual_sparsity = pruner.sparsity(W_pruned)
    assert abs(actual_sparsity - sparsity_ratio) < 0.01, (
        f"Integration: expected sparsity ≈ {sparsity_ratio}, got {actual_sparsity:.4f}"
    )

    # Non-zero weights should be a subset of the original non-zero weights
    non_zero_mask = W_pruned != 0
    assert torch.allclose(W_pruned[non_zero_mask], W[non_zero_mask].float()), (
        "Non-zero surviving weights must equal the original weight values"
    )

    # Registry check
    from src.model import MODEL_COMPONENT_REGISTRY

    assert "wanda_pruner" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["wanda_pruner"] is WandaPruner
