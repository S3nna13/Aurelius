"""Tests for early_exit.py — EarlyExitClassifier and EarlyExitTransformer."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.early_exit import (
    EarlyExitClassifier,
    EarlyExitConfig,
    EarlyExitTransformer,
    ExitStats,
    profile_exit_distribution,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """Small config that matches the project test config."""
    return AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )


@pytest.fixture
def exit_cfg():
    return EarlyExitConfig(n_exit_layers=2, exit_threshold=0.9, loss_weight=0.3, placement="uniform")


@pytest.fixture
def model(cfg, exit_cfg):
    return EarlyExitTransformer(cfg, exit_cfg)


@pytest.fixture
def input_ids(cfg):
    torch.manual_seed(0)
    return torch.randint(0, cfg.vocab_size, (2, 16))


# ---------------------------------------------------------------------------
# 1. EarlyExitClassifier output shapes
# ---------------------------------------------------------------------------

def test_exit_classifier_output_shapes(cfg):
    clf = EarlyExitClassifier(cfg.d_model, cfg.vocab_size)
    B, T = 2, 8
    x = torch.randn(B, T, cfg.d_model)
    logits, confidence = clf(x)
    assert logits.shape == (B, T, cfg.vocab_size), f"Expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    assert confidence.shape == (B, T), f"Expected ({B}, {T}), got {confidence.shape}"


# ---------------------------------------------------------------------------
# 2. EarlyExitClassifier confidence values in [0, 1]
# ---------------------------------------------------------------------------

def test_exit_classifier_confidence_in_range(cfg):
    clf = EarlyExitClassifier(cfg.d_model, cfg.vocab_size)
    x = torch.randn(3, 12, cfg.d_model)
    _, confidence = clf(x)
    assert (confidence >= 0.0).all(), "Confidence contains values below 0"
    assert (confidence <= 1.0).all(), "Confidence contains values above 1"


# ---------------------------------------------------------------------------
# 3. Training forward returns (loss, logits, None)
# ---------------------------------------------------------------------------

def test_early_exit_transformer_training_forward(model, input_ids):
    labels = input_ids.clone()
    result = model(input_ids, labels=labels)
    assert len(result) == 3, "Expected 3-tuple"
    loss, logits, third = result
    assert loss is not None, "Loss should not be None during training"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert logits is not None, "Logits should not be None"
    assert third is None, "Third element should be None during training"


# ---------------------------------------------------------------------------
# 4. Training loss is positive
# ---------------------------------------------------------------------------

def test_early_exit_training_loss_positive(model, input_ids):
    labels = input_ids.clone()
    loss, _, _ = model(input_ids, labels=labels)
    assert float(loss.item()) > 0.0, f"Loss should be positive, got {loss.item()}"


# ---------------------------------------------------------------------------
# 5. Inference without early exit returns (None, logits, None)
# ---------------------------------------------------------------------------

def test_early_exit_inference_no_exit(model, input_ids, cfg):
    with torch.no_grad():
        result = model(input_ids, use_early_exit=False)
    assert len(result) == 3
    none1, logits, none2 = result
    assert none1 is None, "First element should be None (no loss in inference)"
    assert logits is not None
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
    assert none2 is None, "Third element should be None when use_early_exit=False"


# ---------------------------------------------------------------------------
# 6. Inference with early exit returns a 3-tuple
# ---------------------------------------------------------------------------

def test_early_exit_inference_with_exit(model, input_ids, cfg):
    with torch.no_grad():
        result = model(input_ids, use_early_exit=True)
    assert len(result) == 3
    none1, logits, stats = result
    assert none1 is None
    assert logits is not None
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
    assert isinstance(stats, ExitStats)


# ---------------------------------------------------------------------------
# 7. Uniform exit positions are spread across layers
# ---------------------------------------------------------------------------

def test_exit_positions_uniform(cfg):
    exit_cfg = EarlyExitConfig(n_exit_layers=2, placement="uniform")
    model = EarlyExitTransformer(cfg, exit_cfg)
    positions = model.exit_positions
    n_layers = cfg.n_layers

    # No position should be the last layer
    assert n_layers - 1 not in positions, "Last layer should never be an exit position"
    # All positions must be valid layer indices
    for p in positions:
        assert 0 <= p < n_layers - 1, f"Position {p} out of valid range [0, {n_layers-2}]"
    # Should have at most n_exit_layers exits
    assert len(positions) <= exit_cfg.n_exit_layers


# ---------------------------------------------------------------------------
# 8. Late exit positions are in the last half of layers
# ---------------------------------------------------------------------------

def test_exit_positions_late(cfg):
    exit_cfg = EarlyExitConfig(n_exit_layers=2, placement="late")
    model = EarlyExitTransformer(cfg, exit_cfg)
    positions = model.exit_positions
    n_layers = cfg.n_layers
    half = n_layers // 2

    # No position should be the last layer
    assert n_layers - 1 not in positions, "Last layer should never be an exit position"
    # All positions must be in the second half
    for p in positions:
        assert p >= half, f"Position {p} is not in the last half (>= {half})"


# ---------------------------------------------------------------------------
# 9. compute_exit_stats returns ExitStats with correct attributes
# ---------------------------------------------------------------------------

def test_compute_exit_stats_returns_stats(model, input_ids):
    with torch.no_grad():
        stats = model.compute_exit_stats(input_ids)

    assert isinstance(stats, ExitStats), "Expected ExitStats instance"
    assert hasattr(stats, "layer_exit_counts"), "Missing layer_exit_counts"
    assert hasattr(stats, "mean_layers_used"), "Missing mean_layers_used"
    assert hasattr(stats, "flop_savings"), "Missing flop_savings"
    assert isinstance(stats.layer_exit_counts, list)
    assert isinstance(stats.mean_layers_used, float)
    assert isinstance(stats.flop_savings, float)


# ---------------------------------------------------------------------------
# 10. flop_savings is in [0, 1]
# ---------------------------------------------------------------------------

def test_flop_savings_in_range(model, input_ids):
    with torch.no_grad():
        stats = model.compute_exit_stats(input_ids)
    assert 0.0 <= stats.flop_savings <= 1.0, (
        f"flop_savings={stats.flop_savings} is outside [0, 1]"
    )
