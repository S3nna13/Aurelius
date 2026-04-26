"""Tests for src/training/checkpoint_analyzer.py."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.checkpoint_analyzer import (
    CheckpointAnalysis,
    LayerMemoryStats,
    analyze_checkpoint_strategy,
    estimate_memory_with_checkpointing,
    profile_layer_activations,
)


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    return AureliusTransformer(cfg)


@pytest.fixture
def small_input():
    return torch.randint(0, 256, (1, 8))


# ─── test 1 ───────────────────────────────────────────────────────────────────


def test_profile_returns_layer_stats(small_model, small_input):
    """Returns one LayerMemoryStats per TransformerBlock."""
    stats = profile_layer_activations(small_model, small_input)
    n_layers = small_model.config.n_layers
    assert isinstance(stats, list)
    assert len(stats) == n_layers


# ─── test 2 ───────────────────────────────────────────────────────────────────


def test_layer_stats_fields(small_model, small_input):
    """Each LayerMemoryStats has the required fields with correct types."""
    stats = profile_layer_activations(small_model, small_input)
    for s in stats:
        assert isinstance(s, LayerMemoryStats)
        assert isinstance(s.layer_idx, int)
        assert isinstance(s.layer_name, str)
        assert isinstance(s.activation_bytes, int)
        assert isinstance(s.param_bytes, int)
        assert isinstance(s.grad_bytes, int)
        assert isinstance(s.should_checkpoint, bool)


# ─── test 3 ───────────────────────────────────────────────────────────────────


def test_activation_bytes_positive(small_model, small_input):
    """activation_bytes > 0 for each layer."""
    stats = profile_layer_activations(small_model, small_input)
    for s in stats:
        assert s.activation_bytes > 0, f"Layer {s.layer_idx} has zero activation_bytes"


# ─── test 4 ───────────────────────────────────────────────────────────────────


def test_analyze_returns_checkpoint_analysis(small_model, small_input):
    """analyze_checkpoint_strategy returns a CheckpointAnalysis with all fields."""
    analysis = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=16.0)
    assert isinstance(analysis, CheckpointAnalysis)
    assert isinstance(analysis.total_activation_bytes, int)
    assert isinstance(analysis.total_param_bytes, int)
    assert isinstance(analysis.peak_memory_no_checkpoint, int)
    assert isinstance(analysis.peak_memory_with_checkpoint, int)
    assert isinstance(analysis.memory_savings_bytes, int)
    assert isinstance(analysis.memory_savings_pct, float)
    assert isinstance(analysis.layers, list)
    assert isinstance(analysis.recommended_layers, list)
    assert isinstance(analysis.strategy, str)


# ─── test 5 ───────────────────────────────────────────────────────────────────


def test_memory_savings_pct_range(small_model, small_input):
    """0.0 <= memory_savings_pct <= 1.0."""
    analysis = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=16.0)
    assert 0.0 <= analysis.memory_savings_pct <= 1.0


# ─── test 6 ───────────────────────────────────────────────────────────────────


def test_recommended_layers_are_valid_indices(small_model, small_input):
    """All recommended layer indices are in [0, n_layers)."""
    n_layers = small_model.config.n_layers
    analysis = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=16.0)
    for idx in analysis.recommended_layers:
        assert 0 <= idx < n_layers, f"Invalid layer index {idx}"


# ─── test 7 ───────────────────────────────────────────────────────────────────


def test_no_checkpoint_needed_small_model(small_model, small_input):
    """Tiny model + large budget → no layers should be checkpointed."""
    analysis = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=16.0)
    assert analysis.recommended_layers == []
    assert analysis.strategy == "none"


# ─── test 8 ───────────────────────────────────────────────────────────────────


def test_checkpoint_needed_tight_budget(small_model, small_input):
    """Tiny model + near-zero budget → some layers should be checkpointed."""
    analysis = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=0.0001)
    assert len(analysis.recommended_layers) > 0
    assert analysis.strategy in ("uniform", "selective")


# ─── test 9 ───────────────────────────────────────────────────────────────────


def test_estimate_memory_checkpointing_lower(small_model, small_input):
    """Checkpointing all layers produces a lower (or equal) estimate than none."""
    stats = profile_layer_activations(small_model, small_input)
    all_indices = [s.layer_idx for s in stats]
    no_checkpoint = estimate_memory_with_checkpointing(stats, [])
    with_checkpoint = estimate_memory_with_checkpointing(stats, all_indices)
    assert with_checkpoint <= no_checkpoint


# ─── test 10 ──────────────────────────────────────────────────────────────────


def test_strategy_field_valid(small_model, small_input):
    """strategy is one of 'none', 'uniform', or 'selective'."""
    analysis_large = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=16.0)
    assert analysis_large.strategy in ("none", "uniform", "selective")

    analysis_tight = analyze_checkpoint_strategy(small_model, small_input, memory_budget_gb=0.0001)
    assert analysis_tight.strategy in ("none", "uniform", "selective")
