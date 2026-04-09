"""Tests for structured pruning module."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.structured_pruning import (
    IterativePruner,
    StructuredPruningConfig,
    compute_layer_importance,
    compute_neuron_importance,
    compute_sparsity_schedule,
    prune_attention_heads,
    prune_neurons,
    regrow_neurons,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def calibration_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (2, 16))


# ---------------------------------------------------------------------------
# Test 1: StructuredPruningConfig defaults
# ---------------------------------------------------------------------------

def test_structured_pruning_config_defaults():
    cfg = StructuredPruningConfig()
    assert cfg.target_sparsity == 0.5
    assert cfg.pruning_schedule == "linear"
    assert cfg.n_pruning_steps == 10
    assert cfg.regrowth_fraction == 0.0
    assert cfg.min_neurons_per_layer == 1


# ---------------------------------------------------------------------------
# Test 2: compute_sparsity_schedule linear length
# ---------------------------------------------------------------------------

def test_sparsity_schedule_linear_length():
    schedule = compute_sparsity_schedule(0.5, 10, "linear")
    assert len(schedule) == 10


# ---------------------------------------------------------------------------
# Test 3: compute_sparsity_schedule linear ends at target
# ---------------------------------------------------------------------------

def test_sparsity_schedule_linear_ends_at_target():
    target = 0.6
    schedule = compute_sparsity_schedule(target, 8, "linear")
    assert abs(schedule[-1] - target) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: compute_sparsity_schedule cubic ends at target
# ---------------------------------------------------------------------------

def test_sparsity_schedule_cubic_ends_at_target():
    target = 0.4
    schedule = compute_sparsity_schedule(target, 5, "cubic")
    assert len(schedule) == 5
    assert abs(schedule[-1] - target) < 1e-6


# ---------------------------------------------------------------------------
# Test 5: compute_sparsity_schedule one_shot
# ---------------------------------------------------------------------------

def test_sparsity_schedule_one_shot():
    target = 0.7
    n_steps = 6
    schedule = compute_sparsity_schedule(target, n_steps, "one_shot")
    assert len(schedule) == n_steps
    # All zeros except last
    for val in schedule[:-1]:
        assert val == 0.0
    assert abs(schedule[-1] - target) < 1e-6


# ---------------------------------------------------------------------------
# Test 6: compute_neuron_importance shape (out,)
# ---------------------------------------------------------------------------

def test_compute_neuron_importance_shape():
    weight = torch.randn(32, 64)
    scores = compute_neuron_importance(weight, method="magnitude")
    assert scores.shape == (32,)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# Test 7: prune_neurons output shapes
# ---------------------------------------------------------------------------

def test_prune_neurons_output_shapes():
    weight = torch.randn(20, 10)
    bias = torch.randn(20)
    pruned_w, pruned_b, kept_idx = prune_neurons(weight, bias, sparsity=0.5)
    # Should keep at least 1 and at most 20 neurons
    assert pruned_w.ndim == 2
    assert pruned_w.shape[1] == 10
    assert pruned_b is not None
    assert pruned_b.shape[0] == pruned_w.shape[0]
    assert kept_idx.shape[0] == pruned_w.shape[0]


# ---------------------------------------------------------------------------
# Test 8: prune_neurons sparsity respected
# ---------------------------------------------------------------------------

def test_prune_neurons_sparsity_respected():
    torch.manual_seed(1)
    weight = torch.randn(40, 16)
    sparsity = 0.25
    pruned_w, _, kept_idx = prune_neurons(weight, None, sparsity=sparsity)
    expected_keep = max(1, 40 - int(40 * sparsity))
    assert pruned_w.shape[0] == expected_keep
    assert kept_idx.shape[0] == expected_keep


# ---------------------------------------------------------------------------
# Test 9: prune_attention_heads output shapes
# ---------------------------------------------------------------------------

def test_prune_attention_heads_output_shapes():
    torch.manual_seed(7)
    n_heads = 4
    n_kv_heads = 2
    head_dim = 16
    d_model = n_heads * head_dim  # 64
    sparsity = 0.5

    q_weight = torch.randn(n_heads * head_dim, d_model)
    k_weight = torch.randn(n_kv_heads * head_dim, d_model)
    v_weight = torch.randn(n_kv_heads * head_dim, d_model)
    o_weight = torch.randn(d_model, n_heads * head_dim)

    pq, pk, pv, po, kept = prune_attention_heads(
        q_weight, k_weight, v_weight, o_weight, head_dim, sparsity
    )

    n_keep = max(1, n_heads - int(n_heads * sparsity))
    assert pq.shape == (n_keep * head_dim, d_model)
    assert po.shape == (d_model, n_keep * head_dim)
    assert pq.ndim == 2
    assert po.ndim == 2


# ---------------------------------------------------------------------------
# Test 10: prune_attention_heads kept_head_indices correct count
# ---------------------------------------------------------------------------

def test_prune_attention_heads_kept_count():
    torch.manual_seed(3)
    n_heads = 4
    n_kv_heads = 2
    head_dim = 16
    d_model = n_heads * head_dim
    sparsity = 0.5

    q_weight = torch.randn(n_heads * head_dim, d_model)
    k_weight = torch.randn(n_kv_heads * head_dim, d_model)
    v_weight = torch.randn(n_kv_heads * head_dim, d_model)
    o_weight = torch.randn(d_model, n_heads * head_dim)

    _, _, _, _, kept = prune_attention_heads(
        q_weight, k_weight, v_weight, o_weight, head_dim, sparsity
    )

    n_keep = max(1, n_heads - int(n_heads * sparsity))
    assert len(kept) == n_keep
    # All indices within valid range
    assert all(0 <= h < n_heads for h in kept)


# ---------------------------------------------------------------------------
# Test 11: compute_layer_importance shape (n_layers,)
# ---------------------------------------------------------------------------

def test_compute_layer_importance_shape(small_model, calibration_ids):
    scores = compute_layer_importance(small_model, calibration_ids)
    n_layers = len(small_model.layers)
    assert scores.shape == (n_layers,)
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Test 12: IterativePruner.prune_step returns correct keys
# ---------------------------------------------------------------------------

def test_iterative_pruner_prune_step_keys(small_model, calibration_ids):
    cfg = StructuredPruningConfig(
        target_sparsity=0.3,
        n_pruning_steps=5,
        pruning_schedule="linear",
    )
    pruner = IterativePruner(small_model, cfg)
    result = pruner.prune_step(0, calibration_ids)

    assert "step" in result
    assert "sparsity" in result
    assert "n_pruned" in result
    assert result["step"] == 0
    assert isinstance(result["sparsity"], float)
    assert isinstance(result["n_pruned"], int)
