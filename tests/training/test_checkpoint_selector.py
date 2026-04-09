"""Tests for src/training/checkpoint_selector.py"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.checkpoint_selector import (
    CheckpointSelectorConfig,
    CheckpointedLayer,
    CheckpointSelector,
    apply_selective_checkpointing,
    compute_recomputation_cost,
    estimate_activation_bytes,
    select_memory_optimal_layers,
    select_uniform_layers,
)

# Tiny model config used across all tests
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


@pytest.fixture
def tiny_model():
    return AureliusTransformer(TINY_CFG)


# -------------------------------------------------------------------------
# 1. test_config_defaults
# -------------------------------------------------------------------------
def test_config_defaults():
    cfg = CheckpointSelectorConfig()
    assert cfg.strategy == "uniform"
    assert cfg.checkpoint_ratio == 0.5
    assert cfg.memory_budget_gb == 8.0
    assert cfg.bytes_per_element == 4


# -------------------------------------------------------------------------
# 2. test_estimate_activation_bytes_positive
# -------------------------------------------------------------------------
def test_estimate_activation_bytes_positive():
    result = estimate_activation_bytes(
        n_layers=2, seq_len=64, d_model=64, batch_size=2, bytes_per_element=4
    )
    assert result > 0


# -------------------------------------------------------------------------
# 3. test_estimate_activation_bytes_scales_with_batch
# -------------------------------------------------------------------------
def test_estimate_activation_bytes_scales_with_batch():
    base = estimate_activation_bytes(
        n_layers=4, seq_len=32, d_model=64, batch_size=1, bytes_per_element=4
    )
    doubled = estimate_activation_bytes(
        n_layers=4, seq_len=32, d_model=64, batch_size=2, bytes_per_element=4
    )
    assert doubled == 2 * base


# -------------------------------------------------------------------------
# 4. test_select_uniform_zero_ratio
# -------------------------------------------------------------------------
def test_select_uniform_zero_ratio():
    result = select_uniform_layers(n_layers=8, ratio=0.0)
    assert result == []


# -------------------------------------------------------------------------
# 5. test_select_uniform_full_ratio
# -------------------------------------------------------------------------
def test_select_uniform_full_ratio():
    result = select_uniform_layers(n_layers=6, ratio=1.0)
    assert result == list(range(6))


# -------------------------------------------------------------------------
# 6. test_select_uniform_half
# -------------------------------------------------------------------------
def test_select_uniform_half():
    result = select_uniform_layers(n_layers=4, ratio=0.5)
    assert len(result) == 2
    assert result == sorted(result)


# -------------------------------------------------------------------------
# 7. test_select_uniform_indices_sorted
# -------------------------------------------------------------------------
def test_select_uniform_indices_sorted():
    result = select_uniform_layers(n_layers=10, ratio=0.4)
    assert result == sorted(result)


# -------------------------------------------------------------------------
# 8. test_select_memory_optimal_tight_budget
# -------------------------------------------------------------------------
def test_select_memory_optimal_tight_budget():
    # Use a budget of 1 byte — should checkpoint all layers
    result = select_memory_optimal_layers(
        n_layers=4,
        seq_len=64,
        d_model=64,
        batch_size=2,
        budget_bytes=1,
        bytes_per_element=4,
    )
    assert len(result) == 4
    assert result == list(range(4))


# -------------------------------------------------------------------------
# 9. test_select_memory_optimal_large_budget
# -------------------------------------------------------------------------
def test_select_memory_optimal_large_budget():
    # Use a huge budget — no checkpointing needed
    result = select_memory_optimal_layers(
        n_layers=4,
        seq_len=64,
        d_model=64,
        batch_size=2,
        budget_bytes=10 * (1024 ** 3),  # 10 GB
        bytes_per_element=4,
    )
    assert result == []


# -------------------------------------------------------------------------
# 10. test_compute_recomputation_cost_range
# -------------------------------------------------------------------------
def test_compute_recomputation_cost_range():
    for n_ckpt, n_layers in [(0, 4), (2, 4), (4, 4), (3, 10)]:
        cost = compute_recomputation_cost(n_ckpt, n_layers)
        assert 0.0 <= cost <= 1.0, f"cost={cost} out of range for n_ckpt={n_ckpt}, n_layers={n_layers}"


# -------------------------------------------------------------------------
# 11. test_compute_recomputation_cost_all
# -------------------------------------------------------------------------
def test_compute_recomputation_cost_all():
    n = 6
    cost = compute_recomputation_cost(n_checkpointed=n, n_layers=n)
    assert cost == 1.0


# -------------------------------------------------------------------------
# 12. test_apply_selective_checkpointing_wraps_layers
# -------------------------------------------------------------------------
def test_apply_selective_checkpointing_wraps_layers(tiny_model):
    apply_selective_checkpointing(tiny_model, [0])
    assert isinstance(tiny_model.layers[0], CheckpointedLayer)


# -------------------------------------------------------------------------
# 13. test_apply_selective_checkpointing_unselected_unchanged
# -------------------------------------------------------------------------
def test_apply_selective_checkpointing_unselected_unchanged(tiny_model):
    original_layer_1 = tiny_model.layers[1]
    apply_selective_checkpointing(tiny_model, [0])
    # Layer 1 was not selected — should not be a CheckpointedLayer
    assert not isinstance(tiny_model.layers[1], CheckpointedLayer)
    assert tiny_model.layers[1] is original_layer_1


# -------------------------------------------------------------------------
# 14. test_checkpointed_layer_forward_runs
# -------------------------------------------------------------------------
def test_checkpointed_layer_forward_runs(tiny_model):
    """CheckpointedLayer should run forward without errors."""
    layer = tiny_model.layers[0]
    ckpt_layer = CheckpointedLayer(layer)

    batch_size, seq_len, d_model = 1, 8, TINY_CFG.d_model
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    # Build freqs_cis required by TransformerBlock
    freqs_cis = tiny_model.freqs_cis[:seq_len]

    # Run in training mode so checkpoint is active
    ckpt_layer.train()
    output, kv = ckpt_layer(x, freqs_cis)
    assert output.shape == (batch_size, seq_len, d_model)


# -------------------------------------------------------------------------
# 15. test_selector_apply_returns_indices
# -------------------------------------------------------------------------
def test_selector_apply_returns_indices(tiny_model):
    cfg = CheckpointSelectorConfig(strategy="uniform", checkpoint_ratio=0.5)
    selector = CheckpointSelector(tiny_model, cfg)
    indices = selector.apply(seq_len=64, batch_size=2)
    assert isinstance(indices, list)
    assert all(isinstance(i, int) for i in indices)


# -------------------------------------------------------------------------
# 16. test_selector_memory_savings_keys
# -------------------------------------------------------------------------
def test_selector_memory_savings_keys(tiny_model):
    cfg = CheckpointSelectorConfig(strategy="uniform", checkpoint_ratio=0.5)
    selector = CheckpointSelector(tiny_model, cfg)
    savings = selector.estimate_memory_savings(seq_len=64, batch_size=2)
    expected_keys = {"baseline_gb", "saved_gb", "savings_fraction", "recompute_overhead"}
    assert set(savings.keys()) == expected_keys
