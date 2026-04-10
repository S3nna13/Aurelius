"""Tests for structured_pruning module.

Covers StructuredPruningConfig, HeadImportanceScorer, FFNImportanceScorer,
apply_head_mask, apply_ffn_mask, count_active_parameters, and StructuredPruner.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.structured_pruning import (
    FFNImportanceScorer,
    HeadImportanceScorer,
    StructuredPruner,
    StructuredPruningConfig,
    apply_ffn_mask,
    apply_head_mask,
    count_active_parameters,
)


# ---------------------------------------------------------------------------
# Fixtures
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
def calibration_data():
    torch.manual_seed(0)
    return [torch.randint(0, 256, (2, 16)) for _ in range(4)]


# ---------------------------------------------------------------------------
# Test 1: StructuredPruningConfig defaults
# ---------------------------------------------------------------------------

def test_structured_pruning_config_defaults():
    cfg = StructuredPruningConfig()
    assert cfg.head_pruning_ratio == 0.25
    assert cfg.ffn_pruning_ratio == 0.25
    assert cfg.importance_metric == "magnitude"
    assert cfg.n_calibration_steps == 10


# ---------------------------------------------------------------------------
# Test 2: HeadImportanceScorer.compute_head_importance returns dict with layer keys
# ---------------------------------------------------------------------------

def test_head_importance_scorer_returns_dict_with_layer_keys(small_model, calibration_data):
    cfg = StructuredPruningConfig(importance_metric="magnitude")
    scorer = HeadImportanceScorer(small_model, cfg)
    result = scorer.compute_head_importance(calibration_data)
    assert isinstance(result, dict)
    n_layers = len(small_model.layers)
    for i in range(n_layers):
        assert i in result, f"Layer {i} missing from head importance dict"


# ---------------------------------------------------------------------------
# Test 3: HeadImportanceScorer importance shape matches n_heads
# ---------------------------------------------------------------------------

def test_head_importance_shape_matches_n_heads(small_model, calibration_data):
    cfg = StructuredPruningConfig(importance_metric="magnitude")
    scorer = HeadImportanceScorer(small_model, cfg)
    result = scorer.compute_head_importance(calibration_data)
    n_heads = small_model.layers[0].attn.n_heads
    for layer_idx, scores in result.items():
        assert scores.shape == (n_heads,), (
            f"Layer {layer_idx}: expected shape ({n_heads},), got {scores.shape}"
        )


# ---------------------------------------------------------------------------
# Test 4: HeadImportanceScorer.get_heads_to_prune returns correct fraction
# ---------------------------------------------------------------------------

def test_get_heads_to_prune_correct_fraction(small_model, calibration_data):
    ratio = 0.5
    cfg = StructuredPruningConfig(importance_metric="magnitude", head_pruning_ratio=ratio)
    scorer = HeadImportanceScorer(small_model, cfg)
    importance = scorer.compute_head_importance(calibration_data)
    to_prune = scorer.get_heads_to_prune(importance)

    n_heads = small_model.layers[0].attn.n_heads
    n_layers = len(small_model.layers)
    total_heads = n_heads * n_layers
    expected_prune = int(total_heads * ratio)
    actual_prune = sum(len(v) for v in to_prune.values())
    assert actual_prune == expected_prune, (
        f"Expected {expected_prune} heads pruned, got {actual_prune}"
    )


# ---------------------------------------------------------------------------
# Test 5: FFNImportanceScorer.compute_neuron_importance returns dict with correct shapes
# ---------------------------------------------------------------------------

def test_ffn_importance_scorer_shapes(small_model, calibration_data):
    cfg = StructuredPruningConfig(importance_metric="magnitude")
    scorer = FFNImportanceScorer(small_model, cfg)
    result = scorer.compute_neuron_importance(calibration_data)
    assert isinstance(result, dict)
    n_layers = len(small_model.layers)
    for i in range(n_layers):
        assert i in result, f"Layer {i} missing from FFN importance dict"
        d_ff = small_model.layers[i].ffn.gate_proj.out_features
        assert result[i].shape == (d_ff,), (
            f"Layer {i}: expected shape ({d_ff},), got {result[i].shape}"
        )


# ---------------------------------------------------------------------------
# Test 6: FFNImportanceScorer.get_neurons_to_prune returns correct fraction
# ---------------------------------------------------------------------------

def test_get_neurons_to_prune_correct_fraction(small_model, calibration_data):
    ratio = 0.25
    cfg = StructuredPruningConfig(importance_metric="magnitude", ffn_pruning_ratio=ratio)
    scorer = FFNImportanceScorer(small_model, cfg)
    importance = scorer.compute_neuron_importance(calibration_data)
    to_prune = scorer.get_neurons_to_prune(importance)

    for layer_idx, neuron_ids in to_prune.items():
        d_ff = importance[layer_idx].shape[0]
        expected = int(d_ff * ratio)
        assert len(neuron_ids) == expected, (
            f"Layer {layer_idx}: expected {expected} neurons, got {len(neuron_ids)}"
        )


# ---------------------------------------------------------------------------
# Test 7: apply_ffn_mask zeroes out specified neuron weights
# ---------------------------------------------------------------------------

def test_apply_ffn_mask_zeroes_neuron_weights(small_model):
    # Pick neurons 0 and 1 in layer 0
    neurons_to_prune = {0: [0, 1]}
    apply_ffn_mask(small_model, neurons_to_prune)

    ffn = small_model.layers[0].ffn
    # gate_proj rows 0 and 1 should be all zeros
    assert ffn.gate_proj.weight[0].abs().sum().item() == 0.0
    assert ffn.gate_proj.weight[1].abs().sum().item() == 0.0
    # up_proj rows 0 and 1 should be all zeros
    assert ffn.up_proj.weight[0].abs().sum().item() == 0.0
    assert ffn.up_proj.weight[1].abs().sum().item() == 0.0
    # down_proj columns 0 and 1 should be all zeros
    assert ffn.down_proj.weight[:, 0].abs().sum().item() == 0.0
    assert ffn.down_proj.weight[:, 1].abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Test 8: apply_ffn_mask doesn't affect unmasked neurons
# ---------------------------------------------------------------------------

def test_apply_ffn_mask_does_not_affect_unmasked(small_model):
    # Record weight of neuron 5 before
    ffn = small_model.layers[0].ffn
    before = ffn.gate_proj.weight[5].clone()

    # Prune only neurons 0 and 1
    neurons_to_prune = {0: [0, 1]}
    apply_ffn_mask(small_model, neurons_to_prune)

    # Neuron 5 should be unchanged
    after = ffn.gate_proj.weight[5]
    assert torch.allclose(before, after), "Unmasked neuron was modified"


# ---------------------------------------------------------------------------
# Test 9: count_active_parameters returns required keys
# ---------------------------------------------------------------------------

def test_count_active_parameters_keys(small_model):
    result = count_active_parameters(small_model)
    assert "total" in result
    assert "nonzero" in result
    assert "sparsity" in result


# ---------------------------------------------------------------------------
# Test 10: count_active_parameters sparsity in [0, 1]
# ---------------------------------------------------------------------------

def test_count_active_parameters_sparsity_range(small_model):
    result = count_active_parameters(small_model)
    assert 0.0 <= result["sparsity"] <= 1.0
    assert result["total"] > 0
    assert result["nonzero"] <= result["total"]


# ---------------------------------------------------------------------------
# Test 11: StructuredPruner.calibrate returns required keys
# ---------------------------------------------------------------------------

def test_structured_pruner_calibrate_keys(small_model, calibration_data):
    cfg = StructuredPruningConfig()
    pruner = StructuredPruner(small_model, cfg)
    result = pruner.calibrate(calibration_data)
    assert "head_importance" in result
    assert "ffn_importance" in result
    assert isinstance(result["head_importance"], dict)
    assert isinstance(result["ffn_importance"], dict)


# ---------------------------------------------------------------------------
# Test 12: StructuredPruner.prune returns required keys
# ---------------------------------------------------------------------------

def test_structured_pruner_prune_keys(small_model, calibration_data):
    cfg = StructuredPruningConfig(head_pruning_ratio=0.25, ffn_pruning_ratio=0.25)
    pruner = StructuredPruner(small_model, cfg)
    result = pruner.prune(calibration_data)
    assert "heads_pruned" in result
    assert "neurons_pruned" in result
    assert "sparsity" in result


# ---------------------------------------------------------------------------
# Test 13: StructuredPruner.prune sparsity > 0 after pruning
# ---------------------------------------------------------------------------

def test_structured_pruner_prune_increases_sparsity(small_model, calibration_data):
    cfg = StructuredPruningConfig(head_pruning_ratio=0.5, ffn_pruning_ratio=0.5)
    pruner = StructuredPruner(small_model, cfg)
    result = pruner.prune(calibration_data)
    assert result["sparsity"] > 0.0, "Sparsity should be > 0 after pruning"


# ---------------------------------------------------------------------------
# Test 14: StructuredPruner.recover_accuracy returns finite float
# ---------------------------------------------------------------------------

def test_structured_pruner_recover_accuracy_returns_finite(small_model, calibration_data):
    cfg = StructuredPruningConfig(head_pruning_ratio=0.25, ffn_pruning_ratio=0.25)
    pruner = StructuredPruner(small_model, cfg)
    pruner.prune(calibration_data)

    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    loss = pruner.recover_accuracy(small_model, optimizer, calibration_data, n_steps=3)
    assert math.isfinite(loss), f"Expected finite loss, got {loss}"


# ---------------------------------------------------------------------------
# Test 15: count_active_parameters sparsity increases after pruning
# ---------------------------------------------------------------------------

def test_sparsity_increases_after_pruning(small_model, calibration_data):
    before = count_active_parameters(small_model)["sparsity"]
    cfg = StructuredPruningConfig(head_pruning_ratio=0.5, ffn_pruning_ratio=0.5)
    pruner = StructuredPruner(small_model, cfg)
    pruner.prune(calibration_data)
    after = count_active_parameters(small_model)["sparsity"]
    assert after > before, f"Sparsity should increase after pruning: {before} -> {after}"
