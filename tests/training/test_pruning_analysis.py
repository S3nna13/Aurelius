"""Tests for pruning_analysis.py — post-hoc pruning analysis utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.pruning_analysis import (
    GradientSensitivityPruner,
    LotteryTicketAnalyzer,
    MagnitudePruner,
    PruningScheduler,
    SparsityStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_model(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))


# ---------------------------------------------------------------------------
# SparsityStats
# ---------------------------------------------------------------------------


def test_layer_sparsity_all_nonzero():
    stats = SparsityStats()
    t = torch.ones(4, 4)
    assert stats.layer_sparsity(t) == 0.0


def test_layer_sparsity_all_zero():
    stats = SparsityStats()
    t = torch.zeros(4, 4)
    assert stats.layer_sparsity(t) == 1.0


def test_model_sparsity_returns_param_keys():
    stats = SparsityStats()
    model = _small_model()
    result = stats.model_sparsity(model)
    assert isinstance(result, dict)
    expected_keys = {name for name, _ in model.named_parameters()}
    assert set(result.keys()) == expected_keys


def test_total_sparsity_in_unit_interval():
    stats = SparsityStats()
    model = _small_model()
    s = stats.total_sparsity(model)
    assert 0.0 <= s <= 1.0


def test_total_sparsity_all_zero():
    stats = SparsityStats()
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    for p in model.parameters():
        p.data.zero_()
    assert stats.total_sparsity(model) == 1.0


# ---------------------------------------------------------------------------
# MagnitudePruner
# ---------------------------------------------------------------------------


def test_prune_tensor_zeros_correct_fraction():
    torch.manual_seed(42)
    pruner = MagnitudePruner(sparsity=0.5)
    t = torch.randn(100)
    result = pruner.prune_tensor(t)
    n_zeros = (result == 0).sum().item()
    # Should zero roughly 50 values (within ±5 due to ties)
    assert 45 <= n_zeros <= 55


def test_prune_tensor_returns_new_tensor():
    torch.manual_seed(42)
    pruner = MagnitudePruner(sparsity=0.5)
    t = torch.randn(20)
    original = t.clone()
    result = pruner.prune_tensor(t)
    # Original should not be modified
    assert torch.equal(t, original)
    # Result is a different object
    assert result.data_ptr() != t.data_ptr()


def test_prune_layer_returns_bool_mask_correct_shape():
    torch.manual_seed(0)
    pruner = MagnitudePruner(sparsity=0.5)
    param = nn.Parameter(torch.randn(8, 4))
    mask = pruner.prune_layer(param, mask_only=True)
    assert mask.dtype == torch.bool
    assert mask.shape == param.shape


def test_prune_layer_mask_only_does_not_modify_param():
    torch.manual_seed(0)
    pruner = MagnitudePruner(sparsity=0.5)
    param = nn.Parameter(torch.randn(8, 4))
    original_data = param.data.clone()
    pruner.prune_layer(param, mask_only=True)
    assert torch.equal(param.data, original_data)


def test_prune_model_returns_correct_keys():
    pruner = MagnitudePruner(sparsity=0.5)
    model = _small_model()
    masks = pruner.prune_model(model)
    expected_keys = {name for name, _ in model.named_parameters()}
    assert set(masks.keys()) == expected_keys


def test_prune_model_applies_sparsity():
    torch.manual_seed(0)
    pruner = MagnitudePruner(sparsity=0.5)
    model = _small_model()
    pruner.prune_model(model)
    stats = SparsityStats()
    s = stats.total_sparsity(model)
    # Rough check: should be close to 50 %
    assert 0.3 <= s <= 0.7


# ---------------------------------------------------------------------------
# GradientSensitivityPruner
# ---------------------------------------------------------------------------


def test_compute_sensitivity_shape():
    pruner = GradientSensitivityPruner(sparsity=0.5)
    param = nn.Parameter(torch.randn(8, 4))
    # Attach a fake gradient
    param.grad = torch.randn_like(param.data)
    sens = pruner.compute_sensitivity(param)
    assert sens.shape == param.shape


def test_sensitivity_values_nonnegative():
    torch.manual_seed(1)
    pruner = GradientSensitivityPruner(sparsity=0.5)
    param = nn.Parameter(torch.randn(8, 4))
    param.grad = torch.randn_like(param.data)
    sens = pruner.compute_sensitivity(param)
    assert (sens >= 0).all()


def test_prune_by_sensitivity_none_grad_fallback():
    """With no gradients, pruning should still work (falls back to magnitude)."""
    torch.manual_seed(2)
    pruner = GradientSensitivityPruner(sparsity=0.5)
    model = _small_model()
    # No backward pass — grads are None
    masks = pruner.prune_by_sensitivity(model)
    expected_keys = {name for name, _ in model.named_parameters()}
    assert set(masks.keys()) == expected_keys
    for mask in masks.values():
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# LotteryTicketAnalyzer
# ---------------------------------------------------------------------------


def test_save_initial_weights_clones():
    torch.manual_seed(3)
    model = _small_model()
    analyzer = LotteryTicketAnalyzer()
    ticket = analyzer.save_initial_weights(model)
    # Mutate model weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(999.0)
    # Ticket should be unaffected
    for name, w0 in ticket.items():
        assert not torch.all(w0 == 999.0), f"Ticket weight {name} was mutated"


def test_reset_to_ticket_restores_and_applies_mask():
    torch.manual_seed(4)
    model = _small_model()
    analyzer = LotteryTicketAnalyzer()
    ticket = analyzer.save_initial_weights(model)

    # Prune to get masks
    pruner = MagnitudePruner(sparsity=0.5)
    masks = pruner.prune_model(model)

    # Corrupt weights after pruning
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    analyzer.reset_to_ticket(model, ticket, masks)

    # Each param should equal initial_weight * mask
    for name, param in model.named_parameters():
        expected = ticket[name] * masks[name].float()
        assert torch.allclose(param.data, expected), f"Mismatch for {name}"


def test_ticket_similarity_identical():
    torch.manual_seed(5)
    model = _small_model()
    analyzer = LotteryTicketAnalyzer()
    ticket = analyzer.save_initial_weights(model)
    sim = analyzer.ticket_similarity(ticket, ticket)
    assert abs(sim - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# PruningScheduler
# ---------------------------------------------------------------------------


def test_sparsity_at_start_equals_initial():
    scheduler = PruningScheduler(
        initial_sparsity=0.1,
        final_sparsity=0.9,
        start_step=0,
        end_step=1000,
    )
    assert scheduler.sparsity_at_step(0) == pytest.approx(0.1)


def test_sparsity_at_end_equals_final():
    scheduler = PruningScheduler(
        initial_sparsity=0.0,
        final_sparsity=0.9,
        start_step=0,
        end_step=1000,
    )
    assert scheduler.sparsity_at_step(1000) == pytest.approx(0.9)


def test_sparsity_monotone_increasing():
    scheduler = PruningScheduler(
        initial_sparsity=0.0,
        final_sparsity=0.9,
        start_step=0,
        end_step=1000,
    )
    steps = [0, 100, 250, 500, 750, 1000]
    sparsities = [scheduler.sparsity_at_step(s) for s in steps]
    for i in range(len(sparsities) - 1):
        assert sparsities[i] <= sparsities[i + 1] + 1e-9


def test_should_prune_true_at_frequency():
    scheduler = PruningScheduler(start_step=0, end_step=1000, frequency=100)
    assert scheduler.should_prune(0)
    assert scheduler.should_prune(100)
    assert scheduler.should_prune(500)


def test_should_prune_false_outside_range():
    scheduler = PruningScheduler(start_step=100, end_step=500, frequency=100)
    assert not scheduler.should_prune(50)  # before start
    assert not scheduler.should_prune(600)  # after end


def test_should_prune_false_off_frequency():
    scheduler = PruningScheduler(start_step=0, end_step=1000, frequency=100)
    assert not scheduler.should_prune(150)
    assert not scheduler.should_prune(99)
