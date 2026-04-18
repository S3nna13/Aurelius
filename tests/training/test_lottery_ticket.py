"""Tests for the Lottery Ticket Hypothesis / IMP implementation.

Covers all 12 specified test cases using a tiny 2-layer MLP so tests run
without the full AureliusTransformer.

Paper: Frankle & Carlin, arXiv:1803.03635
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.lottery_ticket import LotteryConfig, LotteryTicketPruner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_mlp() -> nn.Module:
    """2-layer MLP: Linear(16→32) → ReLU → Linear(32→8).

    Contains both weight and bias parameters; weight names are:
        layers.0.weight  (32, 16)  — 512 elements
        layers.2.weight  (8, 32)   — 256 elements
    Bias names end with .bias and must NOT be pruned.
    """
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    # Ensure no weights happen to be exactly zero at init.
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data += torch.randn_like(param.data) * 0.01 + 0.01
    return model


@pytest.fixture
def pruner(tiny_mlp) -> LotteryTicketPruner:
    """Pruner with default config (global pruning, rate=0.20)."""
    return LotteryTicketPruner(tiny_mlp)


@pytest.fixture
def pruner_with_saved(tiny_mlp) -> LotteryTicketPruner:
    """Pruner with θ_0 already saved."""
    p = LotteryTicketPruner(tiny_mlp)
    p.save_initial_weights()
    return p


# ── Test 1: save_initial_weights stores deep copies ───────────────────────────

def test_save_initial_weights_deep_copy(pruner):
    """θ_0 must be independent copies, not references to live parameters."""
    pruner.save_initial_weights()

    # Mutate the model parameters in-place.
    with torch.no_grad():
        for param in pruner.model.parameters():
            param.data.fill_(999.0)

    # Stored θ_0 must not reflect the mutation.
    for name, saved in pruner.initial_weights.items():
        assert not torch.all(saved == 999.0), (
            f"Parameter '{name}' in initial_weights was mutated — "
            "save_initial_weights() stored a reference, not a copy."
        )


# ── Test 2: compute_masks returns same keys as non-bias named parameters ───────

def test_compute_masks_keys(pruner_with_saved):
    """compute_masks must return one entry per non-bias named parameter."""
    masks = pruner_with_saved.compute_masks()

    expected_keys = {
        name
        for name, _ in pruner_with_saved.model.named_parameters()
        if not name.endswith("bias") and name.split(".")[-1] != "bias"
    }
    assert set(masks.keys()) == expected_keys


# ── Test 3: mask values are strictly 0 or 1 ───────────────────────────────────

def test_compute_masks_binary(pruner_with_saved):
    """Each mask element must be exactly 0.0 or 1.0."""
    masks = pruner_with_saved.compute_masks()

    for name, mask in masks.items():
        unique_vals = mask.unique()
        for val in unique_vals:
            assert val.item() in (0.0, 1.0), (
                f"Mask for '{name}' contains non-binary value {val.item()}"
            )


# ── Test 4: fraction of zeros ≈ pruning_rate ──────────────────────────────────

def test_compute_masks_pruning_fraction(pruner_with_saved):
    """The fraction of zeros in returned masks must be close to pruning_rate."""
    rate = 0.20
    masks = pruner_with_saved.compute_masks(pruning_rate=rate)

    total = sum(m.numel() for m in masks.values())
    zeros = sum((m == 0).sum().item() for m in masks.values())
    actual_rate = zeros / total

    # Allow ±5 % tolerance (threshold ties can shift the count slightly).
    assert abs(actual_rate - rate) < 0.05, (
        f"Expected ~{rate:.0%} zeros, got {actual_rate:.2%}"
    )


# ── Test 5: apply_masks zeros out pruned weights ──────────────────────────────

def test_apply_masks_zeroes_weights(pruner_with_saved):
    """After apply_masks, parameters at mask==0 positions must equal 0."""
    masks = pruner_with_saved.compute_masks()
    pruner_with_saved.apply_masks(masks)

    for name, param in pruner_with_saved.model.named_parameters():
        if name in masks:
            pruned_positions = masks[name] == 0
            assert (param.data[pruned_positions] == 0).all(), (
                f"'{name}': pruned positions are not zero after apply_masks()"
            )


# ── Test 6: rewind_to_initial restores unmasked weights to θ_0 ───────────────

def test_rewind_restores_unpruned_weights(pruner_with_saved):
    """Unmasked (kept) weights must equal θ_0 after rewind."""
    # Store θ_0 before any changes.
    theta_0 = {
        name: tensor.clone()
        for name, tensor in pruner_with_saved.initial_weights.items()
    }

    masks = pruner_with_saved.compute_masks()

    # Simulate training: perturb the model weights (as if θ_k ≠ θ_0).
    with torch.no_grad():
        for param in pruner_with_saved.model.parameters():
            param.data += torch.randn_like(param.data) * 5.0

    pruner_with_saved.rewind_to_initial(masks)

    for name, param in pruner_with_saved.model.named_parameters():
        if name in masks:
            kept = masks[name].bool()
            assert torch.allclose(param.data[kept], theta_0[name][kept]), (
                f"'{name}': kept weights were not rewound to θ_0"
            )


# ── Test 7: rewind_to_initial keeps masked weights at zero ────────────────────

def test_rewind_keeps_pruned_zero(pruner_with_saved):
    """Pruned (mask==0) positions must stay exactly zero after rewind."""
    masks = pruner_with_saved.compute_masks()
    pruner_with_saved.rewind_to_initial(masks)

    for name, param in pruner_with_saved.model.named_parameters():
        if name in masks:
            pruned = masks[name] == 0
            assert (param.data[pruned] == 0).all(), (
                f"'{name}': pruned positions are non-zero after rewind"
            )


# ── Test 8: sparsity is 0 before any pruning ──────────────────────────────────

def test_sparsity_zero_before_pruning(pruner):
    """A freshly constructed pruner must report 0 % sparsity."""
    # The fixture adds a small offset so no weight is accidentally zero.
    assert pruner.sparsity() == pytest.approx(0.0), (
        "Expected 0 % sparsity on untouched model"
    )


# ── Test 9: sparsity increases after run_round ────────────────────────────────

def test_sparsity_increases_after_run_round(pruner_with_saved):
    """sparsity() must be strictly larger after run_round() than before."""
    before = pruner_with_saved.sparsity()
    pruner_with_saved.run_round()
    after = pruner_with_saved.sparsity()

    assert after > before, (
        f"Expected sparsity to increase; before={before:.4f}, after={after:.4f}"
    )


# ── Test 10: global_pruning can fully prune small layers ──────────────────────

def test_global_pruning_can_fully_prune_small_layer():
    """With global pruning and a high rate, small layers may be 100% pruned."""
    torch.manual_seed(0)

    # Deliberately create a model where layer A is tiny and layer B is large.
    # All weights in A are small; all weights in B are large.
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False),   # tiny: 4 weights — set to near-zero
        nn.Linear(32, 32, bias=False), # large: 1024 weights — set to large values
    )
    with torch.no_grad():
        model[0].weight.data.fill_(0.001)   # all small magnitudes
        model[1].weight.data.fill_(10.0)    # all large magnitudes

    config = LotteryConfig(pruning_rate=0.004, global_pruning=True)
    p = LotteryTicketPruner(model, config)
    p.save_initial_weights()

    # Prune globally: threshold derived from all 1028 weights together.
    # The 4 tiny weights should all fall below the threshold.
    masks = p.compute_masks(pruning_rate=0.004)
    p.apply_masks(masks)

    tiny_weight = model[0].weight.data
    assert (tiny_weight == 0).all(), (
        "Global pruning should have zeroed all weights of the small layer "
        f"(magnitudes: {model[0].weight.data})"
    )


# ── Test 11: run_round increases sparsity by ≈ pruning_rate of remaining ───────

def test_run_round_sparsity_delta(pruner_with_saved):
    """Sparsity increase per round should be ≈ pruning_rate * (1 - sparsity_before)."""
    rate = pruner_with_saved.config.pruning_rate  # 0.20

    before = pruner_with_saved.sparsity()
    pruner_with_saved.run_round()
    after = pruner_with_saved.sparsity()

    expected_delta = rate * (1.0 - before)
    actual_delta = after - before

    # Allow ±5 % tolerance for threshold ties at boundaries.
    assert abs(actual_delta - expected_delta) < 0.05, (
        f"Expected sparsity delta ≈ {expected_delta:.4f}, got {actual_delta:.4f}"
    )


# ── Test 12: bias parameters are never modified ───────────────────────────────

def test_bias_params_never_pruned(pruner_with_saved):
    """Bias parameters must be completely untouched by run_round()."""
    # Record original bias values.
    original_biases = {
        name: param.data.clone()
        for name, param in pruner_with_saved.model.named_parameters()
        if name.split(".")[-1] == "bias"
    }

    assert len(original_biases) > 0, "Fixture model must have bias parameters."

    pruner_with_saved.run_round()

    for name, orig in original_biases.items():
        current = dict(pruner_with_saved.model.named_parameters())[name].data
        assert torch.equal(current, orig), (
            f"Bias '{name}' was modified by run_round() — biases must not be pruned."
        )
