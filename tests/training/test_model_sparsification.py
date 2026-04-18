"""
Tests for src/training/model_sparsification.py

Tiny model: d_model=16, n_layers=2, B=2, T=4, vocab_size=32
Pure PyTorch — no heavy dependencies.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest

from src.training.model_sparsification import (
    MagnitudePruner,
    MovementPruner,
    SparsityScheduler,
    HeadPruner,
    LoRARegrowth,
    PruningTrainer,
    SparsificationConfig,
)

# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------

D_MODEL = 16
N_LAYERS = 2
VOCAB_SIZE = 32
B, T = 2, 4


class TinyTransformer(nn.Module):
    """Minimal 2-layer transformer with multi-head attention for testing."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        vocab_size: int = VOCAB_SIZE,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.ffns = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)  # [B, T, d_model]
        for attn, ffn in zip(self.layers, self.ffns):
            x, _ = attn(x, x, x, need_weights=False)
            x = ffn(x)
        return self.head(x)  # [B, T, vocab_size]


def make_model() -> TinyTransformer:
    torch.manual_seed(42)
    return TinyTransformer()


def make_input():
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    labels = torch.randint(0, VOCAB_SIZE, (B, T))
    return input_ids, labels


# ---------------------------------------------------------------------------
# MagnitudePruner tests
# ---------------------------------------------------------------------------


def test_magnitude_compute_thresholds_returns_dict():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.5)
    thresholds = pruner.compute_thresholds(model)
    assert isinstance(thresholds, dict)
    assert len(thresholds) > 0
    # Keys should be subset of model parameter names
    param_names = {n for n, p in model.named_parameters() if p.dim() >= 2}
    for key in thresholds:
        assert key in param_names


def test_magnitude_create_masks_binary():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.5)
    masks = pruner.create_masks(model)
    assert isinstance(masks, dict)
    assert len(masks) > 0
    for name, mask in masks.items():
        unique_vals = mask.unique().tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Non-binary value {v} in mask for {name}"


def test_magnitude_apply_masks_zeros_pruned_weights():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.5)
    masks = pruner.create_masks(model)
    pruner.apply_masks(model, masks)
    param_dict = dict(model.named_parameters())
    for name, mask in masks.items():
        weight = param_dict[name].data
        # Wherever mask == 0, weight must be 0
        pruned_positions = mask == 0
        assert (weight[pruned_positions] == 0).all(), (
            f"Non-zero weights at pruned positions in {name}"
        )


def test_magnitude_actual_sparsity_approx_target():
    model = make_model()
    target = 0.5
    pruner = MagnitudePruner(model, target_sparsity=target)
    masks = pruner.create_masks(model)
    pruner.apply_masks(model, masks)
    sparsity = pruner.actual_sparsity(model)
    # Allow ±5 percentage points tolerance
    assert abs(sparsity - target) < 0.05, (
        f"Sparsity {sparsity:.3f} not close to target {target}"
    )


# ---------------------------------------------------------------------------
# MovementPruner tests
# ---------------------------------------------------------------------------


def _do_movement_step(model, pruner):
    """Run one forward/backward to populate gradients, then update scores."""
    input_ids, labels = make_input()
    logits = model(input_ids)
    B2, T2, V = logits.shape
    loss = nn.functional.cross_entropy(logits.reshape(B2 * T2, V), labels.reshape(B2 * T2))
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    pruner.update_scores(model, optimizer)
    optimizer.zero_grad()


def test_movement_update_scores_populates():
    model = make_model()
    pruner = MovementPruner(model, target_sparsity=0.5, warmup_steps=1)
    _do_movement_step(model, pruner)
    assert len(pruner.scores) > 0
    # At least some scores should be non-zero after a real backward pass
    total_nonzero = sum(s.abs().sum().item() for s in pruner.scores.values())
    assert total_nonzero > 0, "All movement scores are zero after backward pass"


def test_movement_scores_non_negative():
    model = make_model()
    pruner = MovementPruner(model, target_sparsity=0.5)
    _do_movement_step(model, pruner)
    for name, score in pruner.scores.items():
        assert (score >= 0).all(), f"Negative movement score in {name}"


def test_movement_create_masks_binary():
    model = make_model()
    pruner = MovementPruner(model, target_sparsity=0.5)
    _do_movement_step(model, pruner)
    masks = pruner.create_masks(threshold_percentile=50.0)
    assert isinstance(masks, dict)
    assert len(masks) > 0
    for name, mask in masks.items():
        unique_vals = mask.unique().tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Non-binary mask value {v} in {name}"


# ---------------------------------------------------------------------------
# SparsityScheduler tests
# ---------------------------------------------------------------------------


def test_scheduler_at_begin_step_returns_initial():
    sched = SparsityScheduler(
        initial_sparsity=0.1, final_sparsity=0.9, begin_step=0, end_step=1000
    )
    assert sched.current_sparsity(0) == pytest.approx(0.1, abs=1e-6)


def test_scheduler_at_end_step_returns_final():
    sched = SparsityScheduler(
        initial_sparsity=0.0, final_sparsity=0.9, begin_step=0, end_step=1000
    )
    assert sched.current_sparsity(1000) == pytest.approx(0.9, abs=1e-6)


def test_scheduler_monotonically_increasing():
    sched = SparsityScheduler(
        initial_sparsity=0.0, final_sparsity=0.9, begin_step=0, end_step=1000
    )
    steps = list(range(0, 1001, 50))
    sparsities = [sched.current_sparsity(s) for s in steps]
    for i in range(1, len(sparsities)):
        assert sparsities[i] >= sparsities[i - 1] - 1e-9, (
            f"Non-monotone at step {steps[i]}: {sparsities[i-1]:.4f} → {sparsities[i]:.4f}"
        )


def test_scheduler_should_prune_at_multiples_of_freq():
    sched = SparsityScheduler(begin_step=0, end_step=1000)
    freq = 100
    # Should prune at multiples
    for step in [0, 100, 200, 500, 1000]:
        assert sched.should_prune(step, freq=freq), f"Expected prune at step {step}"
    # Should NOT prune at non-multiples
    for step in [50, 150, 333]:
        assert not sched.should_prune(step, freq=freq), (
            f"Unexpected prune at step {step}"
        )


def test_scheduler_should_not_prune_outside_range():
    sched = SparsityScheduler(begin_step=100, end_step=500)
    # Step before begin_step
    assert not sched.should_prune(0, freq=100)
    # Step after end_step
    assert not sched.should_prune(600, freq=100)


# ---------------------------------------------------------------------------
# HeadPruner tests
# ---------------------------------------------------------------------------


def _make_attn_weights(n_layers=2, n_heads=2, B=2, T=4):
    """Random attention weights [B, n_heads, T, T], softmax-normalised."""
    weights = []
    for _ in range(n_layers):
        raw = torch.rand(B, n_heads, T, T)
        attn = raw / raw.sum(dim=-1, keepdim=True)
        weights.append(attn)
    return weights


def test_head_pruner_compute_importance_returns_dict():
    model = make_model()
    pruner = HeadPruner(model, target_head_sparsity=0.5)
    attn_weights = _make_attn_weights()
    importance = pruner.compute_head_importance(attn_weights)
    assert isinstance(importance, dict)
    assert len(importance) == len(attn_weights)
    for key, val in importance.items():
        assert isinstance(val, torch.Tensor)
        assert val.dim() == 1  # [n_heads]


def test_head_pruner_prune_heads_correct_count():
    model = make_model()
    pruner = HeadPruner(model, target_head_sparsity=0.5)
    attn_weights = _make_attn_weights(n_layers=2, n_heads=2)
    importance = pruner.compute_head_importance(attn_weights)
    n_to_prune = 2
    to_prune = pruner.prune_heads(importance, n_to_prune=n_to_prune)
    total_pruned = sum(len(v) for v in to_prune.values())
    assert total_pruned == n_to_prune, (
        f"Expected {n_to_prune} pruned heads, got {total_pruned}"
    )


# ---------------------------------------------------------------------------
# PruningTrainer tests
# ---------------------------------------------------------------------------


def test_pruning_trainer_train_step_returns_finite_loss_and_sparsity():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.5)
    scheduler = SparsityScheduler(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=200
    )
    trainer = PruningTrainer(model, pruner, scheduler, lr=1e-3)
    input_ids, labels = make_input()
    loss, sparsity = trainer.train_step(input_ids, labels, step=0)
    assert math.isfinite(loss), f"Loss is not finite: {loss}"
    assert math.isfinite(sparsity), f"Sparsity is not finite: {sparsity}"
    assert 0.0 <= sparsity <= 1.0, f"Sparsity out of range: {sparsity}"


def test_pruning_trainer_sparsity_increases_over_steps():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.9)
    scheduler = SparsityScheduler(
        initial_sparsity=0.0, final_sparsity=0.9, begin_step=0, end_step=200,
    )
    trainer = PruningTrainer(model, pruner, scheduler, lr=1e-4)
    input_ids, labels = make_input()

    sparsities = []
    # Run steps at prune checkpoints: 0, 100, 200
    for step in [0, 100, 200]:
        _, sp = trainer.train_step(input_ids, labels, step=step)
        sparsities.append(sp)

    # Sparsity at last checkpoint should be strictly greater than at first
    assert sparsities[-1] > sparsities[0], (
        f"Sparsity did not increase: {sparsities}"
    )


# ---------------------------------------------------------------------------
# SparsificationConfig test
# ---------------------------------------------------------------------------


def test_sparsification_config_defaults():
    cfg = SparsificationConfig()
    assert cfg.target_sparsity == 0.9
    assert cfg.initial_sparsity == 0.0
    assert cfg.begin_step == 0
    assert cfg.end_step == 500
    assert cfg.prune_freq == 50
    assert cfg.target_head_sparsity == 0.5
    assert cfg.rank == 4
    assert cfg.lr == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# LoRARegrowth smoke test (bonus — counts toward 16+)
# ---------------------------------------------------------------------------


def test_lora_regrowth_returns_ab_pairs():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.5)
    masks = pruner.create_masks(model)
    pruner.apply_masks(model, masks)

    lora = LoRARegrowth(model, rank=4)
    lora_params = lora.regrow_pruned(masks)

    assert isinstance(lora_params, dict)
    for name, (A, B) in lora_params.items():
        assert isinstance(A, nn.Parameter)
        assert isinstance(B, nn.Parameter)
        assert A.dim() == 2  # [rank, d_in]
        assert B.dim() == 2  # [d_out, rank]


def test_lora_merge_and_reprune_keeps_masks():
    model = make_model()
    pruner = MagnitudePruner(model, target_sparsity=0.5)
    masks = pruner.create_masks(model)
    pruner.apply_masks(model, masks)

    lora = LoRARegrowth(model, rank=2)
    lora_params = lora.regrow_pruned(masks)
    lora.merge_and_reprune(model, lora_params, masks)

    # After re-pruning, masked positions must still be zero
    param_dict = dict(model.named_parameters())
    for name, mask in masks.items():
        weight = param_dict[name].data
        assert (weight[mask == 0] == 0).all(), (
            f"Non-zero weight at pruned position after merge_and_reprune in {name}"
        )
