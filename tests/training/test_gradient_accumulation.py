"""Tests for gradient_accumulation.py.

Tiny configuration throughout:
  - 2-layer MLP: 16 → 16 → 16
  - micro_batch_size = 2
  - virtual_batch_size = 4
  - seq_len = 8, batch = 4
"""

import copy
import math

import torch
import torch.nn as nn

from aurelius.training.gradient_accumulation import (
    AccumulationStats,
    GradientAccumulator,
    LossScaler,
    MicroBatchSplitter,
    VirtualBatchTrainer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 16
BATCH = 4
MICRO = 2
VIRTUAL = 4
SEQ_LEN = 8


def make_mlp() -> nn.Module:
    """Tiny 2-layer MLP that maps (batch, seq_len) → (batch, seq_len, vocab)."""

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(VOCAB, D_MODEL)
            self.fc1 = nn.Linear(D_MODEL, D_MODEL)
            self.fc2 = nn.Linear(D_MODEL, VOCAB)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
            h = self.embed(x)
            h = torch.relu(self.fc1(h))
            return self.fc2(h)

    return TinyMLP()


def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels) of shape (BATCH, SEQ_LEN)."""
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return ids, labels


# ---------------------------------------------------------------------------
# MicroBatchSplitter tests
# ---------------------------------------------------------------------------

def test_splitter_correct_n_micro_batches():
    """split() returns the expected number of micro-batches."""
    splitter = MicroBatchSplitter(micro_batch_size=MICRO)
    ids, labels = make_batch()
    micro_batches = splitter.split(ids, labels)
    assert len(micro_batches) == BATCH // MICRO


def test_splitter_shapes_correct():
    """Each micro-batch has shape (micro_batch_size, SEQ_LEN) or smaller for last."""
    splitter = MicroBatchSplitter(micro_batch_size=MICRO)
    ids, labels = make_batch()
    micro_batches = splitter.split(ids, labels)
    for mb_ids, mb_labels in micro_batches:
        assert mb_ids.shape[1] == SEQ_LEN
        assert mb_labels.shape[1] == SEQ_LEN
        assert mb_ids.shape[0] <= MICRO
        assert mb_labels.shape[0] <= MICRO


def test_splitter_last_micro_batch_smaller():
    """Last micro-batch is smaller when batch_size is not divisible by micro_batch_size."""
    splitter = MicroBatchSplitter(micro_batch_size=3)
    ids = torch.zeros(7, SEQ_LEN, dtype=torch.long)
    labels = torch.zeros(7, SEQ_LEN, dtype=torch.long)
    micro_batches = splitter.split(ids, labels)
    # ceil(7/3) = 3 micro-batches; last has size 1
    assert len(micro_batches) == 3
    assert micro_batches[-1][0].shape[0] == 1


def test_splitter_n_micro_batches_ceil():
    """n_micro_batches returns ceil(batch / micro)."""
    splitter = MicroBatchSplitter(micro_batch_size=3)
    assert splitter.n_micro_batches(7) == 3
    assert splitter.n_micro_batches(6) == 2
    assert splitter.n_micro_batches(1) == 1


def test_splitter_effective_batch_size():
    """effective_batch_size = micro_batch_size * n_accumulation_steps."""
    splitter = MicroBatchSplitter(micro_batch_size=MICRO)
    assert splitter.effective_batch_size(VIRTUAL // MICRO) == MICRO * (VIRTUAL // MICRO)


# ---------------------------------------------------------------------------
# GradientAccumulator tests
# ---------------------------------------------------------------------------

def test_accumulator_gradients_accumulate():
    """Gradients should accumulate across multiple step() calls."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    accum = GradientAccumulator(model, optimizer, n_accumulation_steps=2)

    ids, labels = make_batch()
    out1 = model(ids[:MICRO])
    loss1 = nn.functional.cross_entropy(out1.view(-1, VOCAB), labels[:MICRO].view(-1))
    accum.step(loss1, is_last=False)
    norm_after_1 = accum.grad_norm()

    out2 = model(ids[MICRO:])
    loss2 = nn.functional.cross_entropy(out2.view(-1, VOCAB), labels[MICRO:].view(-1))
    accum.step(loss2, is_last=False)
    norm_after_2 = accum.grad_norm()

    # Gradient norm should be non-negative and grow (or stay) with more steps
    assert norm_after_1 >= 0.0
    assert norm_after_2 >= norm_after_1


def test_accumulator_is_last_triggers_optimizer_step():
    """is_last=True should change model parameters."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.zero_grad()
    accum = GradientAccumulator(model, optimizer, n_accumulation_steps=2)

    params_before = [p.clone() for p in model.parameters()]

    ids, labels = make_batch()
    out1 = model(ids[:MICRO])
    loss1 = nn.functional.cross_entropy(out1.view(-1, VOCAB), labels[:MICRO].view(-1))
    accum.step(loss1, is_last=False)

    out2 = model(ids[MICRO:])
    loss2 = nn.functional.cross_entropy(out2.view(-1, VOCAB), labels[MICRO:].view(-1))
    accum.step(loss2, is_last=True)

    params_after = list(model.parameters())
    changed = any(
        not torch.equal(pb, pa) for pb, pa in zip(params_before, params_after)
    )
    assert changed, "Parameters should change after is_last=True step"


def test_accumulator_not_last_does_not_trigger_optimizer_step():
    """is_last=False should NOT change model parameters."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.zero_grad()
    accum = GradientAccumulator(model, optimizer, n_accumulation_steps=4)

    params_before = [p.clone() for p in model.parameters()]

    ids, labels = make_batch()
    out = model(ids[:MICRO])
    loss = nn.functional.cross_entropy(out.view(-1, VOCAB), labels[:MICRO].view(-1))
    accum.step(loss, is_last=False)

    params_after = list(model.parameters())
    unchanged = all(
        torch.equal(pb, pa) for pb, pa in zip(params_before, params_after)
    )
    assert unchanged, "Parameters must not change when is_last=False"


def test_accumulator_grad_norm_non_negative():
    """grad_norm() should always be >= 0."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    accum = GradientAccumulator(model, optimizer, n_accumulation_steps=2)

    ids, labels = make_batch()
    out = model(ids[:MICRO])
    loss = nn.functional.cross_entropy(out.view(-1, VOCAB), labels[:MICRO].view(-1))
    accum.step(loss, is_last=False)

    norm = accum.grad_norm()
    assert norm >= 0.0
    assert math.isfinite(norm)


# ---------------------------------------------------------------------------
# VirtualBatchTrainer tests
# ---------------------------------------------------------------------------

def test_virtual_trainer_train_step_keys():
    """train_step must return dict with loss, grad_norm, n_micro_batches."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = VirtualBatchTrainer(model, optimizer, VIRTUAL, MICRO)
    ids, labels = make_batch()
    result = trainer.train_step(ids, labels)
    assert "loss" in result
    assert "grad_norm" in result
    assert "n_micro_batches" in result


def test_virtual_trainer_train_step_loss_finite():
    """train_step loss should be a finite float."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = VirtualBatchTrainer(model, optimizer, VIRTUAL, MICRO)
    ids, labels = make_batch()
    result = trainer.train_step(ids, labels)
    assert math.isfinite(result["loss"])


def test_virtual_trainer_n_accumulation_steps():
    """n_accumulation_steps should equal virtual / micro."""
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = VirtualBatchTrainer(model, optimizer, VIRTUAL, MICRO)
    assert trainer.n_accumulation_steps == VIRTUAL // MICRO


def test_virtual_trainer_verify_equivalence():
    """verify_equivalence should return True for a small model within tolerance."""
    torch.manual_seed(0)
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = VirtualBatchTrainer(model, optimizer, VIRTUAL, MICRO)

    # Make reference model with identical weights
    ref_model = copy.deepcopy(model)

    ids, labels = make_batch()
    result = trainer.verify_equivalence(ref_model, ids, labels, tol=1e-4)
    assert result, "Gradient accumulation should match full-batch gradients within tol"


# ---------------------------------------------------------------------------
# LossScaler tests
# ---------------------------------------------------------------------------

def test_loss_scaler_scale():
    """scale() should return loss * current_scale."""
    scaler = LossScaler(init_scale=256.0)
    loss = torch.tensor(2.0)
    scaled = scaler.scale(loss)
    assert abs(scaled.item() - 512.0) < 1e-5


def test_loss_scaler_update_found_inf_decreases_scale():
    """update(found_inf=True) should decrease the scale."""
    scaler = LossScaler(init_scale=256.0, backoff_factor=0.5)
    scaler.update(found_inf=True)
    assert scaler.get_scale() == 128.0


def test_loss_scaler_update_no_inf_for_growth_interval_increases_scale():
    """After growth_interval steps without inf, scale should increase."""
    scaler = LossScaler(init_scale=256.0, growth_factor=2.0, growth_interval=5)
    for _ in range(5):
        scaler.update(found_inf=False)
    assert scaler.get_scale() == 512.0


# ---------------------------------------------------------------------------
# AccumulationStats tests
# ---------------------------------------------------------------------------

def test_accumulation_stats_mean_loss():
    """mean_loss should return the mean of all recorded losses."""
    stats = AccumulationStats()
    stats.record_micro_batch(loss=1.0)
    stats.record_micro_batch(loss=3.0)
    assert abs(stats.mean_loss() - 2.0) < 1e-9


def test_accumulation_stats_reset_clears_history():
    """After reset(), mean_loss should return 0.0 (empty history)."""
    stats = AccumulationStats()
    stats.record_micro_batch(loss=5.0)
    stats.record_micro_batch(loss=7.0)
    stats.reset()
    assert stats.mean_loss() == 0.0
