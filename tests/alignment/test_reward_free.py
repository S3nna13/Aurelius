"""Tests for src/alignment/reward_free.py

Covers:
  1.  RewardFreeConfig defaults correct
  2.  SLiCLoss.hinge_loss returns shape (batch,)
  3.  Hinge loss is 0 when chosen - rejected > delta
  4.  Hinge loss is positive when chosen - rejected < delta
  5.  SLiCLoss.forward returns scalar total loss
  6.  Metrics dict has all required keys
  7.  Accuracy > 0.5 when chosen_logps > rejected_logps
  8.  ULMATrainer.compute_sequence_logps returns (batch,) tensor
  9.  calibration_loss returns scalar
  10. contrastive_loss returns scalar (always <= 0 for log-prob)
  11. train_step returns dict with loss key
  12. Gradient flows through SLiC loss (backward works)
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.reward_free import (
    RewardFreeConfig,
    SLiCLoss,
    ULMATrainer,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

BATCH = 4
VOCAB = 32
SEQ = 8


class _TinyLM(nn.Module):
    """Minimal language model that returns fixed-seed logits."""

    def __init__(self, vocab: int = VOCAB, seed: int = 0) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, 16)
        self.proj = nn.Linear(16, vocab, bias=False)
        self._seed = seed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T, V)
        x = self.embed(input_ids)  # (B, T, 16)
        return self.proj(x)  # (B, T, V)


def _make_ids(b: int = BATCH, t: int = SEQ, vocab: int = VOCAB) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, vocab, (b, t))


def _make_labels(b: int = BATCH, t: int = SEQ, vocab: int = VOCAB) -> torch.Tensor:
    """Labels with the first half masked as prompt (-100)."""
    torch.manual_seed(13)
    labels = torch.randint(0, vocab, (b, t))
    labels[:, : t // 2] = -100  # mask prompt tokens
    return labels


def _make_logps(b: int = BATCH, high: bool = True) -> torch.Tensor:
    """Return log-prob-like tensors (negative values)."""
    torch.manual_seed(1 if high else 2)
    base = torch.randn(b) - 5.0  # keeps values in a plausible log-prob range
    return base


# ---------------------------------------------------------------------------
# Test 1 — RewardFreeConfig defaults
# ---------------------------------------------------------------------------


def test_reward_free_config_defaults():
    cfg = RewardFreeConfig()
    assert cfg.method == "slic"
    assert cfg.beta == pytest.approx(0.1)
    assert cfg.margin == pytest.approx(1.0)
    assert cfg.lambda_reg == pytest.approx(1.0)
    assert cfg.delta == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2 — SLiCLoss.hinge_loss shape
# ---------------------------------------------------------------------------


def test_slic_hinge_loss_shape():
    criterion = SLiCLoss()
    chosen = _make_logps(BATCH, high=True)
    rejected = _make_logps(BATCH, high=False)
    out = criterion.hinge_loss(chosen, rejected)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3 — Hinge loss is 0 when chosen - rejected > delta
# ---------------------------------------------------------------------------


def test_slic_hinge_zero_when_margin_exceeded():
    delta = 1.0
    criterion = SLiCLoss(delta=delta)
    # chosen is clearly above rejected by more than delta
    chosen = torch.tensor([-1.0, -2.0, -3.0])
    rejected = torch.tensor([-4.0, -5.0, -6.0])  # gap = 3 > delta=1
    out = criterion.hinge_loss(chosen, rejected)
    assert torch.all(out == 0.0), f"Expected all zeros, got {out}"


# ---------------------------------------------------------------------------
# Test 4 — Hinge loss is positive when chosen - rejected < delta
# ---------------------------------------------------------------------------


def test_slic_hinge_positive_when_margin_insufficient():
    delta = 5.0
    criterion = SLiCLoss(delta=delta)
    # gap is only 0.5 < delta=5
    chosen = torch.tensor([-3.0, -3.0])
    rejected = torch.tensor([-3.5, -3.5])
    out = criterion.hinge_loss(chosen, rejected)
    assert torch.all(out > 0.0), f"Expected positive values, got {out}"


# ---------------------------------------------------------------------------
# Test 5 — SLiCLoss.forward returns scalar total loss
# ---------------------------------------------------------------------------


def test_slic_forward_returns_scalar():
    criterion = SLiCLoss()
    chosen = _make_logps(BATCH, high=True)
    rejected = _make_logps(BATCH, high=False)
    loss, _ = criterion(chosen, rejected)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss should be finite"


# ---------------------------------------------------------------------------
# Test 6 — Metrics dict has all required keys
# ---------------------------------------------------------------------------


def test_slic_forward_metrics_keys():
    criterion = SLiCLoss()
    chosen = _make_logps(BATCH)
    rejected = _make_logps(BATCH)
    _, metrics = criterion(chosen, rejected)
    required_keys = {"hinge_loss", "reg_loss", "accuracy"}
    assert required_keys.issubset(set(metrics.keys())), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Accuracy > 0.5 when chosen_logps consistently > rejected_logps
# ---------------------------------------------------------------------------


def test_slic_accuracy_when_chosen_dominates():
    criterion = SLiCLoss()
    # Construct tensors so chosen is always greater than rejected
    chosen = torch.tensor([-1.0, -1.5, -2.0, -2.5, -3.0])
    rejected = chosen - 2.0  # rejected is 2 units below chosen everywhere
    _, metrics = criterion(chosen, rejected)
    assert metrics["accuracy"].item() > 0.5, (
        f"Expected accuracy > 0.5, got {metrics['accuracy'].item()}"
    )


# ---------------------------------------------------------------------------
# Test 8 — ULMATrainer.compute_sequence_logps returns (batch,)
# ---------------------------------------------------------------------------


def test_ulma_compute_sequence_logps_shape():
    torch.manual_seed(0)
    policy = _TinyLM()
    ref = copy.deepcopy(policy)
    trainer = ULMATrainer(policy, ref, method="slic")

    input_ids = _make_ids()
    labels = _make_labels()
    logps = trainer.compute_sequence_logps(policy, input_ids, labels)
    assert logps.shape == (BATCH,), f"Expected ({BATCH},), got {logps.shape}"


# ---------------------------------------------------------------------------
# Test 9 — calibration_loss returns scalar
# ---------------------------------------------------------------------------


def test_ulma_calibration_loss_scalar():
    torch.manual_seed(0)
    policy = _TinyLM()
    ref = copy.deepcopy(policy)
    trainer = ULMATrainer(policy, ref, method="calibration", margin=1.0)

    chosen = _make_logps(BATCH, high=True)
    rejected = _make_logps(BATCH, high=False)
    ref_chosen = _make_logps(BATCH)
    ref_rejected = _make_logps(BATCH)

    loss = trainer.calibration_loss(chosen, rejected, ref_chosen, ref_rejected)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 10 — contrastive_loss returns scalar and is always <= 0
# ---------------------------------------------------------------------------


def test_ulma_contrastive_loss_scalar_and_non_positive():
    torch.manual_seed(0)
    policy = _TinyLM()
    ref = copy.deepcopy(policy)
    trainer = ULMATrainer(policy, ref, method="contrastive")

    # Test with various random seeds to verify the (<= 0) property holds
    for seed in range(5):
        torch.manual_seed(seed)
        chosen = torch.randn(BATCH) - 5.0
        rejected = torch.randn(BATCH) - 5.0
        loss = trainer.contrastive_loss(chosen, rejected)
        assert loss.ndim == 0, f"Expected scalar, got {loss.shape}"
        assert loss.item() <= 0.0 + 1e-6, (
            f"contrastive_loss should be <= 0 (it is a log-prob), got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Test 11 — train_step returns dict with 'loss' key
# ---------------------------------------------------------------------------


def test_ulma_train_step_returns_loss_key():
    for method in ("slic", "calibration", "contrastive"):
        torch.manual_seed(0)
        policy = _TinyLM()
        ref = copy.deepcopy(policy)
        for p in ref.parameters():
            p.requires_grad_(False)

        trainer = ULMATrainer(policy, ref, method=method)

        chosen_ids = _make_ids()
        rejected_ids = _make_ids()
        chosen_labels = _make_labels()
        rejected_labels = _make_labels()

        result = trainer.train_step(chosen_ids, rejected_ids, chosen_labels, rejected_labels)
        assert "loss" in result, f"method={method}: 'loss' key missing from result"
        assert torch.isfinite(result["loss"]), f"method={method}: loss is not finite"


# ---------------------------------------------------------------------------
# Test 12 — Gradients flow through SLiC loss
# ---------------------------------------------------------------------------


def test_slic_backward_produces_gradients():
    criterion = SLiCLoss(delta=1.0, lambda_reg=0.5)

    chosen = torch.tensor([-2.0, -3.0, -4.0], requires_grad=True)
    rejected = torch.tensor([-5.0, -4.5, -2.5], requires_grad=True)
    ref_logps = torch.tensor([-2.5, -3.5, -4.5])

    loss, _ = criterion(chosen, rejected, ref_logps=ref_logps)
    loss.backward()

    assert chosen.grad is not None, "No gradient on chosen_logps"
    assert rejected.grad is not None, "No gradient on rejected_logps"
    assert torch.isfinite(chosen.grad).all(), "chosen.grad contains non-finite values"
    assert torch.isfinite(rejected.grad).all(), "rejected.grad contains non-finite values"
