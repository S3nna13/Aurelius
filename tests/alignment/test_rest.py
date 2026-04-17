"""Tests for src/alignment/rest.py — ReST (Reinforced Self-Training).

Covers the 14-test floor specified in the implementation brief.
Pure PyTorch only; no scipy, sklearn, HuggingFace, etc.
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from src.alignment.rest import (
    ReSTDataset,
    ReSTLoss,
    ReSTThresholdSchedule,
    ReSTTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 32


def _make_dataset(n: int = 6, seed: int = 0) -> ReSTDataset:
    """Create a small dataset with predictable rewards 0, 1/(n-1), …, 1."""
    torch.manual_seed(seed)
    prompts = [torch.randint(0, VOCAB, (4,)) for _ in range(n)]
    completions = [torch.randint(0, VOCAB, (6,)) for _ in range(n)]
    rewards = [i / max(n - 1, 1) for i in range(n)]
    return ReSTDataset(prompts, completions, rewards)


def _log_probs(batch: int, seq: int | None = None, seed: int = 42) -> torch.Tensor:
    """Return log-probabilities in [-5, 0) — same range as real log-softmax."""
    torch.manual_seed(seed)
    if seq is None:
        return -torch.rand(batch) * 5.0
    return -torch.rand(batch, seq) * 5.0


def _rewards(batch: int, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(batch)


# ---------------------------------------------------------------------------
# Test 1 — ReSTLoss output is a scalar
# ---------------------------------------------------------------------------

def test_rest_loss_output_is_scalar():
    loss_fn = ReSTLoss()
    lp = _log_probs(8)
    r = _rewards(8)
    loss, _ = loss_fn(lp, r, threshold=0.4)
    assert loss.shape == torch.Size([]), "loss must be a 0-d scalar tensor"


# ---------------------------------------------------------------------------
# Test 2 — Gradients are finite on accepted samples
# ---------------------------------------------------------------------------

def test_rest_loss_gradients_finite():
    loss_fn = ReSTLoss()
    lp = _log_probs(8).requires_grad_(True)
    r = _rewards(8)
    loss, n_accepted = loss_fn(lp, r, threshold=0.2)
    assert n_accepted > 0, "expected some accepted samples with threshold=0.2"
    loss.backward()
    assert lp.grad is not None
    assert torch.isfinite(lp.grad).all(), "gradients must be finite"


# ---------------------------------------------------------------------------
# Test 3 — All samples below threshold → loss is zero (NaN-free)
# ---------------------------------------------------------------------------

def test_rest_loss_all_rejected_is_zero():
    loss_fn = ReSTLoss()
    lp = _log_probs(6)
    r = torch.zeros(6)            # all rewards are 0
    loss, n_accepted = loss_fn(lp, r, threshold=1.0)  # threshold=1.0 → none pass
    assert n_accepted == 0
    assert not torch.isnan(loss), "loss must not be NaN when all samples rejected"
    assert loss.item() == 0.0, "loss must be 0.0 when all samples rejected"


# ---------------------------------------------------------------------------
# Test 4 — n_accepted is the correct count
# ---------------------------------------------------------------------------

def test_rest_loss_n_accepted_correct():
    loss_fn = ReSTLoss()
    r = torch.tensor([0.1, 0.5, 0.6, 0.9, 0.3])
    lp = -torch.ones(5)
    threshold = 0.5
    expected = int((r >= threshold).sum().item())
    _, n_accepted = loss_fn(lp, r, threshold=threshold)
    assert n_accepted == expected


# ---------------------------------------------------------------------------
# Test 5 — Higher threshold → fewer (or equal) accepted samples
# ---------------------------------------------------------------------------

def test_higher_threshold_fewer_accepted():
    loss_fn = ReSTLoss()
    r = _rewards(16)
    lp = _log_probs(16)
    _, n_low = loss_fn(lp, r, threshold=0.1)
    _, n_high = loss_fn(lp, r, threshold=0.8)
    assert n_high <= n_low, "higher threshold must not increase accepted count"


# ---------------------------------------------------------------------------
# Test 6 — ReSTThresholdSchedule: threshold increases with iteration k
# ---------------------------------------------------------------------------

def test_threshold_schedule_increases_with_k():
    sched = ReSTThresholdSchedule(base_percentile=40.0, increment=10.0)
    rewards = torch.linspace(0.0, 1.0, 100)
    thresholds = [sched.threshold_for_iteration(k, rewards) for k in range(5)]
    for i in range(len(thresholds) - 1):
        assert thresholds[i] <= thresholds[i + 1], (
            f"threshold at k={i} ({thresholds[i]:.4f}) must be <= "
            f"k={i+1} ({thresholds[i+1]:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 7 — ReSTThresholdSchedule: threshold at k=0 is the 50th percentile
# ---------------------------------------------------------------------------

def test_threshold_schedule_k0_is_median():
    sched = ReSTThresholdSchedule(base_percentile=50.0, increment=10.0)
    rewards = torch.linspace(0.0, 1.0, 101)   # 0.00, 0.01, … 1.00
    tau_0 = sched.threshold_for_iteration(0, rewards)
    expected = torch.quantile(rewards.float(), 0.5).item()
    assert math.isclose(tau_0, expected, rel_tol=1e-5), (
        f"k=0 threshold {tau_0} != 50th-percentile {expected}"
    )


# ---------------------------------------------------------------------------
# Test 8 — ReSTDataset.filter: returns subset where reward >= threshold
# ---------------------------------------------------------------------------

def test_dataset_filter_correct_subset():
    ds = _make_dataset(n=10)
    threshold = 0.5
    filtered = ds.filter(threshold)
    for r in filtered.rewards:
        assert r >= threshold, f"reward {r} is below threshold {threshold}"
    expected_count = sum(1 for r in ds.rewards if r >= threshold)
    assert len(filtered) == expected_count


# ---------------------------------------------------------------------------
# Test 9 — ReSTDataset.filter: no samples when threshold > max reward
# ---------------------------------------------------------------------------

def test_dataset_filter_all_rejected():
    ds = _make_dataset(n=8)
    max_r = max(ds.rewards)
    filtered = ds.filter(max_r + 1.0)
    assert len(filtered) == 0, "expected empty dataset when threshold > max reward"


# ---------------------------------------------------------------------------
# Test 10 — ReSTDataset.filter: all samples when threshold <= min reward
# ---------------------------------------------------------------------------

def test_dataset_filter_all_accepted():
    ds = _make_dataset(n=8)
    min_r = min(ds.rewards)
    filtered = ds.filter(min_r - 1.0)
    assert len(filtered) == len(ds), (
        "expected full dataset when threshold <= min reward"
    )


# ---------------------------------------------------------------------------
# Test 11 — ReSTLoss determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_rest_loss_determinism():
    torch.manual_seed(99)
    lp1 = -torch.rand(10) * 5.0
    r1 = torch.rand(10)

    torch.manual_seed(99)
    lp2 = -torch.rand(10) * 5.0
    r2 = torch.rand(10)

    loss_fn = ReSTLoss()
    loss_a, _ = loss_fn(lp1, r1, threshold=0.3)
    loss_b, _ = loss_fn(lp2, r2, threshold=0.3)
    assert torch.isclose(loss_a, loss_b), "loss must be deterministic given same seed"


# ---------------------------------------------------------------------------
# Test 12 — No NaN/Inf on normal inputs
# ---------------------------------------------------------------------------

def test_rest_loss_no_nan_inf():
    loss_fn = ReSTLoss()
    lp = _log_probs(32, seq=16)
    r = _rewards(32)
    loss, _ = loss_fn(lp, r, threshold=0.3)
    assert torch.isfinite(loss), f"loss must be finite; got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 13 — ReSTTrainer.compute_loss matches ReSTLoss directly
# ---------------------------------------------------------------------------

def test_trainer_compute_loss_matches_rest_loss():
    loss_fn = ReSTLoss()
    trainer = ReSTTrainer(loss_fn=loss_fn)

    lp = _log_probs(12)
    r = _rewards(12)
    threshold = 0.4

    trainer_loss = trainer.compute_loss(lp, r, threshold)
    direct_loss, _ = loss_fn(lp, r, threshold)

    assert torch.isclose(trainer_loss, direct_loss), (
        f"trainer loss {trainer_loss.item():.6f} != "
        f"direct loss {direct_loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 14 — batch_size=1 works
# ---------------------------------------------------------------------------

def test_rest_loss_batch_size_one():
    loss_fn = ReSTLoss()
    lp = _log_probs(1)
    r = torch.tensor([0.8])
    loss, n_accepted = loss_fn(lp, r, threshold=0.5)
    assert loss.shape == torch.Size([])
    assert n_accepted == 1
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Bonus tests for ReSTDataset.grow and ReSTTrainer.grow_dataset
# ---------------------------------------------------------------------------

def test_dataset_grow_size():
    ds = _make_dataset(n=3)
    n_per = 4

    def _gen(prompt: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return torch.randint(0, VOCAB, (5,)), 0.5

    grown = ds.grow(_gen, n_samples_per_prompt=n_per)
    assert len(grown) == len(ds) * n_per


def test_trainer_grow_dataset_size():
    trainer = ReSTTrainer(n_per_prompt=3)
    ds = _make_dataset(n=4)
    grown = trainer.grow_dataset(ds, n_per_prompt=3, seed=0)
    assert len(grown) == len(ds) * 3


def test_rest_loss_2d_log_probs():
    """ReSTLoss should handle (B, S) log_probs by averaging over S."""
    loss_fn = ReSTLoss()
    lp = _log_probs(8, seq=10)
    r = _rewards(8)
    loss, _ = loss_fn(lp, r, threshold=0.3)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
