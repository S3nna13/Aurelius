"""Tests for SPIN: Self-Play Fine-Tuning (Chen et al. 2024).

Covers SPINLoss, SPINDataCollector, and SPINTrainer with >= 12 tests.
All models are pure PyTorch; no external dependencies.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from aurelius.alignment.spin import (
    SPINLoss,
    SPINDataCollector,
    SPINTrainer,
)


# ---------------------------------------------------------------------------
# Tiny helper model (pure PyTorch)
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal causal-LM-like model for tests.

    input_ids (B, T) -> logits (B, T, V)
    """

    def __init__(self, V: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(V, V)
        self.proj  = nn.Linear(V, V)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:  # x: (B, T) -> (B, T, V)
        return self.proj(self.embed(x).float())


V = 8    # vocabulary size used throughout tests
B = 4    # default batch size
T = 6    # default sequence length


def make_model() -> _TinyModel:
    torch.manual_seed(0)
    return _TinyModel(V)


def make_trainer(policy: nn.Module | None = None, ref: nn.Module | None = None):
    if policy is None:
        policy = make_model()
    if ref is None:
        ref = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    loss_fn   = SPINLoss(beta=0.1)
    return SPINTrainer(
        policy_model=policy,
        ref_model=ref,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )


# ---------------------------------------------------------------------------
# SPINLoss tests
# ---------------------------------------------------------------------------

def test_spin_loss_returns_scalar_and_dict():
    """SPINLoss.forward must return a 0-d tensor and a dict."""
    loss_fn = SPINLoss(beta=0.1)
    pi_real  = torch.randn(B)
    pi_gen   = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen  = torch.randn(B)

    loss, metrics = loss_fn(pi_real, pi_gen, ref_real, ref_gen)

    assert loss.ndim == 0, f"Expected 0-d scalar, got shape {loss.shape}"
    assert isinstance(metrics, dict), "Second return value must be a dict"


def test_spin_loss_dict_keys():
    """metrics dict must contain exactly the required keys."""
    loss_fn = SPINLoss(beta=0.1)
    pi_real  = torch.randn(B)
    pi_gen   = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen  = torch.randn(B)

    _, metrics = loss_fn(pi_real, pi_gen, ref_real, ref_gen)

    required = {"accuracy", "reward_real", "reward_gen", "margin"}
    assert set(metrics.keys()) == required, (
        f"Expected keys {required}, got {set(metrics.keys())}"
    )


def test_spin_loss_is_finite():
    """Loss must be finite for random inputs."""
    loss_fn = SPINLoss(beta=0.1)
    torch.manual_seed(7)
    pi_real  = torch.randn(B)
    pi_gen   = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen  = torch.randn(B)

    loss, _ = loss_fn(pi_real, pi_gen, ref_real, ref_gen)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_spin_loss_perfect_separation_accuracy():
    """When reward_real >> reward_gen for all samples, accuracy must be 1.0."""
    loss_fn = SPINLoss(beta=1.0)
    # pi_real - ref_real >> pi_gen - ref_gen  =>  reward_real > reward_gen
    pi_real  = torch.full((B,), 10.0)
    pi_gen   = torch.full((B,), -10.0)
    ref_real = torch.zeros(B)
    ref_gen  = torch.zeros(B)

    _, metrics = loss_fn(pi_real, pi_gen, ref_real, ref_gen)

    assert metrics["accuracy"] == 1.0, (
        f"Expected accuracy=1.0 for perfect separation, got {metrics['accuracy']}"
    )


def test_spin_loss_gradients_flow():
    """Loss.backward() must produce non-None, non-zero gradients on pi_real."""
    loss_fn = SPINLoss(beta=0.1)
    pi_real  = torch.randn(B, requires_grad=True)
    pi_gen   = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen  = torch.randn(B)

    loss, _ = loss_fn(pi_real, pi_gen, ref_real, ref_gen)
    loss.backward()

    assert pi_real.grad is not None, "No gradient on pi_real"
    assert pi_real.grad.abs().sum().item() > 0, "Zero gradient on pi_real"


def test_spin_loss_margin_sign():
    """When real completions are clearly better, margin must be positive."""
    loss_fn = SPINLoss(beta=0.1)
    pi_real  = torch.full((B,), 0.0)
    pi_gen   = torch.full((B,), -10.0)
    ref_real = torch.full((B,), -5.0)
    ref_gen  = torch.full((B,), -5.0)

    _, metrics = loss_fn(pi_real, pi_gen, ref_real, ref_gen)

    assert metrics["margin"] > 0.0, (
        f"Expected positive margin, got {metrics['margin']}"
    )


# ---------------------------------------------------------------------------
# SPINDataCollector tests
# ---------------------------------------------------------------------------

def test_data_collector_sequence_log_prob_shape_1d():
    """sequence_log_prob on a 1-D input must return a scalar (0-d) tensor."""
    collector = SPINDataCollector(beta=0.1)
    lp = torch.randn(T)

    out = collector.sequence_log_prob(lp)

    assert out.ndim == 0, f"Expected 0-d tensor, got shape {out.shape}"


def test_data_collector_sequence_log_prob_shape_2d():
    """sequence_log_prob on (B, T) input must return shape (B,)."""
    collector = SPINDataCollector(beta=0.1)
    lp = torch.randn(B, T)

    out = collector.sequence_log_prob(lp)

    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_data_collector_build_pairs_length():
    """build_pairs must return a list with the same length as the inputs."""
    collector = SPINDataCollector(beta=0.1)
    N = 5
    real_lps = [torch.randn(T) for _ in range(N)]
    gen_lps  = [torch.randn(T) for _ in range(N)]

    pairs = collector.build_pairs(real_lps, gen_lps)

    assert len(pairs) == N, f"Expected {N} pairs, got {len(pairs)}"


def test_data_collector_build_pairs_elements_are_scalars():
    """Each element of build_pairs must be a (scalar, scalar) tuple."""
    collector = SPINDataCollector(beta=0.1)
    real_lps = [torch.randn(T) for _ in range(3)]
    gen_lps  = [torch.randn(T) for _ in range(3)]

    pairs = collector.build_pairs(real_lps, gen_lps)

    for i, (r, g) in enumerate(pairs):
        assert r.ndim == 0, f"Pair {i}: real element should be scalar, got shape {r.shape}"
        assert g.ndim == 0, f"Pair {i}: gen element should be scalar, got shape {g.shape}"


# ---------------------------------------------------------------------------
# SPINTrainer tests
# ---------------------------------------------------------------------------

def test_trainer_freeze_ref_freezes_all_params():
    """After freeze_ref(), every ref model parameter must have requires_grad=False."""
    trainer = make_trainer()
    trainer.freeze_ref()

    for name, p in trainer.ref_model.named_parameters():
        assert not p.requires_grad, (
            f"Parameter '{name}' still requires grad after freeze_ref()"
        )


def test_trainer_compute_sequence_log_prob_shape():
    """compute_sequence_log_prob must return a (B,) tensor."""
    model     = make_model()
    input_ids = torch.randint(0, V, (B, T))
    labels    = torch.randint(0, V, (B, T))

    trainer = make_trainer(policy=model)
    out = trainer.compute_sequence_log_prob(model, input_ids, labels)

    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_trainer_compute_sequence_log_prob_masking():
    """Positions where labels==-100 must not affect the sum (masking test)."""
    model = make_model()

    input_ids      = torch.randint(0, V, (1, T))
    labels_full    = torch.randint(0, V, (1, T))
    labels_partial = labels_full.clone()
    labels_partial[0, -2:] = -100  # mask last 2 tokens

    trainer = make_trainer(policy=model)

    lp_full    = trainer.compute_sequence_log_prob(model, input_ids, labels_full)
    lp_partial = trainer.compute_sequence_log_prob(model, input_ids, labels_partial)

    # Partial sum (T-2 tokens) must be <= full sum (T tokens) in absolute value
    # but they must differ (unless all last tokens had log-prob 0, which is impossible)
    assert not torch.allclose(lp_full, lp_partial), (
        "Masking labels=-100 had no effect on the sequence log prob"
    )


def test_trainer_spin_step_metric_keys():
    """spin_step must return a dict with the required metric keys."""
    trainer   = make_trainer()
    real_ids  = torch.randint(0, V, (B, T))
    gen_ids   = torch.randint(0, V, (B, T))
    labels    = torch.randint(0, V, (B, T))

    _, metrics = trainer.spin_step(real_ids, gen_ids, labels)

    required = {"accuracy", "reward_real", "reward_gen", "margin"}
    assert set(metrics.keys()) == required, (
        f"Expected keys {required}, got {set(metrics.keys())}"
    )


def test_trainer_spin_step_loss_is_finite():
    """spin_step must return a finite loss."""
    trainer  = make_trainer()
    real_ids = torch.randint(0, V, (B, T))
    gen_ids  = torch.randint(0, V, (B, T))
    labels   = torch.randint(0, V, (B, T))

    loss, _ = trainer.spin_step(real_ids, gen_ids, labels)

    assert torch.isfinite(loss), f"spin_step loss is not finite: {loss.item()}"


def test_trainer_spin_step_updates_policy_weights():
    """Policy model weights must change after spin_step (grad != 0)."""
    trainer  = make_trainer()
    real_ids = torch.randint(0, V, (B, T))
    gen_ids  = torch.randint(0, V, (B, T))
    labels   = torch.randint(0, V, (B, T))

    weights_before = {
        name: p.clone().detach()
        for name, p in trainer.policy_model.named_parameters()
    }

    trainer.spin_step(real_ids, gen_ids, labels)

    weights_after = {
        name: p.clone().detach()
        for name, p in trainer.policy_model.named_parameters()
    }

    changed = any(
        not torch.allclose(weights_before[n], weights_after[n])
        for n in weights_before
    )
    assert changed, "Policy model weights did not change after spin_step"
