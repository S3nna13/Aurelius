"""
Tests for src/training/differential_privacy.py

Covers:
  - PrivacyAccountant (4 tests)
  - PerSampleGradientClipper (3 tests)
  - GaussianMechanism (3 tests)
  - DPSGDOptimizer (2 tests)
  - DPTrainer (3 tests)
  - DPConfig (1 test)
  = 16 tests total
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from src.training.differential_privacy import (
    DPConfig,
    DPSGDOptimizer,
    DPTrainer,
    GaussianMechanism,
    PerSampleGradientClipper,
    PrivacyAccountant,
)

# ---------------------------------------------------------------------------
# Tiny model for tests
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB_SIZE = 16
B = 4
T = 4
MAX_GRAD_NORM = 1.0


class TinyModel(nn.Module):
    """Minimal language model: embedding → linear logits."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T] → [B, T, vocab]
        return self.proj(self.embed(x))


def make_model() -> TinyModel:
    torch.manual_seed(42)
    return TinyModel()


def make_batch():
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    labels = torch.randint(0, VOCAB_SIZE, (B, T))
    return input_ids, labels


def make_per_sample_grads(model: nn.Module, batch_size: int = B) -> dict:
    """Create fake per-sample grads with shape [B, *param_shape]."""
    torch.manual_seed(7)
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = torch.randn(batch_size, *param.shape)
    return grads


# ---------------------------------------------------------------------------
# PrivacyAccountant tests
# ---------------------------------------------------------------------------


def test_privacy_accountant_epsilon_from_steps_positive():
    acct = PrivacyAccountant(noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5)
    eps = acct.epsilon_from_steps(n_steps=100, sample_rate=0.01)
    assert isinstance(eps, float)
    assert eps > 0.0


def test_privacy_accountant_higher_noise_lower_epsilon():
    """Higher noise_multiplier → lower privacy cost (lower epsilon)."""
    acct_low = PrivacyAccountant(noise_multiplier=0.5, max_grad_norm=1.0, delta=1e-5)
    acct_high = PrivacyAccountant(noise_multiplier=2.0, max_grad_norm=1.0, delta=1e-5)
    eps_low = acct_low.epsilon_from_steps(100, 0.01)
    eps_high = acct_high.epsilon_from_steps(100, 0.01)
    assert eps_low > eps_high, (
        f"Higher noise should give lower epsilon, got {eps_low=} {eps_high=}"
    )


def test_privacy_accountant_moments_accountant_positive():
    acct = PrivacyAccountant(noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5)
    eps = acct.moments_accountant_epsilon(n_steps=50, sample_rate=0.05, alpha=10)
    assert isinstance(eps, float)
    assert eps > 0.0


def test_privacy_accountant_total_privacy_spent_keys():
    acct = PrivacyAccountant(noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5)
    result = acct.total_privacy_spent(n_steps=100, dataset_size=1000, batch_size=32)
    assert isinstance(result, dict)
    assert "epsilon" in result
    assert "delta" in result
    assert "sample_rate" in result
    assert result["epsilon"] > 0.0
    assert math.isclose(result["delta"], 1e-5)
    assert math.isclose(result["sample_rate"], 32 / 1000)


# ---------------------------------------------------------------------------
# PerSampleGradientClipper tests
# ---------------------------------------------------------------------------


def test_per_sample_clipper_output_shape_unchanged():
    model = make_model()
    clipper = PerSampleGradientClipper(model, max_norm=MAX_GRAD_NORM)
    grads = make_per_sample_grads(model)
    clipped = clipper.clip_gradients(grads)
    for name in grads:
        assert clipped[name].shape == grads[name].shape, (
            f"Shape mismatch for {name}: {clipped[name].shape} != {grads[name].shape}"
        )


def test_per_sample_clipper_norms_leq_max_norm():
    """After clipping, per-sample global L2 norms must be ≤ max_norm."""
    model = make_model()
    max_norm = MAX_GRAD_NORM
    clipper = PerSampleGradientClipper(model, max_norm=max_norm)
    # Use large grads to ensure clipping actually happens
    torch.manual_seed(1)
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = torch.randn(B, *param.shape) * 10.0

    clipped = clipper.clip_gradients(grads)

    # Compute per-sample norms of clipped grads
    squared = torch.zeros(B)
    for g in clipped.values():
        flat = g.reshape(B, -1)
        squared += flat.pow(2).sum(dim=1)
    norms = squared.sqrt()

    for i, n in enumerate(norms):
        assert n.item() <= max_norm + 1e-4, (
            f"Sample {i} norm {n.item():.4f} exceeds max_norm {max_norm}"
        )


def test_per_sample_clipper_aggregate_sets_param_grad():
    model = make_model()
    clipper = PerSampleGradientClipper(model, max_norm=MAX_GRAD_NORM)
    grads = make_per_sample_grads(model)
    clipped = clipper.clip_gradients(grads)
    clipper.aggregate(clipped)
    for name, param in model.named_parameters():
        assert param.grad is not None, f"param.grad not set for {name}"
        assert param.grad.shape == param.shape


# ---------------------------------------------------------------------------
# GaussianMechanism tests
# ---------------------------------------------------------------------------


def test_gaussian_mechanism_output_shape_unchanged():
    mech = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0)
    g = torch.randn(8, 4)
    noisy = mech.add_noise(g)
    assert noisy.shape == g.shape


def test_gaussian_mechanism_adds_stochasticity():
    """Two independent calls must differ (with overwhelming probability)."""
    torch.manual_seed(0)
    mech = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0)
    g = torch.ones(16)
    noisy1 = mech.add_noise(g)
    noisy2 = mech.add_noise(g)
    assert not torch.allclose(noisy1, noisy2), "Two noise samples should differ"


def test_gaussian_mechanism_sensitivity():
    mech = GaussianMechanism(noise_multiplier=1.1, max_grad_norm=2.5)
    s = mech.sensitivity(2.5)
    assert math.isclose(s, 2.5)


# ---------------------------------------------------------------------------
# DPSGDOptimizer tests
# ---------------------------------------------------------------------------


def test_dpsgd_optimizer_step_runs():
    """DPSGDOptimizer.step should run without raising."""
    model = make_model()
    opt = DPSGDOptimizer(
        model, lr=1e-3, max_grad_norm=MAX_GRAD_NORM, noise_multiplier=1.1
    )
    input_ids, labels = make_batch()

    # Build per-sample losses with grad_fn
    losses = []
    for b in range(B):
        logits = model(input_ids[b : b + 1]).view(-1, VOCAB_SIZE)
        lbl = labels[b : b + 1].view(-1)
        loss = nn.functional.cross_entropy(logits, lbl)
        losses.append(loss)
    per_sample = torch.stack(losses)

    opt.step(per_sample)  # should not raise


def test_dpsgd_optimizer_modifies_params():
    """Parameters should change after a DP-SGD step."""
    torch.manual_seed(99)
    model = make_model()
    # Snapshot initial parameters
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    opt = DPSGDOptimizer(
        model, lr=0.1, max_grad_norm=MAX_GRAD_NORM, noise_multiplier=1.1
    )
    input_ids, labels = make_batch()

    losses = []
    for b in range(B):
        logits = model(input_ids[b : b + 1]).view(-1, VOCAB_SIZE)
        lbl = labels[b : b + 1].view(-1)
        losses.append(nn.functional.cross_entropy(logits, lbl))
    per_sample = torch.stack(losses)

    opt.step(per_sample)

    changed = False
    for n, p in model.named_parameters():
        if not torch.allclose(before[n], p):
            changed = True
            break
    assert changed, "No parameter changed after DPSGDOptimizer.step"


# ---------------------------------------------------------------------------
# DPTrainer tests
# ---------------------------------------------------------------------------


def test_dptrainer_train_step_returns_finite():
    torch.manual_seed(42)
    model = make_model()
    trainer = DPTrainer(
        model,
        lr=1e-3,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=1.1,
        delta=1e-5,
    )
    input_ids, labels = make_batch()
    loss, eps = trainer.train_step(input_ids, labels)
    assert math.isfinite(loss), f"loss is not finite: {loss}"
    assert math.isfinite(eps), f"epsilon is not finite: {eps}"
    assert loss > 0.0
    assert eps > 0.0


def test_dptrainer_epsilon_increases_with_steps():
    torch.manual_seed(0)
    model = make_model()
    trainer = DPTrainer(
        model,
        lr=1e-3,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=1.1,
        delta=1e-5,
    )
    input_ids, labels = make_batch()

    _, eps1 = trainer.train_step(input_ids, labels)
    _, eps2 = trainer.train_step(input_ids, labels)
    _, eps3 = trainer.train_step(input_ids, labels)

    assert eps2 > eps1, f"Expected eps to increase: eps1={eps1} eps2={eps2}"
    assert eps3 > eps2, f"Expected eps to increase: eps2={eps2} eps3={eps3}"


def test_dptrainer_privacy_report_returns_dict():
    torch.manual_seed(1)
    model = make_model()
    trainer = DPTrainer(
        model,
        lr=1e-3,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=1.1,
        delta=1e-5,
    )
    input_ids, labels = make_batch()
    trainer.train_step(input_ids, labels)
    report = trainer.privacy_report()
    assert isinstance(report, dict)
    assert "epsilon" in report
    assert "delta" in report
    assert "n_steps" in report
    assert report["n_steps"] == 1


# ---------------------------------------------------------------------------
# DPConfig test
# ---------------------------------------------------------------------------


def test_dpconfig_defaults():
    cfg = DPConfig()
    assert math.isclose(cfg.max_grad_norm, 1.0)
    assert math.isclose(cfg.noise_multiplier, 1.1)
    assert math.isclose(cfg.delta, 1e-5)
    assert math.isclose(cfg.lr, 1e-4)
    assert math.isclose(cfg.target_epsilon, 8.0)
