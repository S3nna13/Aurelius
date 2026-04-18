"""
Tests for src/training/architecture_regularization.py

Uses a tiny transformer-like model (d_model=16, vocab_size=16, B=2, T=4)
built entirely from native PyTorch primitives.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.architecture_regularization import (
    GradientPenalty,
    LipschitzConstraint,
    OrthogonalRegularizer,
    RegConfig,
    RegularizedTrainer,
    SpectralNorm,
    WeightConstraints,
)

# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB = 16
B = 2
T = 4


class TinyModel(nn.Module):
    """Minimal embedding + linear model that returns logits."""

    def __init__(self, vocab: int = VOCAB, d: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.fc1 = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T] -> [B, T, vocab]
        h = self.embed(x)
        h = self.norm(F.relu(self.fc1(h)))
        return self.head(h)


def make_model() -> TinyModel:
    torch.manual_seed(42)
    return TinyModel()


def make_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# SpectralNorm tests
# ---------------------------------------------------------------------------

def test_spectral_norm_compute_sigma_positive():
    """compute_sigma should return a positive scalar."""
    sn = SpectralNorm(n_power_iterations=5)
    W = torch.randn(8, 4)
    u = F.normalize(torch.randn(8), dim=0)
    v = F.normalize(torch.randn(4), dim=0)
    sigma, u_new, v_new = sn.compute_sigma(W, u, v, n_iters=5)
    assert sigma.item() > 0, f"Expected positive sigma, got {sigma.item()}"


def test_spectral_norm_applied_forward_works():
    """A module with SpectralNorm applied should still produce correct output shape."""
    sn = SpectralNorm(n_power_iterations=1)
    linear = nn.Linear(8, 4, bias=False)
    sn.apply(linear)
    x = torch.randn(3, 8)
    out = linear(x)
    assert out.shape == (3, 4)


def test_spectral_norm_singular_value_leq_one():
    """After many power iterations the effective weight's spectral norm should be ≤ 1."""
    sn = SpectralNorm(n_power_iterations=20)
    # Linear(in=16, out=8) → weight shape (8, 16)
    linear = nn.Linear(16, 8, bias=False)
    # Inflate the weights so we can clearly check normalisation
    nn.init.normal_(linear.weight, std=5.0)
    sn.apply(linear)
    # Run a forward pass with correct in_features=16 to trigger u/v update
    _ = linear(torch.randn(4, 16))
    # Recompute spectral norm of the normalised weight used inside forward
    W = linear.weight
    u = linear._sn_u
    v = linear._sn_v
    sigma, _, _ = sn.compute_sigma(W, u, v, n_iters=50)
    W_hat = W / (sigma + 1e-12)
    sv = torch.linalg.svdvals(W_hat)
    assert sv[0].item() <= 1.05, f"Spectral norm of W_hat: {sv[0].item()}"


def test_spectral_norm_remove_restores_forward():
    """remove() should restore original forward without error."""
    sn = SpectralNorm()
    linear = nn.Linear(4, 4, bias=False)
    sn.apply(linear)
    sn.remove(linear)
    x = torch.randn(2, 4)
    out = linear(x)
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# OrthogonalRegularizer tests
# ---------------------------------------------------------------------------

def test_orthogonal_loss_non_negative():
    """Orthogonal loss must be non-negative for arbitrary weight."""
    reg = OrthogonalRegularizer()
    W = torch.randn(8, 8)
    loss_val = reg.loss(W)
    assert loss_val.item() >= 0.0


def test_orthogonal_loss_near_zero_for_orthogonal_matrix():
    """Loss should be close to 0 for an orthogonal matrix."""
    reg = OrthogonalRegularizer()
    W = torch.zeros(8, 8)
    nn.init.orthogonal_(W)
    loss_val = reg.loss(W)
    assert loss_val.item() < 1e-6, f"Expected near-zero loss, got {loss_val.item()}"


def test_orthogonal_init_produces_near_orthogonal():
    """orthogonal_init should produce a matrix W with W W^T ≈ I."""
    W = torch.empty(8, 8)
    W_orth = OrthogonalRegularizer.orthogonal_init(W)
    gram = W_orth.mm(W_orth.t())
    I = torch.eye(8)
    err = (gram - I).norm().item()
    assert err < 1e-4, f"Orthogonal init deviation: {err}"


def test_orthogonal_apply_to_model_returns_scalar():
    """apply_to_model should return a scalar (0-dim or 1-element) tensor."""
    reg = OrthogonalRegularizer(lambda_orth=1e-3)
    model = make_model()
    result = reg.apply_to_model(model)
    assert result.numel() == 1
    assert result.item() >= 0.0


# ---------------------------------------------------------------------------
# GradientPenalty tests
# ---------------------------------------------------------------------------

def test_gradient_norm_penalty_non_negative():
    """gradient_norm_penalty should return a non-negative scalar."""
    gp = GradientPenalty(lambda_gp=10.0)
    model = make_model()
    input_ids = make_ids()

    def loss_fn(m, ids):
        logits = m(ids)
        B_, T_, V_ = logits.shape
        labels = ids.view(B_ * T_)
        return F.cross_entropy(logits.view(B_ * T_, V_), labels)

    penalty = gp.gradient_norm_penalty(model, input_ids, loss_fn, threshold=1.0)
    assert penalty.item() >= 0.0


def test_gradient_norm_penalty_is_scalar():
    """gradient_norm_penalty result must be a scalar tensor."""
    gp = GradientPenalty(lambda_gp=5.0)
    model = make_model()
    input_ids = make_ids()

    def loss_fn(m, ids):
        logits = m(ids)
        B_, T_, V_ = logits.shape
        return F.cross_entropy(logits.view(B_ * T_, V_), ids.view(B_ * T_))

    penalty = gp.gradient_norm_penalty(model, input_ids, loss_fn)
    assert penalty.dim() == 0 or penalty.numel() == 1


# ---------------------------------------------------------------------------
# WeightConstraints tests
# ---------------------------------------------------------------------------

def test_clip_weights_bounds():
    """After clip_weights all non-bias params should be within [-max_norm, max_norm]."""
    model = make_model()
    # Inflate weights
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "bias" not in name:
                p.fill_(5.0)
    max_norm = 1.0
    WeightConstraints.clip_weights(model, max_norm=max_norm)
    for name, p in model.named_parameters():
        if "bias" not in name:
            assert p.abs().max().item() <= max_norm + 1e-6, (
                f"Param {name} exceeds max_norm: {p.abs().max().item()}"
            )


def test_weight_decay_selective_excludes_bias():
    """weight_decay_selective with exclude=['bias'] should not penalize biases."""
    model = make_model()
    # Set bias params to large values
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "bias" in name:
                p.fill_(100.0)
            else:
                p.fill_(0.0)
    reg = WeightConstraints.weight_decay_selective(model, wd=1.0, exclude=["bias"])
    # All non-bias params are zero, so reg should be zero
    assert reg.item() < 1e-6, f"Expected ~0 reg loss, got {reg.item()}"


def test_weight_decay_selective_non_negative():
    """weight_decay_selective should return a non-negative value."""
    model = make_model()
    reg = WeightConstraints.weight_decay_selective(model, wd=1e-4, exclude=["bias", "norm"])
    assert reg.item() >= 0.0


def test_nuclear_norm_non_negative():
    """Nuclear norm regularizer should be non-negative."""
    W = torch.randn(8, 4)
    val = WeightConstraints.nuclear_norm_regularizer(W, lambda_nuc=1e-3)
    assert val.item() >= 0.0


# ---------------------------------------------------------------------------
# LipschitzConstraint tests
# ---------------------------------------------------------------------------

def test_lipschitz_enforce_via_clipping_bounds_singular_values():
    """After enforce_via_clipping all singular values should be ≤ k."""
    k = 0.5
    lc = LipschitzConstraint(k=k)
    linear = nn.Linear(8, 8, bias=False)
    nn.init.normal_(linear.weight, std=5.0)
    lc.enforce_via_clipping(linear)
    sv = torch.linalg.svdvals(linear.weight.data)
    assert sv.max().item() <= k + 1e-5, f"Max sv: {sv.max().item()} > k={k}"


def test_lipschitz_estimate_lipschitz_positive():
    """estimate_lipschitz should return a positive float."""
    lc = LipschitzConstraint(k=1.0)
    linear = nn.Linear(4, 4, bias=False)
    val = lc.estimate_lipschitz(linear)
    assert isinstance(val, float)
    assert val > 0.0


# ---------------------------------------------------------------------------
# RegularizedTrainer tests
# ---------------------------------------------------------------------------

def test_regularized_trainer_train_step_finite():
    """train_step should return finite (task_loss, reg_loss)."""
    model = make_model()
    cfg = RegConfig()
    trainer = RegularizedTrainer(model, lr=cfg.lr, config=cfg)
    input_ids = make_ids()
    labels = make_ids()
    task_loss, reg_loss = trainer.train_step(input_ids, labels)
    assert math.isfinite(task_loss.item()), f"task_loss is not finite: {task_loss.item()}"
    assert math.isfinite(reg_loss.item()), f"reg_loss is not finite: {reg_loss.item()}"


def test_regularized_trainer_apply_constraints_runs():
    """apply_constraints should run without raising exceptions."""
    model = make_model()
    cfg = RegConfig(max_weight_norm=0.5, k_lipschitz=0.5)
    trainer = RegularizedTrainer(model, lr=cfg.lr, config=cfg)
    trainer.apply_constraints()  # Must not raise


# ---------------------------------------------------------------------------
# RegConfig defaults
# ---------------------------------------------------------------------------

def test_regconfig_defaults():
    """RegConfig should have the specified default values."""
    cfg = RegConfig()
    assert cfg.lambda_orth == 1e-3
    assert cfg.lambda_gp == 10.0
    assert cfg.lambda_nuc == 1e-4
    assert cfg.max_weight_norm == 1.0
    assert cfg.k_lipschitz == 1.0
    assert cfg.n_power_iterations == 1
    assert cfg.lr == 1e-4


def test_regconfig_custom_values():
    """RegConfig should accept custom values."""
    cfg = RegConfig(lambda_orth=0.01, k_lipschitz=2.0, lr=3e-4)
    assert cfg.lambda_orth == 0.01
    assert cfg.k_lipschitz == 2.0
    assert cfg.lr == 3e-4
