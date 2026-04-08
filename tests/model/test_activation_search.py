"""Tests for activation_search.py — activation library and learnable meta-activation."""

import torch
import torch.nn as nn
import pytest

from src.model.activation_search import (
    ActivationInfo,
    ACTIVATION_REGISTRY,
    get_activation,
    Mish,
    Squareplus,
    StarReLU,
    LearnableActivation,
    ActivationSearchSpace,
)


# ── Activation library tests ──────────────────────────────────────────


def test_get_activation_gelu():
    act = get_activation("gelu")
    assert isinstance(act, nn.Module)
    x = torch.randn(4, 32)
    out = act(x)
    assert out.shape == x.shape


def test_get_activation_mish_smooth():
    act = get_activation("mish")
    x = torch.randn(8, 64)
    out = act(x)
    assert out.shape == (8, 64)


def test_get_activation_squareplus_positive_ish():
    act = get_activation("squareplus")
    # Squareplus is always positive: f(x) = (x + sqrt(x^2 + b)) / 2 > 0
    x = torch.full((10,), -100.0)
    out = act(x)
    assert (out > 0).all(), f"Expected all positive, got min={out.min().item()}"


def test_starrelu_learnable_params():
    act = StarReLU()
    param_names = {n for n, _ in act.named_parameters()}
    assert "s" in param_names, f"Expected 's' in params, got {param_names}"
    assert "b" in param_names, f"Expected 'b' in params, got {param_names}"
    # Both should require grad
    assert act.s.requires_grad
    assert act.b.requires_grad


def test_starrelu_output_shape():
    act = StarReLU()
    x = torch.randn(3, 16)
    out = act(x)
    assert out.shape == (3, 16)


def test_get_activation_unknown_raises():
    with pytest.raises(ValueError, match="unknown"):
        get_activation("nonexistent_activation_xyz")


def test_activation_registry_populated():
    expected = {"gelu", "relu", "silu", "mish", "squareplus", "starrelu"}
    assert expected <= set(ACTIVATION_REGISTRY.keys()), (
        f"Missing: {expected - set(ACTIVATION_REGISTRY.keys())}"
    )
    for name, info in ACTIVATION_REGISTRY.items():
        assert isinstance(info, ActivationInfo)
        assert info.name == name


# ── LearnableActivation tests ─────────────────────────────────────────


def test_learnable_activation_output_shape():
    act = LearnableActivation(hidden_dim=16)
    x = torch.randn(4, 32)
    out = act(x)
    assert out.shape == x.shape


def test_learnable_activation_fit_to():
    import torch.nn.functional as F

    act = LearnableActivation(hidden_dim=16)
    # Measure initial loss
    x = torch.linspace(-3, 3, 50)
    with torch.no_grad():
        initial_out = act(x)
    target = F.gelu(x)
    initial_mse = ((initial_out - target) ** 2).mean().item()

    # Fit to gelu
    final_mse = act.fit_to(F.gelu, n_steps=100, lr=0.01)
    assert final_mse < initial_mse, (
        f"MSE should decrease after fitting: {initial_mse:.4f} → {final_mse:.4f}"
    )


def test_learnable_activation_init_from():
    # Should not raise
    act = LearnableActivation(hidden_dim=16, init_from="gelu")
    x = torch.randn(2, 8)
    out = act(x)
    assert out.shape == x.shape


# ── ActivationSearchSpace tests ───────────────────────────────────────


def test_activation_search_space_sample():
    space = ActivationSearchSpace()
    name = space.sample()
    assert isinstance(name, str)
    assert name in space.choices


def test_activation_search_space_search():
    space = ActivationSearchSpace()

    def model_fn(act_name: str) -> nn.Module:
        act = get_activation(act_name)
        return act

    def eval_fn(model: nn.Module) -> float:
        # Simple deterministic metric based on output mean
        x = torch.randn(16, 32)
        with torch.no_grad():
            out = model(x)
        return float(out.mean())

    best = space.search(model_fn, eval_fn, n_trials=6)
    assert isinstance(best, str)
    assert best in space.choices
