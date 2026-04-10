"""Tests for Mixture Density Networks (mixture_density.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.model.mixture_density import (
    MDNConfig,
    MDNHead,
    MDNModel,
    mdn_loss,
    mdn_mean,
    mdn_sample,
    mdn_variance,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BATCH = 4
K = 5   # n_components
D = 1   # output_dim
H_IN = 64


@pytest.fixture()
def default_config() -> MDNConfig:
    return MDNConfig()


@pytest.fixture()
def small_config() -> MDNConfig:
    return MDNConfig(n_components=K, input_dim=H_IN, output_dim=D, hidden_dim=128)


@pytest.fixture()
def mdn_head(small_config):
    return MDNHead(small_config)


@pytest.fixture()
def fake_hidden():
    torch.manual_seed(0)
    return torch.randn(BATCH, H_IN)


@pytest.fixture()
def mdn_params(mdn_head, fake_hidden):
    with torch.no_grad():
        return mdn_head(fake_hidden)


@pytest.fixture()
def target():
    torch.manual_seed(1)
    return torch.randn(BATCH, D)


# ---------------------------------------------------------------------------
# 1. MDNConfig defaults
# ---------------------------------------------------------------------------

def test_mdn_config_default_n_components(default_config):
    assert default_config.n_components == 5


def test_mdn_config_default_input_dim(default_config):
    assert default_config.input_dim == 64


def test_mdn_config_default_output_dim(default_config):
    assert default_config.output_dim == 1


def test_mdn_config_default_hidden_dim(default_config):
    assert default_config.hidden_dim == 128


# ---------------------------------------------------------------------------
# 2. MDNHead output shapes
# ---------------------------------------------------------------------------

def test_mdn_head_pi_shape(mdn_params):
    pi, mu, sigma = mdn_params
    assert pi.shape == (BATCH, K), f"pi shape {pi.shape}"


def test_mdn_head_mu_shape(mdn_params):
    pi, mu, sigma = mdn_params
    assert mu.shape == (BATCH, K, D), f"mu shape {mu.shape}"


def test_mdn_head_sigma_shape(mdn_params):
    pi, mu, sigma = mdn_params
    assert sigma.shape == (BATCH, K, D), f"sigma shape {sigma.shape}"


# ---------------------------------------------------------------------------
# 3. pi sums to 1
# ---------------------------------------------------------------------------

def test_pi_sums_to_one(mdn_params):
    pi, _, _ = mdn_params
    sums = pi.sum(dim=-1)  # (B,)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), f"pi sums: {sums}"


# ---------------------------------------------------------------------------
# 4. sigma is strictly positive
# ---------------------------------------------------------------------------

def test_sigma_positive(mdn_params):
    _, _, sigma = mdn_params
    assert (sigma > 0).all(), "sigma must be strictly positive"


# ---------------------------------------------------------------------------
# 5. mdn_loss returns positive scalar
# ---------------------------------------------------------------------------

def test_mdn_loss_scalar(mdn_params, target):
    pi, mu, sigma = mdn_params
    loss = mdn_loss(pi, mu, sigma, target)
    assert loss.shape == (), f"loss should be scalar, got {loss.shape}"


def test_mdn_loss_positive(mdn_params, target):
    pi, mu, sigma = mdn_params
    loss = mdn_loss(pi, mu, sigma, target)
    assert loss.item() > 0, f"NLL loss should be positive, got {loss.item()}"


# ---------------------------------------------------------------------------
# 6. mdn_loss lower for better-fit parameters
# ---------------------------------------------------------------------------

def test_mdn_loss_lower_for_good_fit():
    """A mixture centred on the target should have lower NLL."""
    torch.manual_seed(42)
    B, K2, D2 = 8, 3, 2
    target = torch.zeros(B, D2)

    # Good fit: all components centred on zero, tiny sigma
    pi_good    = torch.full((B, K2), 1.0 / K2)
    mu_good    = torch.zeros(B, K2, D2)
    sigma_good = torch.full((B, K2, D2), 0.1)

    # Bad fit: all components far from zero, large sigma
    pi_bad    = torch.full((B, K2), 1.0 / K2)
    mu_bad    = torch.full((B, K2, D2), 10.0)
    sigma_bad = torch.full((B, K2, D2), 2.0)

    loss_good = mdn_loss(pi_good, mu_good, sigma_good, target)
    loss_bad  = mdn_loss(pi_bad,  mu_bad,  sigma_bad,  target)

    assert loss_good.item() < loss_bad.item(), (
        f"Good-fit loss ({loss_good.item():.4f}) should be < "
        f"bad-fit loss ({loss_bad.item():.4f})"
    )


# ---------------------------------------------------------------------------
# 7. mdn_sample output shape
# ---------------------------------------------------------------------------

def test_mdn_sample_shape(mdn_params):
    pi, mu, sigma = mdn_params
    samples = mdn_sample(pi, mu, sigma)
    assert samples.shape == (BATCH, D), f"sample shape {samples.shape}"


def test_mdn_sample_is_finite(mdn_params):
    pi, mu, sigma = mdn_params
    samples = mdn_sample(pi, mu, sigma)
    assert torch.isfinite(samples).all(), "samples should be finite"


# ---------------------------------------------------------------------------
# 8. mdn_mean output shape
# ---------------------------------------------------------------------------

def test_mdn_mean_shape(mdn_params):
    pi, mu, sigma = mdn_params
    mean = mdn_mean(pi, mu)
    assert mean.shape == (BATCH, D), f"mean shape {mean.shape}"


def test_mdn_mean_equals_weighted_sum():
    """mdn_mean should equal manually computed weighted sum."""
    torch.manual_seed(7)
    B, K2, D2 = 3, 4, 2
    pi = torch.softmax(torch.randn(B, K2), dim=-1)
    mu = torch.randn(B, K2, D2)

    result   = mdn_mean(pi, mu)
    expected = (pi.unsqueeze(-1) * mu).sum(dim=1)

    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 9. mdn_variance positive values
# ---------------------------------------------------------------------------

def test_mdn_variance_shape(mdn_params):
    pi, mu, sigma = mdn_params
    var = mdn_variance(pi, mu, sigma)
    assert var.shape == (BATCH, D), f"variance shape {var.shape}"


def test_mdn_variance_positive(mdn_params):
    pi, mu, sigma = mdn_params
    var = mdn_variance(pi, mu, sigma)
    assert (var >= 0).all(), "variance must be non-negative"


# ---------------------------------------------------------------------------
# 10. MDNModel forward returns 3-tuple
# ---------------------------------------------------------------------------

def test_mdn_model_returns_tuple(small_config):
    encoder = nn.Linear(H_IN, H_IN)
    model   = MDNModel(encoder, small_config)
    x       = torch.randn(BATCH, H_IN)
    out     = model(x)
    assert isinstance(out, tuple) and len(out) == 3, (
        f"MDNModel.forward should return a 3-tuple, got {type(out)} len={len(out)}"
    )


def test_mdn_model_output_shapes(small_config):
    encoder = nn.Linear(H_IN, H_IN)
    model   = MDNModel(encoder, small_config)
    x       = torch.randn(BATCH, H_IN)
    pi, mu, sigma = model(x)
    assert pi.shape    == (BATCH, K),    f"pi {pi.shape}"
    assert mu.shape    == (BATCH, K, D), f"mu {mu.shape}"
    assert sigma.shape == (BATCH, K, D), f"sigma {sigma.shape}"


# ---------------------------------------------------------------------------
# 11. Gradient flows through mdn_loss
# ---------------------------------------------------------------------------

def test_gradient_flows_through_loss(small_config):
    """mdn_loss should be differentiable w.r.t. MDNHead parameters."""
    head   = MDNHead(small_config)
    h      = torch.randn(BATCH, H_IN)
    target = torch.randn(BATCH, D)

    pi, mu, sigma = head(h)
    loss = mdn_loss(pi, mu, sigma, target)
    loss.backward()

    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for param: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for: {name}"


def test_gradient_flows_through_mdn_model(small_config):
    """End-to-end gradient: encoder + MDNHead."""
    encoder = nn.Linear(H_IN, H_IN)
    model   = MDNModel(encoder, small_config)
    x       = torch.randn(BATCH, H_IN, requires_grad=True)
    target  = torch.randn(BATCH, D)

    pi, mu, sigma = model(x)
    loss = mdn_loss(pi, mu, sigma, target)
    loss.backward()

    assert x.grad is not None, "Gradient should flow back to input"
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 12. Multi-dimensional output (D > 1)
# ---------------------------------------------------------------------------

def test_multi_dim_output():
    """Ensure MDNHead works correctly when output_dim > 1."""
    cfg  = MDNConfig(n_components=3, input_dim=32, output_dim=4, hidden_dim=64)
    head = MDNHead(cfg)
    h    = torch.randn(BATCH, 32)
    pi, mu, sigma = head(h)
    assert pi.shape    == (BATCH, 3)
    assert mu.shape    == (BATCH, 3, 4)
    assert sigma.shape == (BATCH, 3, 4)

    target = torch.randn(BATCH, 4)
    loss = mdn_loss(pi, mu, sigma, target)
    assert torch.isfinite(loss)
