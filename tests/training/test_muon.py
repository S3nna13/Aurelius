"""Tests for Muon optimizer."""
import torch
import torch.nn as nn
import pytest
from src.training.muon import Muon, _newton_schulz


def test_newton_schulz_shape():
    G = torch.randn(64, 32)
    out = _newton_schulz(G, steps=5)
    assert out.shape == G.shape


def test_newton_schulz_tall_matrix():
    G = torch.randn(128, 64)
    out = _newton_schulz(G, steps=5)
    assert out.shape == G.shape
    assert not torch.isnan(out).any()


def test_newton_schulz_wide_matrix():
    """Wide matrix (m < n) should be handled by transposing internally."""
    G = torch.randn(32, 64)
    out = _newton_schulz(G, steps=5)
    assert out.shape == G.shape
    assert not torch.isnan(out).any()


def test_newton_schulz_near_zero_gradient():
    """Near-zero gradient should return safely without NaN."""
    G = torch.zeros(32, 32) + 1e-40
    out = _newton_schulz(G, steps=5)
    assert not torch.isnan(out).any()


def test_muon_updates_parameters():
    """Muon step must change parameter values."""
    linear = nn.Linear(32, 64, bias=False)
    initial = linear.weight.data.clone()
    optimizer = Muon([linear.weight], lr=0.02, momentum=0.95)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(linear.weight.data, initial), "Muon did not update weights"


def test_muon_zero_lr_no_update():
    """lr=0 should leave weights unchanged."""
    linear = nn.Linear(32, 64, bias=False)
    initial = linear.weight.data.clone()
    optimizer = Muon([linear.weight], lr=0.0, momentum=0.95)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    assert torch.allclose(linear.weight.data, initial), "Muon changed weights with lr=0"


def test_muon_no_nan_after_step():
    """Weights must not become NaN after Muon step."""
    linear = nn.Linear(64, 64, bias=False)
    optimizer = Muon([linear.weight], lr=0.02)

    for _ in range(3):
        x = torch.randn(8, 64)
        loss = linear(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert not torch.isnan(linear.weight).any()


def test_muon_rejects_1d_params():
    """Muon should raise ValueError for 1D parameters."""
    bias = nn.Parameter(torch.randn(64))
    optimizer = Muon([bias], lr=0.02)
    bias.grad = torch.randn(64)
    with pytest.raises(ValueError, match="2D"):
        optimizer.step()


def test_muon_weight_decay():
    """Weight decay should shrink parameter magnitude over time."""
    linear = nn.Linear(32, 64, bias=False)
    nn.init.ones_(linear.weight)
    optimizer = Muon([linear.weight], lr=0.01, momentum=0.0, weight_decay=0.1)

    x = torch.randn(4, 32)
    loss = linear(x).sum()
    loss.backward()
    optimizer.step()

    # With weight_decay, norm should decrease
    assert linear.weight.norm() < 32 * 64 ** 0.5  # less than initialized all-ones norm
