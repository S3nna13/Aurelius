"""Tests for BalancedMoEFFN (auxiliary-loss-free MoE load balancing)."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.moe import MoEConfig
from src.model.moe_balanced import BalancedMoEFFN


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> AureliusConfig:
    """Small AureliusConfig suitable for fast unit tests."""
    return AureliusConfig(
        d_model=2048,
        n_layers=2,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        d_ff=5632,
        vocab_size=128_000,
        max_seq_len=64,
        dropout=0.0,
    )


@pytest.fixture()
def moe_cfg() -> MoEConfig:
    return MoEConfig(n_experts=8, top_k=2, load_balance_alpha=0.01)


@pytest.fixture()
def model(config, moe_cfg) -> BalancedMoEFFN:
    return BalancedMoEFFN(config, moe_cfg)


@pytest.fixture()
def x() -> torch.Tensor:
    """Input tensor of shape (2, 16, 2048)."""
    torch.manual_seed(42)
    return torch.randn(2, 16, 2048)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_balanced_moe_output_shape(model, x):
    """Output shape must be (2, 16, 2048) and aux_loss must be a scalar."""
    output, aux_loss = model(x)

    assert output.shape == (2, 16, 2048), (
        f"Expected output shape (2, 16, 2048), got {output.shape}"
    )
    assert aux_loss.shape == torch.Size([]), (
        f"aux_loss should be a scalar, got shape {aux_loss.shape}"
    )


def test_aux_loss_is_zero(model, x):
    """aux_loss must be exactly 0.0 - no auxiliary loss in this design."""
    _, aux_loss = model(x)
    assert aux_loss.item() == 0.0, (
        f"Expected aux_loss == 0.0, got {aux_loss.item()}"
    )


def test_expert_bias_updates(config, moe_cfg, x):
    """After a forward pass, expert_bias should differ from its initial zeros."""
    m = BalancedMoEFFN(config, moe_cfg)

    bias_before = m.expert_bias.clone()
    assert torch.all(bias_before == 0.0), "expert_bias should start at zeros"

    m(x)

    bias_after = m.expert_bias.clone()
    assert not torch.all(bias_after == bias_before), (
        "expert_bias should change after a forward pass with imbalanced routing"
    )


def test_expert_bias_no_gradient(model):
    """expert_bias.requires_grad must be False - bias updates are manual."""
    assert model.expert_bias.requires_grad is False, (
        "expert_bias should have requires_grad=False so the optimiser ignores it"
    )


def test_load_balancing_convergence(config, moe_cfg):
    """Running multiple forward passes should reduce variance in expert loads."""
    torch.manual_seed(0)
    m = BalancedMoEFFN(config, moe_cfg)

    x = torch.randn(2, 16, 2048)

    def load_variance() -> float:
        loads = m.get_expert_loads(x)
        fracs = list(loads.values())
        mean = sum(fracs) / len(fracs)
        return sum((f - mean) ** 2 for f in fracs) / len(fracs)

    initial_var = load_variance()

    for _ in range(10):
        m(x)

    final_var = load_variance()

    assert final_var <= initial_var, (
        f"Expected variance to decrease or stay the same after bias updates. "
        f"Initial: {initial_var:.6f}, Final: {final_var:.6f}"
    )


def test_interface_compatible_with_sparse_moe(config, moe_cfg, x):
    """BalancedMoEFFN must be a drop-in replacement for SparseMoEFFN.

    Both should return a tuple (output, aux_loss) where output has the same
    shape as the input and aux_loss is a scalar tensor.
    """
    from src.model.moe import SparseMoEFFN

    sparse = SparseMoEFFN(config, moe_cfg)
    balanced = BalancedMoEFFN(config, moe_cfg)

    sparse_out, sparse_aux = sparse(x)
    balanced_out, balanced_aux = balanced(x)

    assert isinstance(sparse_out, torch.Tensor)
    assert isinstance(balanced_out, torch.Tensor)
    assert isinstance(sparse_aux, torch.Tensor)
    assert isinstance(balanced_aux, torch.Tensor)

    assert sparse_out.shape == x.shape
    assert balanced_out.shape == x.shape

    assert sparse_aux.shape == torch.Size([])
    assert balanced_aux.shape == torch.Size([])
