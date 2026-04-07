"""Tests for Mixture of Experts FFN."""
import torch
import pytest
from src.model.moe import SparseMoEFFN, MoEConfig
from src.model.config import AureliusConfig


@pytest.fixture
def small_config():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )


def test_moe_output_shape(small_config):
    """SparseMoEFFN must return output of same shape as input."""
    moe = SparseMoEFFN(small_config, MoEConfig(n_experts=4, top_k=2))
    x = torch.randn(2, 8, 64)
    out, aux = moe(x)
    assert out.shape == (2, 8, 64)
    assert out.dtype == x.dtype


def test_moe_aux_loss_scalar(small_config):
    """Auxiliary loss must be a finite positive scalar."""
    moe = SparseMoEFFN(small_config, MoEConfig(n_experts=4, top_k=2))
    x = torch.randn(2, 8, 64)
    _, aux = moe(x)
    assert aux.ndim == 0
    assert torch.isfinite(aux)
    assert aux.item() > 0


def test_moe_aux_loss_backprop(small_config):
    """Gradients must flow through aux_loss to router weights."""
    moe = SparseMoEFFN(small_config, MoEConfig(n_experts=4, top_k=2))
    x = torch.randn(2, 4, 64)
    out, aux = moe(x)
    loss = out.sum() + aux
    loss.backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0


def test_moe_top_k_1(small_config):
    """top_k=1 (hard routing) must still produce valid outputs."""
    moe = SparseMoEFFN(small_config, MoEConfig(n_experts=4, top_k=1))
    x = torch.randn(3, 6, 64)
    out, aux = moe(x)
    assert out.shape == (3, 6, 64)
    assert torch.isfinite(out).all()


def test_moe_load_balance_encourages_spread(small_config):
    """After multiple steps, load balance loss should push toward uniform routing."""
    import torch.optim as optim
    moe = SparseMoEFFN(small_config, MoEConfig(n_experts=4, top_k=2, load_balance_alpha=1.0))
    optimizer = optim.Adam(moe.parameters(), lr=0.01)

    # Train only on aux loss to force balanced routing
    for _ in range(20):
        x = torch.randn(8, 16, 64)
        _, aux = moe(x)
        optimizer.zero_grad()
        aux.backward()
        optimizer.step()

    # Check utilization is more uniform
    x_eval = torch.randn(16, 32, 64)
    util = moe.expert_utilization(x_eval)
    # All experts should have some utilization (not dead)
    assert all(v > 0 for v in util.values()), f"Dead experts found: {util}"


def test_moe_n_experts_1(small_config):
    """n_experts=1 is a dense FFN (all tokens to one expert)."""
    moe = SparseMoEFFN(small_config, MoEConfig(n_experts=1, top_k=1))
    x = torch.randn(2, 4, 64)
    out, aux = moe(x)
    assert out.shape == (2, 4, 64)
    assert torch.isfinite(out).all()
