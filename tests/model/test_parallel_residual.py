"""Tests for ParallelBlock (parallel attention + FFN residual)."""

import pytest
import torch

from src.model.parallel_residual import ParallelBlock, ParallelConfig


@pytest.fixture
def cfg():
    return ParallelConfig(d_model=64, n_heads=4, head_dim=16, d_ff=128, dropout=0.0)


@pytest.fixture
def block(cfg):
    return ParallelBlock(cfg)


def test_parallel_config_instantiates():
    cfg = ParallelConfig(d_model=128, n_heads=8, head_dim=16, d_ff=256)
    assert cfg.d_model == 128
    assert cfg.dropout == 0.0


def test_parallel_block_instantiates(cfg):
    block = ParallelBlock(cfg)
    assert isinstance(block, ParallelBlock)


def test_forward_output_shape(block, cfg):
    B, T = 2, 8
    x = torch.randn(B, T, cfg.d_model)
    out = block(x)
    assert out.shape == (B, T, cfg.d_model)


def test_output_is_finite(block, cfg):
    x = torch.randn(2, 8, cfg.d_model)
    out = block(x)
    assert torch.isfinite(out).all()


def test_output_differs_from_input(block, cfg):
    x = torch.randn(2, 8, cfg.d_model)
    out = block(x)
    assert not torch.allclose(out, x)


def test_gradient_flows_to_w1(block, cfg):
    x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert block.W1.weight.grad is not None
    assert block.W1.weight.grad.abs().sum() > 0


def test_gradient_flows_to_q_proj(block, cfg):
    x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert block.q_proj.weight.grad is not None
    assert block.q_proj.weight.grad.abs().sum() > 0


def test_works_with_b1_t1(cfg):
    block = ParallelBlock(cfg)
    x = torch.randn(1, 1, cfg.d_model)
    out = block(x)
    assert out.shape == (1, 1, cfg.d_model)
    assert torch.isfinite(out).all()


def test_works_with_t32(cfg):
    block = ParallelBlock(cfg)
    x = torch.randn(2, 32, cfg.d_model)
    out = block(x)
    assert out.shape == (2, 32, cfg.d_model)


def test_attn_and_ffn_contribute_independently(cfg):
    """Zero W2 (FFN down-proj) and verify output changes; same for o_proj."""
    torch.manual_seed(0)
    block_ref = ParallelBlock(cfg)
    x = torch.randn(1, 4, cfg.d_model)

    with torch.no_grad():
        out_ref = block_ref(x)

        # Zero FFN contribution
        block_no_ffn = ParallelBlock(cfg)
        block_no_ffn.load_state_dict(block_ref.state_dict())
        block_no_ffn.W2.weight.zero_()
        out_no_ffn = block_no_ffn(x)

        # Zero attn contribution
        block_no_attn = ParallelBlock(cfg)
        block_no_attn.load_state_dict(block_ref.state_dict())
        block_no_attn.o_proj.weight.zero_()
        out_no_attn = block_no_attn(x)

    assert not torch.allclose(out_ref, out_no_ffn), "FFN should contribute"
    assert not torch.allclose(out_ref, out_no_attn), "Attention should contribute"
