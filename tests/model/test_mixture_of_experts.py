"""Tests for src/model/mixture_of_experts.py"""
import pytest
import torch
import torch.nn as nn

from src.model.mixture_of_experts import (
    MoEConfig,
    ExpertFFN,
    TopKRouter,
    compute_load_balancing_loss,
    SparseMoELayer,
    MoEBlock,
)

# ---------------------------------------------------------------------------
# Tiny constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
N_EXPERTS = 4
TOP_K = 2
BATCH = 2
SEQ = 6


@pytest.fixture
def cfg():
    return MoEConfig(
        d_model=D_MODEL,
        d_ff=D_FF,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        aux_loss_coef=0.01,
    )


@pytest.fixture
def input_tensor():
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# MoEConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = MoEConfig()
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.aux_loss_coef == 0.01


# ---------------------------------------------------------------------------
# ExpertFFN
# ---------------------------------------------------------------------------

def test_expert_ffn_output_shape(cfg):
    expert = ExpertFFN(D_MODEL, D_FF)
    x = torch.randn(BATCH * SEQ, D_MODEL)
    out = expert(x)
    assert out.shape == (BATCH * SEQ, D_MODEL)


def test_expert_ffn_gradient_flows(cfg):
    expert = ExpertFFN(D_MODEL, D_FF)
    x = torch.randn(BATCH * SEQ, D_MODEL, requires_grad=True)
    out = expert(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# TopKRouter
# ---------------------------------------------------------------------------

def test_router_probs_shape(cfg):
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    probs, indices, weights = router(x)
    N = BATCH * SEQ
    assert probs.shape == (N, N_EXPERTS)
    assert indices.shape == (N, TOP_K)
    assert weights.shape == (N, TOP_K)


def test_router_probs_sum_to_one(cfg):
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    probs, _, _ = router(x)
    # Softmax probs sum to 1 per token
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH * SEQ), atol=1e-5)


def test_router_weights_sum_to_one(cfg):
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, _, weights = router(x)
    # Top-k weights normalized to sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH * SEQ), atol=1e-5)


def test_router_indices_in_range(cfg):
    router = TopKRouter(D_MODEL, N_EXPERTS, TOP_K)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, indices, _ = router(x)
    assert (indices >= 0).all() and (indices < N_EXPERTS).all()


# ---------------------------------------------------------------------------
# compute_load_balancing_loss
# ---------------------------------------------------------------------------

def test_aux_loss_is_scalar(cfg):
    N = BATCH * SEQ
    probs = torch.softmax(torch.randn(N, N_EXPERTS), dim=-1)
    indices = torch.randint(0, N_EXPERTS, (N, TOP_K))
    loss = compute_load_balancing_loss(probs, indices, N_EXPERTS)
    assert loss.shape == ()


def test_aux_loss_nonneg(cfg):
    N = BATCH * SEQ
    probs = torch.softmax(torch.randn(N, N_EXPERTS), dim=-1)
    indices = torch.randint(0, N_EXPERTS, (N, TOP_K))
    loss = compute_load_balancing_loss(probs, indices, N_EXPERTS)
    assert loss.item() >= 0.0


def test_aux_loss_has_gradient(cfg):
    N = BATCH * SEQ
    logits = torch.randn(N, N_EXPERTS, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    indices = probs.topk(TOP_K, dim=-1).indices.detach()
    loss = compute_load_balancing_loss(probs, indices, N_EXPERTS)
    loss.backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# SparseMoELayer
# ---------------------------------------------------------------------------

def test_sparse_moe_output_shape(cfg, input_tensor):
    moe = SparseMoELayer(cfg)
    out, aux = moe(input_tensor)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_sparse_moe_aux_loss_scalar(cfg, input_tensor):
    moe = SparseMoELayer(cfg)
    _, aux = moe(input_tensor)
    assert aux.shape == ()


def test_sparse_moe_aux_loss_nonneg(cfg, input_tensor):
    moe = SparseMoELayer(cfg)
    _, aux = moe(input_tensor)
    assert aux.item() >= 0.0


def test_sparse_moe_gradient_flows(cfg, input_tensor):
    moe = SparseMoELayer(cfg)
    x = input_tensor.clone().requires_grad_(True)
    out, aux = moe(x)
    (out.sum() + aux).backward()
    assert x.grad is not None


def test_sparse_moe_output_finite(cfg, input_tensor):
    moe = SparseMoELayer(cfg)
    out, _ = moe(input_tensor)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MoEBlock
# ---------------------------------------------------------------------------

def test_moe_block_output_shape(cfg, input_tensor):
    block = MoEBlock(cfg)
    out, aux = block(input_tensor)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_moe_block_residual(cfg, input_tensor):
    torch.manual_seed(1)
    block = MoEBlock(cfg)
    out, _ = block(input_tensor)
    # With random init, output != input (residual adds non-zero)
    assert not torch.allclose(out, input_tensor, atol=1e-4)


def test_moe_block_gradient_flows(cfg, input_tensor):
    block = MoEBlock(cfg)
    x = input_tensor.clone().requires_grad_(True)
    out, aux = block(x)
    (out.sum() + aux).backward()
    assert x.grad is not None
