"""Tests for Mamba-2 SSD v2 — pure PyTorch implementation.

Import path: aurelius.model.mamba2_v2

Tiny test config: B=2, T=16, d_model=32, d_state=8, n_heads=4,
                  n_layers=2, vocab_size=64, chunk_size=8.
"""

import pytest
import torch
from aurelius.model.mamba2_v2 import Mamba2Block, Mamba2Config, Mamba2Model, SSMKernel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B = 2
T = 16
D_MODEL = 32
D_STATE = 8
N_HEADS = 4
N_LAYERS = 2
VOCAB_SIZE = 64
CHUNK_SIZE = 8


@pytest.fixture
def cfg() -> Mamba2Config:
    return Mamba2Config(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=4,
        expand=2,
        n_heads=N_HEADS,
        chunk_size=CHUNK_SIZE,
    )


@pytest.fixture
def block(cfg) -> Mamba2Block:
    return Mamba2Block(cfg)


@pytest.fixture
def model(cfg) -> Mamba2Model:
    return Mamba2Model(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        vocab_size=VOCAB_SIZE,
        config=cfg,
    )


def make_ssm_kernel(cfg) -> SSMKernel:
    return SSMKernel(
        d_state=cfg.d_state,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
    )


def make_ssm_inputs(cfg, batch=B, seq=T):
    nh, dh, ds = cfg.n_heads, cfg.head_dim, cfg.d_state
    x = torch.randn(batch, seq, nh, dh)
    A = torch.randn(batch, seq, nh)
    B_mat = torch.randn(batch, seq, nh, ds)
    C_mat = torch.randn(batch, seq, nh, ds)
    return x, A, B_mat, C_mat


# ---------------------------------------------------------------------------
# SSMKernel tests
# ---------------------------------------------------------------------------


def test_ssm_output_shape(cfg):
    kernel = make_ssm_kernel(cfg)
    x, A, B_mat, C_mat = make_ssm_inputs(cfg)
    y = kernel(x, A, B_mat, C_mat)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"


def test_ssm_output_finite(cfg):
    kernel = make_ssm_kernel(cfg)
    x, A, B_mat, C_mat = make_ssm_inputs(cfg)
    y = kernel(x, A, B_mat, C_mat)
    assert torch.isfinite(y).all()


def test_ssm_different_inputs_different_outputs(cfg):
    kernel = make_ssm_kernel(cfg)
    x1, A, B_mat, C_mat = make_ssm_inputs(cfg)
    x2 = x1 + torch.randn_like(x1) * 0.5
    y1 = kernel(x1, A, B_mat, C_mat)
    y2 = kernel(x2, A, B_mat, C_mat)
    assert not torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Mamba2Block tests
# ---------------------------------------------------------------------------


def test_block_output_shape(block):
    x = torch.randn(B, T, D_MODEL)
    y = block(x)
    assert y.shape == (B, T, D_MODEL)


def test_block_output_finite(block):
    x = torch.randn(B, T, D_MODEL)
    y = block(x)
    assert torch.isfinite(y).all()


def test_block_gradient_flows(cfg):
    blk = Mamba2Block(cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    y = blk(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_block_different_inputs(block):
    x1 = torch.randn(B, T, D_MODEL)
    x2 = torch.randn(B, T, D_MODEL)
    y1 = block(x1)
    y2 = block(x2)
    assert not torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Mamba2Model tests
# ---------------------------------------------------------------------------


def test_model_output_shape(model):
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE)


def test_model_output_finite(model):
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    logits = model(ids)
    assert torch.isfinite(logits).all()


def test_model_gradient_to_embedding(cfg):
    mdl = Mamba2Model(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB_SIZE, config=cfg)
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    logits = mdl(ids)
    logits.sum().backward()
    assert mdl.embedding.weight.grad is not None
    assert torch.isfinite(mdl.embedding.weight.grad).all()


def test_model_b1_t1(cfg):
    mdl = Mamba2Model(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB_SIZE, config=cfg)
    ids = torch.randint(0, VOCAB_SIZE, (1, 1))
    logits = mdl(ids)
    assert logits.shape == (1, 1, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_model_t_not_multiple_of_chunk_size(cfg):
    T_odd = 13
    mdl = Mamba2Model(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB_SIZE, config=cfg)
    ids = torch.randint(0, VOCAB_SIZE, (B, T_odd))
    logits = mdl(ids)
    assert logits.shape == (B, T_odd, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_model_t_longer_than_chunk_size(cfg):
    T_long = 32
    mdl = Mamba2Model(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB_SIZE, config=cfg)
    ids = torch.randint(0, VOCAB_SIZE, (B, T_long))
    logits = mdl(ids)
    assert logits.shape == (B, T_long, VOCAB_SIZE)
    assert torch.isfinite(logits).all()
