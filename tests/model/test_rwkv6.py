"""Tests for RWKV-6 (Eagle) time-mixing with matrix-valued states.

Reference: Peng et al. 2024, "Eagle and Finch: RWKV with Matrix-Valued States
and Dynamic Recurrence". https://arxiv.org/abs/2404.05892

All tests use small configs to keep CI fast:
  d_model=32, n_heads=4, n_layers=2, batch=2, seq_len=8, vocab_size=256
"""

from __future__ import annotations

import pytest
import torch
from aurelius.model.rwkv6 import (
    RWKV6Block,
    RWKV6ChannelMix,
    RWKV6Model,
    RWKV6TimeMix,
)

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
BATCH = 2
SEQ_LEN = 8
VOCAB_SIZE = 256


@pytest.fixture
def time_mix():
    torch.manual_seed(0)
    return RWKV6TimeMix(D_MODEL, N_HEADS)


@pytest.fixture
def channel_mix():
    torch.manual_seed(0)
    return RWKV6ChannelMix(D_MODEL)


@pytest.fixture
def block():
    torch.manual_seed(0)
    return RWKV6Block(D_MODEL, N_HEADS)


@pytest.fixture
def rwkv_model():
    torch.manual_seed(0)
    return RWKV6Model(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS)


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


@pytest.fixture
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


def test_time_mix_output_shape(time_mix, x):
    out = time_mix(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_time_mix_output_finite(time_mix, x):
    out = time_mix(x)
    assert torch.isfinite(out).all()


def test_time_mix_causal(time_mix):
    torch.manual_seed(7)
    x_short = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    x_long = torch.cat([x_short, torch.randn(BATCH, 4, D_MODEL)], dim=1)
    time_mix.eval()
    with torch.no_grad():
        out_short = time_mix(x_short)
        out_long = time_mix(x_long)
    assert torch.allclose(out_short, out_long[:, :SEQ_LEN, :], atol=1e-5)


def test_time_mix_gradients(time_mix, x):
    x_grad = x.clone().requires_grad_(True)
    out = time_mix(x_grad)
    out.sum().backward()
    assert x_grad.grad is not None
    assert torch.isfinite(x_grad.grad).all()


def test_time_mix_single_token(time_mix):
    torch.manual_seed(3)
    x_single = torch.randn(1, 1, D_MODEL)
    out = time_mix(x_single)
    assert out.shape == (1, 1, D_MODEL)
    assert torch.isfinite(out).all()


def test_channel_mix_output_shape(channel_mix, x):
    out = channel_mix(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_channel_mix_output_finite(channel_mix, x):
    out = channel_mix(x)
    assert torch.isfinite(out).all()


def test_channel_mix_gradients(channel_mix, x):
    x_grad = x.clone().requires_grad_(True)
    out = channel_mix(x_grad)
    out.sum().backward()
    assert x_grad.grad is not None
    assert torch.isfinite(x_grad.grad).all()


def test_block_output_shape(block, x):
    out = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_block_output_finite(block, x):
    out = block(x)
    assert torch.isfinite(out).all()


def test_block_residual_non_trivial(block, x):
    out = block(x)
    assert not torch.allclose(out, x, atol=1e-6)


def test_model_output_shape(rwkv_model, input_ids):
    out = rwkv_model(input_ids)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_model_output_finite(rwkv_model, input_ids):
    out = rwkv_model(input_ids)
    assert torch.isfinite(out).all()


def test_model_gradients(rwkv_model, input_ids):
    out = rwkv_model(input_ids)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in rwkv_model.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()
