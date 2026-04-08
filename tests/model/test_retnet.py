"""Tests for RetNet-style retention layers."""

import pytest
import torch

from src.model.retnet import MultiScaleRetention, RetNetConfig


def make_module() -> MultiScaleRetention:
    config = RetNetConfig(d_model=64, n_heads=4, head_dim=16)
    return MultiScaleRetention(config)


def test_retnet_parallel_shape():
    module = make_module()
    x = torch.randn(2, 6, 64)
    y = module.forward_parallel(x)
    assert y.shape == x.shape


def test_retnet_recurrent_matches_parallel():
    module = make_module()
    x = torch.randn(2, 5, 64)
    parallel = module.forward_parallel(x)
    recurrent, _ = module.forward_recurrent(x)
    assert torch.allclose(parallel, recurrent, atol=1e-5)


def test_retnet_chunked_recurrent_matches_full_recurrent():
    module = make_module()
    x = torch.randn(2, 6, 64)
    full, _ = module.forward_recurrent(x)
    first, state = module.forward_recurrent(x[:, :3])
    second, _ = module.forward_recurrent(x[:, 3:], state)
    chunked = torch.cat([first, second], dim=1)
    assert torch.allclose(full, chunked, atol=1e-5)


def test_retnet_forward_step_updates_state_shape():
    module = make_module()
    token = torch.randn(2, 64)
    output, state = module.forward_step(token)
    assert output.shape == (2, 64)
    assert state.shape == (2, 4, 16, 16)


def test_retnet_backward_produces_gradients():
    module = make_module()
    x = torch.randn(2, 4, 64, requires_grad=True)
    loss = module(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert module.q_proj.weight.grad is not None


def test_retnet_rejects_bad_config():
    with pytest.raises(ValueError):
        MultiScaleRetention(RetNetConfig(d_model=60, n_heads=4, head_dim=16))


def test_retnet_rejects_bad_step_input_rank():
    module = make_module()
    with pytest.raises(ValueError):
        module.forward_step(torch.randn(2, 1, 64))
