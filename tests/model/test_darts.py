"""Tests for DARTS-style Neural Architecture Search (src/model/darts.py)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.darts import (
    DARTSConfig,
    DARTSCell,
    DARTSNetwork,
    DARTSTrainer,
    MixedOp,
    compute_arch_entropy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> DARTSConfig:
    return DARTSConfig()


@pytest.fixture
def small_config() -> DARTSConfig:
    return DARTSConfig(n_ops=4, n_cells=3, d_model=32, temperature=1.0, arch_lr=3e-4)


# ---------------------------------------------------------------------------
# 1. DARTSConfig defaults
# ---------------------------------------------------------------------------

def test_darts_config_defaults():
    cfg = DARTSConfig()
    assert cfg.n_ops == 4
    assert cfg.n_cells == 4
    assert cfg.d_model == 64
    assert cfg.temperature == 1.0
    assert cfg.arch_lr == 3e-4


def test_darts_config_custom():
    cfg = DARTSConfig(n_ops=2, n_cells=6, d_model=128, temperature=0.5, arch_lr=1e-3)
    assert cfg.n_ops == 2
    assert cfg.n_cells == 6
    assert cfg.d_model == 128
    assert cfg.temperature == 0.5
    assert cfg.arch_lr == 1e-3


# ---------------------------------------------------------------------------
# 2. MixedOp — output shape matches input
# ---------------------------------------------------------------------------

def test_mixed_op_output_shape():
    d_model, n_ops = 64, 4
    op = MixedOp(d_model, n_ops)
    x = torch.randn(2, 10, d_model)
    weights = torch.softmax(torch.ones(n_ops), dim=-1)
    out = op(x, weights)
    assert out.shape == x.shape


def test_mixed_op_output_shape_2d():
    d_model, n_ops = 32, 3
    op = MixedOp(d_model, n_ops)
    x = torch.randn(8, d_model)
    weights = torch.softmax(torch.ones(n_ops), dim=-1)
    out = op(x, weights)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 3. MixedOp weighted sum with uniform weights
# ---------------------------------------------------------------------------

def test_mixed_op_uniform_weights_is_mean():
    """With uniform weights, output should equal the uniform average of op outputs."""
    d_model, n_ops = 16, 4
    op = MixedOp(d_model, n_ops)
    x = torch.randn(1, d_model)
    uniform_w = torch.full((n_ops,), 1.0 / n_ops)

    out = op(x, uniform_w)

    # Manually compute expected
    expected = sum(uniform_w[i].item() * o(x) for i, o in enumerate(op.ops))
    assert torch.allclose(out, expected, atol=1e-5)


def test_mixed_op_weights_sum_to_one_respected():
    """MixedOp does not re-normalize weights — caller is responsible."""
    d_model, n_ops = 16, 4
    op = MixedOp(d_model, n_ops)
    x = torch.randn(2, d_model)
    w1 = torch.softmax(torch.zeros(n_ops), dim=-1)   # uniform
    w2 = torch.softmax(torch.zeros(n_ops), dim=-1)
    out1 = op(x, w1)
    out2 = op(x, w2)
    assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. DARTSCell forward shape
# ---------------------------------------------------------------------------

def test_darts_cell_forward_shape():
    cell = DARTSCell(d_model=64, n_ops=4)
    x = torch.randn(2, 10, 64)
    out = cell(x)
    assert out.shape == x.shape


def test_darts_cell_forward_shape_2d():
    cell = DARTSCell(d_model=32, n_ops=3)
    x = torch.randn(4, 32)
    out = cell(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 5. DARTSCell arch_params are learnable parameters
# ---------------------------------------------------------------------------

def test_darts_cell_arch_params_is_parameter():
    cell = DARTSCell(d_model=64, n_ops=4)
    assert isinstance(cell.arch_params, nn.Parameter)
    assert cell.arch_params.shape == (4,)
    assert cell.arch_params.requires_grad


def test_darts_cell_arch_params_shape_custom():
    cell = DARTSCell(d_model=32, n_ops=2)
    assert cell.arch_params.shape == (2,)


# ---------------------------------------------------------------------------
# 6. DARTSNetwork forward shape
# ---------------------------------------------------------------------------

def test_darts_network_forward_shape(default_config):
    net = DARTSNetwork(default_config)
    x = torch.randn(2, 5, default_config.d_model)
    out = net(x)
    assert out.shape == x.shape


def test_darts_network_forward_shape_small(small_config):
    net = DARTSNetwork(small_config)
    x = torch.randn(3, 7, small_config.d_model)
    out = net(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 7. DARTSNetwork arch_parameters returns list
# ---------------------------------------------------------------------------

def test_darts_network_arch_parameters_is_list(default_config):
    net = DARTSNetwork(default_config)
    params = net.arch_parameters()
    assert isinstance(params, list)
    assert len(params) == default_config.n_cells


def test_darts_network_arch_parameters_are_parameters(default_config):
    net = DARTSNetwork(default_config)
    for p in net.arch_parameters():
        assert isinstance(p, nn.Parameter)


# ---------------------------------------------------------------------------
# 8. DARTSNetwork discretize returns list of ints
# ---------------------------------------------------------------------------

def test_darts_network_discretize_returns_list(default_config):
    net = DARTSNetwork(default_config)
    indices = net.discretize()
    assert isinstance(indices, list)
    assert len(indices) == default_config.n_cells


def test_darts_network_discretize_values_are_ints(default_config):
    net = DARTSNetwork(default_config)
    indices = net.discretize()
    for idx in indices:
        assert isinstance(idx, int)
        assert 0 <= idx < default_config.n_ops


# ---------------------------------------------------------------------------
# 9. DARTSTrainer weight_step returns scalar
# ---------------------------------------------------------------------------

def test_darts_trainer_weight_step_returns_scalar(default_config):
    net = DARTSNetwork(default_config)
    arch_p = net.arch_parameters()
    model_p = [p for p in net.parameters() if not any(p is a for a in arch_p)]
    trainer = DARTSTrainer(model_p, arch_p, default_config)

    x = torch.randn(2, 5, default_config.d_model)
    out = net(x)
    loss = out.mean()

    result = trainer.weight_step(loss)
    assert result.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# 10. DARTSTrainer arch_step returns scalar
# ---------------------------------------------------------------------------

def test_darts_trainer_arch_step_returns_scalar(default_config):
    net = DARTSNetwork(default_config)
    arch_p = net.arch_parameters()
    model_p = [p for p in net.parameters() if not any(p is a for a in arch_p)]
    trainer = DARTSTrainer(model_p, arch_p, default_config)

    x = torch.randn(2, 5, default_config.d_model)
    out = net(x)
    loss = out.mean()

    result = trainer.arch_step(loss)
    assert result.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# 11. compute_arch_entropy range [0, log(n_ops)]
# ---------------------------------------------------------------------------

def test_compute_arch_entropy_range():
    n_ops = 4
    # Uniform → maximum entropy
    uniform = torch.zeros(n_ops)
    entropy = compute_arch_entropy(uniform, temperature=1.0)
    assert 0.0 <= entropy.item() <= math.log(n_ops) + 1e-6


def test_compute_arch_entropy_uniform_is_max():
    n_ops = 4
    uniform = torch.zeros(n_ops)
    entropy = compute_arch_entropy(uniform, temperature=1.0)
    expected = math.log(n_ops)
    assert abs(entropy.item() - expected) < 1e-5


def test_compute_arch_entropy_one_hot_near_zero():
    """Very peaked distribution should have near-zero entropy."""
    n_ops = 4
    peaked = torch.tensor([100.0, 0.0, 0.0, 0.0])
    entropy = compute_arch_entropy(peaked, temperature=1.0)
    assert entropy.item() < 0.01


# ---------------------------------------------------------------------------
# 12. High temperature → higher entropy
# ---------------------------------------------------------------------------

def test_high_temperature_higher_entropy():
    n_ops = 4
    # Use a non-uniform logit vector so temperature has a visible effect
    logits = torch.tensor([2.0, 0.5, -0.5, -2.0])
    low_temp_entropy = compute_arch_entropy(logits, temperature=0.1)
    high_temp_entropy = compute_arch_entropy(logits, temperature=10.0)
    assert high_temp_entropy.item() > low_temp_entropy.item()


def test_temperature_monotone_entropy():
    """Increasing temperature monotonically increases entropy for non-uniform logits."""
    logits = torch.tensor([3.0, 1.0, 0.0, -1.0])
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    entropies = [compute_arch_entropy(logits, t).item() for t in temperatures]
    for i in range(len(entropies) - 1):
        assert entropies[i] <= entropies[i + 1] + 1e-6


# ---------------------------------------------------------------------------
# Bonus: gradient flow through DARTSNetwork
# ---------------------------------------------------------------------------

def test_darts_network_gradients_flow(default_config):
    net = DARTSNetwork(default_config)
    x = torch.randn(2, 5, default_config.d_model)
    out = net(x)
    loss = out.mean()
    loss.backward()
    for cell in net.cells:
        assert cell.arch_params.grad is not None
