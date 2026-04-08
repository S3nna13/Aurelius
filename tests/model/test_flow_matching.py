"""Tests for src/model/flow_matching.py"""

import torch
import pytest

from aurelius.model.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingSchedule,
    VelocityNetwork,
    flow_matching_loss,
    ODESolver,
    FlowMatchingTrainer,
)

# Use small config for all tests
CFG = FlowMatchingConfig(embed_dim=32, n_timesteps=10)
DEVICE = torch.device("cpu")
B, D = 4, CFG.embed_dim


@pytest.fixture
def schedule():
    return FlowMatchingSchedule(sigma_min=CFG.sigma_min)


@pytest.fixture
def velocity_net():
    return VelocityNetwork(embed_dim=D)


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------

def test_interpolate_at_t0_is_noise(schedule):
    """t=0 => x_t should equal x_0 (up to sigma_min scaling)."""
    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)
    t = torch.zeros(B)
    x_t, _ = schedule.interpolate(x_0, x_1, t)
    # alpha at t=0 is 1.0, contribution of x_1 is 0
    # x_t = 1.0 * x_0 + 0 * x_1 = x_0
    assert torch.allclose(x_t, x_0, atol=1e-5), f"Max diff: {(x_t - x_0).abs().max()}"


def test_interpolate_at_t1_is_data(schedule):
    """t=1 => x_t should equal x_1 exactly."""
    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)
    t = torch.ones(B)
    x_t, _ = schedule.interpolate(x_0, x_1, t)
    # alpha at t=1 is sigma_min, so x_t = sigma_min * x_0 + x_1
    # The hard constraint says x_t = (1-(1-sigma_min)*1)*x_0 + 1*x_1
    #                                = sigma_min * x_0 + x_1
    expected = CFG.sigma_min * x_0 + x_1
    assert torch.allclose(x_t, expected, atol=1e-5)


def test_target_velocity_is_difference(schedule):
    """target velocity = x_1 - x_0."""
    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)
    t = torch.rand(B)
    _, v_target = schedule.interpolate(x_0, x_1, t)
    expected = x_1 - x_0
    assert torch.allclose(v_target, expected, atol=1e-6)


def test_sample_timesteps_uniform_range(schedule):
    """Uniform samples must all be in [0, 1]."""
    t = schedule.sample_timesteps(1000, DEVICE, mode="uniform")
    assert t.shape == (1000,)
    assert t.min().item() >= 0.0
    assert t.max().item() <= 1.0


def test_sample_timesteps_cosine_range(schedule):
    """Cosine samples must all be in [0, 1]."""
    t = schedule.sample_timesteps(1000, DEVICE, mode="cosine")
    assert t.shape == (1000,)
    assert t.min().item() >= 0.0
    assert t.max().item() <= 1.0


# ---------------------------------------------------------------------------
# VelocityNetwork tests
# ---------------------------------------------------------------------------

def test_velocity_network_output_shape_2d(velocity_net):
    """(B, D) input -> (B, D) output."""
    x = torch.randn(B, D)
    t = torch.rand(B)
    out = velocity_net(x, t)
    assert out.shape == (B, D), f"Expected {(B, D)}, got {out.shape}"


def test_velocity_network_output_shape_3d(velocity_net):
    """(B, T, D) input -> (B, T, D) output."""
    T = 8
    x = torch.randn(B, T, D)
    t = torch.rand(B)
    out = velocity_net(x, t)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Loss test
# ---------------------------------------------------------------------------

def test_flow_matching_loss_positive(velocity_net, schedule):
    """CFM loss should be > 0 for random data."""
    x_1 = torch.randn(B, D)
    loss = flow_matching_loss(velocity_net, x_1, schedule)
    assert loss.item() > 0.0
    assert loss.shape == ()  # scalar


# ---------------------------------------------------------------------------
# ODE Solver tests
# ---------------------------------------------------------------------------

def test_ode_solver_euler_output_shape(velocity_net):
    """Euler solver: sample((B, D)) -> (B, D)."""
    solver = ODESolver(velocity_net, n_steps=CFG.n_timesteps, solver="euler")
    out = solver.sample((B, D), DEVICE)
    assert out.shape == (B, D)


def test_ode_solver_midpoint_output_shape(velocity_net):
    """Midpoint solver: sample((B, D)) -> (B, D)."""
    solver = ODESolver(velocity_net, n_steps=CFG.n_timesteps, solver="midpoint")
    out = solver.sample((B, D), DEVICE)
    assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_flow_matching_trainer_step():
    """train_step returns a dict with 'loss' key."""
    net = VelocityNetwork(embed_dim=D)
    sched = FlowMatchingSchedule(sigma_min=CFG.sigma_min)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    trainer = FlowMatchingTrainer(net, sched, opt)

    embeddings = torch.randn(B, D)
    result = trainer.train_step(embeddings)

    assert isinstance(result, dict)
    assert "loss" in result
    assert isinstance(result["loss"], float)


def test_flow_matching_generate_shape():
    """generate(n, device) -> (n, embed_dim)."""
    net = VelocityNetwork(embed_dim=D)
    sched = FlowMatchingSchedule(sigma_min=CFG.sigma_min)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    trainer = FlowMatchingTrainer(net, sched, opt)

    n = 4
    out = trainer.generate(n, DEVICE)
    assert out.shape == (n, D), f"Expected ({n}, {D}), got {out.shape}"
