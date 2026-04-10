"""Tests for src/model/neural_ode.py."""

import pytest
import torch

from src.model.neural_ode import (
    ODEConfig,
    ODEFunction,
    NeuralODEBlock,
    ODETransformerBlock,
    compute_trajectory,
    euler_solve,
    rk4_solve,
    midpoint_solve,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, D = 2, 8, 64  # batch, seq_len, d_model


@pytest.fixture()
def default_config() -> ODEConfig:
    return ODEConfig()


@pytest.fixture()
def sample_x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D)


# ---------------------------------------------------------------------------
# 1. ODEConfig defaults
# ---------------------------------------------------------------------------

def test_ode_config_defaults(default_config: ODEConfig) -> None:
    assert default_config.n_steps == 4
    assert default_config.step_size == 0.25
    assert default_config.d_model == 64
    assert default_config.solver == "euler"
    assert default_config.augment_dim == 0


# ---------------------------------------------------------------------------
# 2. ODEFunction output shape matches input
# ---------------------------------------------------------------------------

def test_ode_function_output_shape(sample_x: torch.Tensor) -> None:
    fn = ODEFunction(d_model=D)
    dhdt = fn(t=0.0, h=sample_x)
    assert dhdt.shape == sample_x.shape, f"Expected {sample_x.shape}, got {dhdt.shape}"


# ---------------------------------------------------------------------------
# 3. ODEFunction time broadcast causes no error
# ---------------------------------------------------------------------------

def test_ode_function_time_broadcast(sample_x: torch.Tensor) -> None:
    fn = ODEFunction(d_model=D)
    for t_val in [0.0, 0.5, 1.0]:
        out = fn(t=t_val, h=sample_x)
        assert out.shape == sample_x.shape


# ---------------------------------------------------------------------------
# 4. euler_solve returns same shape as h0
# ---------------------------------------------------------------------------

def test_euler_solve_shape(sample_x: torch.Tensor) -> None:
    fn = ODEFunction(d_model=D)
    result = euler_solve(fn, sample_x, t0=0.0, t1=1.0, n_steps=4)
    assert result.shape == sample_x.shape


# ---------------------------------------------------------------------------
# 5. euler_solve with n_steps=1 changes h0
# ---------------------------------------------------------------------------

def test_euler_solve_changes_state(sample_x: torch.Tensor) -> None:
    fn = ODEFunction(d_model=D)
    result = euler_solve(fn, sample_x, t0=0.0, t1=1.0, n_steps=1)
    assert not torch.allclose(result, sample_x), "Euler step should change the state"


# ---------------------------------------------------------------------------
# 6. rk4_solve returns same shape as h0
# ---------------------------------------------------------------------------

def test_rk4_solve_shape(sample_x: torch.Tensor) -> None:
    fn = ODEFunction(d_model=D)
    result = rk4_solve(fn, sample_x, t0=0.0, t1=1.0, n_steps=4)
    assert result.shape == sample_x.shape


# ---------------------------------------------------------------------------
# 7. midpoint_solve returns same shape as h0
# ---------------------------------------------------------------------------

def test_midpoint_solve_shape(sample_x: torch.Tensor) -> None:
    fn = ODEFunction(d_model=D)
    result = midpoint_solve(fn, sample_x, t0=0.0, t1=1.0, n_steps=4)
    assert result.shape == sample_x.shape


# ---------------------------------------------------------------------------
# 8. NeuralODEBlock output shape matches input
# ---------------------------------------------------------------------------

def test_neural_ode_block_output_shape(sample_x: torch.Tensor) -> None:
    block = NeuralODEBlock(ODEConfig(d_model=D))
    out = block(sample_x)
    assert out.shape == sample_x.shape


# ---------------------------------------------------------------------------
# 9. NeuralODEBlock output differs from input (integration changes state)
# ---------------------------------------------------------------------------

def test_neural_ode_block_changes_state(sample_x: torch.Tensor) -> None:
    block = NeuralODEBlock(ODEConfig(d_model=D))
    out = block(sample_x)
    assert not torch.allclose(out, sample_x), "ODE block should transform the state"


# ---------------------------------------------------------------------------
# 10. Euler vs RK4 give different results (different numerical accuracy)
# ---------------------------------------------------------------------------

def test_euler_vs_rk4_differ(sample_x: torch.Tensor) -> None:
    cfg_euler = ODEConfig(d_model=D, solver="euler", n_steps=4)
    cfg_rk4 = ODEConfig(d_model=D, solver="rk4", n_steps=4)

    torch.manual_seed(42)
    block_euler = NeuralODEBlock(cfg_euler)

    # Share the same ODE function weights for a fair comparison
    cfg_rk4_shared = ODEConfig(d_model=D, solver="rk4", n_steps=4)
    block_rk4 = NeuralODEBlock(cfg_rk4_shared)
    block_rk4.ode_func.load_state_dict(block_euler.ode_func.state_dict())

    out_euler = block_euler(sample_x)
    out_rk4 = block_rk4(sample_x)
    assert not torch.allclose(out_euler, out_rk4, atol=1e-5), (
        "Euler and RK4 should produce different trajectories"
    )


# ---------------------------------------------------------------------------
# 11. ODETransformerBlock output shape (B, T, D)
# ---------------------------------------------------------------------------

def test_ode_transformer_block_output_shape(sample_x: torch.Tensor) -> None:
    cfg = ODEConfig(d_model=D)
    block = ODETransformerBlock(d_model=D, config=cfg)
    out = block(sample_x)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 12. ODETransformerBlock has residual (output != bare ODE output)
# ---------------------------------------------------------------------------

def test_ode_transformer_block_has_residual(sample_x: torch.Tensor) -> None:
    cfg = ODEConfig(d_model=D)
    block = ODETransformerBlock(d_model=D, config=cfg)

    # Compute the raw ODE output through the same sub-modules
    normed = block.norm(sample_x)
    ode_only = block.ode(normed)

    full_out = block(sample_x)
    assert not torch.allclose(full_out, ode_only, atol=1e-6), (
        "ODETransformerBlock output should differ from bare ODE output (residual present)"
    )


# ---------------------------------------------------------------------------
# 13. compute_trajectory returns list of n_snapshots tensors
# ---------------------------------------------------------------------------

def test_compute_trajectory_count(sample_x: torch.Tensor) -> None:
    block = NeuralODEBlock(ODEConfig(d_model=D))
    n_snapshots = 4
    traj = compute_trajectory(block, sample_x, n_snapshots=n_snapshots)
    assert isinstance(traj, list)
    assert len(traj) == n_snapshots


# ---------------------------------------------------------------------------
# 14. compute_trajectory shapes are consistent with input
# ---------------------------------------------------------------------------

def test_compute_trajectory_shapes(sample_x: torch.Tensor) -> None:
    block = NeuralODEBlock(ODEConfig(d_model=D))
    traj = compute_trajectory(block, sample_x, n_snapshots=5)
    for i, snap in enumerate(traj):
        assert snap.shape == sample_x.shape, (
            f"Snapshot {i} has shape {snap.shape}, expected {sample_x.shape}"
        )
