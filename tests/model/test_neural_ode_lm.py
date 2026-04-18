"""
test_neural_ode_lm.py

Tests for neural_ode_lm.py — Neural ODE continuous-depth sequence modeling.
All tests use small dimensions: d_model=16, vocab_size=16, B=2, T=6.
"""

from __future__ import annotations

import math
import torch
import pytest

from src.model.neural_ode_lm import (
    ODEFunc,
    EulerSolver,
    RK4Solver,
    NeuralODEBlock,
    NeuralODELanguageModel,
    NeuralODEConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
B = 2
T = 6
N_STEPS = 4


def make_z0() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D_MODEL)


def make_input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ===========================================================================
# ODEFunc tests
# ===========================================================================

def test_odefunc_output_shape():
    """ODEFunc forward output shape matches input [B, T, d_model]."""
    func = ODEFunc(D_MODEL)
    z0 = make_z0()
    dz = func(0.0, z0)
    assert dz.shape == z0.shape, f"Expected {z0.shape}, got {dz.shape}"


def test_odefunc_nfe_increments():
    """ODEFunc.nfe increments by 1 on each call."""
    func = ODEFunc(D_MODEL)
    assert func.nfe == 0
    func(0.0, make_z0())
    assert func.nfe == 1
    func(0.5, make_z0())
    assert func.nfe == 2
    func(1.0, make_z0())
    assert func.nfe == 3


def test_odefunc_time_conditioning():
    """ODEFunc output should differ when called with different time values."""
    torch.manual_seed(42)
    func = ODEFunc(D_MODEL)
    z0 = make_z0()
    dz_t0 = func(0.0, z0)
    dz_t1 = func(1.0, z0)
    assert not torch.allclose(dz_t0, dz_t1), \
        "ODEFunc outputs should differ for t=0 vs t=1 (time conditioning)"


# ===========================================================================
# EulerSolver tests
# ===========================================================================

def test_euler_output_shape():
    """EulerSolver.solve output shape matches z0."""
    func = ODEFunc(D_MODEL)
    solver = EulerSolver()
    z0 = make_z0()
    zT = solver.solve(func, z0, (0.0, 1.0), n_steps=N_STEPS)
    assert zT.shape == z0.shape, f"Expected {z0.shape}, got {zT.shape}"


def test_euler_single_step():
    """EulerSolver with n_steps=1 equals exactly one Euler step."""
    torch.manual_seed(7)
    func = ODEFunc(D_MODEL)
    z0 = make_z0()
    t0, t1 = 0.0, 1.0
    h = t1 - t0

    solver = EulerSolver()
    zT_solver = solver.solve(func, z0, (t0, t1), n_steps=1)

    # Manual single step
    func.nfe = 0
    with torch.no_grad():
        dz = func(t0, z0)
        zT_manual = z0 + h * dz

    assert torch.allclose(zT_solver, zT_manual, atol=1e-5), \
        "EulerSolver n_steps=1 should match manual one-step update"


def test_euler_output_differs_from_z0():
    """EulerSolver output should differ from the initial z0."""
    func = ODEFunc(D_MODEL)
    solver = EulerSolver()
    z0 = make_z0()
    zT = solver.solve(func, z0, (0.0, 1.0), n_steps=N_STEPS)
    assert not torch.allclose(zT, z0), \
        "EulerSolver output must differ from initial z0"


# ===========================================================================
# RK4Solver tests
# ===========================================================================

def test_rk4_output_shape():
    """RK4Solver.solve output shape matches z0."""
    func = ODEFunc(D_MODEL)
    solver = RK4Solver()
    z0 = make_z0()
    zT = solver.solve(func, z0, (0.0, 1.0), n_steps=N_STEPS)
    assert zT.shape == z0.shape, f"Expected {z0.shape}, got {zT.shape}"


def test_rk4_output_differs_from_z0():
    """RK4Solver output should differ from the initial z0."""
    func = ODEFunc(D_MODEL)
    solver = RK4Solver()
    z0 = make_z0()
    zT = solver.solve(func, z0, (0.0, 1.0), n_steps=N_STEPS)
    assert not torch.allclose(zT, z0), \
        "RK4Solver output must differ from initial z0"


def test_rk4_more_accurate_than_euler_for_simple_dynamics():
    """RK4 should be closer to the analytical solution than Euler on a simple linear ODE.

    We test on dz/dt = -z (analytical: z(t) = z0 * exp(-t)).
    We patch ODEFunc.forward temporarily with a known dynamics to isolate solver accuracy.
    """
    # Simple scalar: dz/dt = -z, z(0)=1, z(1) = exp(-1) ≈ 0.3679
    # We'll run 1-D (batch 1, seq 1, d_model=1) with identity-scaled linear func
    class LinearDecayFunc(torch.nn.Module):
        nfe = 0
        def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
            self.nfe += 1
            return -z  # dz/dt = -z

    func = LinearDecayFunc()
    z0 = torch.ones(1, 1, 1)
    t_span = (0.0, 1.0)
    n_steps = 10

    euler_solver = EulerSolver()
    rk4_solver = RK4Solver()

    z_euler = euler_solver.solve(func, z0, t_span, n_steps)
    func.nfe = 0  # reset
    z_rk4 = rk4_solver.solve(func, z0, t_span, n_steps)

    analytical = torch.exp(torch.tensor(-1.0))
    err_euler = (z_euler.squeeze() - analytical).abs().item()
    err_rk4 = (z_rk4.squeeze() - analytical).abs().item()

    assert err_rk4 < err_euler, (
        f"RK4 error ({err_rk4:.6f}) should be smaller than Euler error ({err_euler:.6f})"
    )


# ===========================================================================
# NeuralODEBlock tests
# ===========================================================================

def test_ode_block_output_shape():
    """NeuralODEBlock forward output shape is [B, T, d_model]."""
    block = NeuralODEBlock(D_MODEL, solver="rk4", n_steps=N_STEPS)
    x = make_z0()
    out = block(x)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


def test_ode_block_nfe_after_forward():
    """NeuralODEBlock.nfe() > 0 after a forward pass."""
    block = NeuralODEBlock(D_MODEL, solver="rk4", n_steps=N_STEPS)
    x = make_z0()
    _ = block(x)
    assert block.nfe() > 0, "nfe should be > 0 after forward pass"


def test_ode_block_gradient_flow():
    """Gradients must flow through NeuralODEBlock integration."""
    block = NeuralODEBlock(D_MODEL, solver="rk4", n_steps=2)
    x = make_z0().requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient w.r.t. input should not be None"
    assert x.grad.shape == x.shape
    assert not torch.all(x.grad == 0), "Gradients should not all be zero"


# ===========================================================================
# NeuralODELanguageModel tests
# ===========================================================================

def test_lm_forward_output_shape():
    """NeuralODELanguageModel forward output shape is [B, T, vocab_size]."""
    model = NeuralODELanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        n_ode_blocks=2, solver="rk4", n_steps=N_STEPS
    )
    ids = make_input_ids()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE), \
        f"Expected {(B, T, VOCAB_SIZE)}, got {logits.shape}"


def test_lm_compute_loss_finite_positive():
    """NeuralODELanguageModel.compute_loss returns a finite positive scalar."""
    model = NeuralODELanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        n_ode_blocks=2, solver="rk4", n_steps=N_STEPS
    )
    ids = make_input_ids()
    loss = model.compute_loss(ids)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert math.isfinite(loss.item()), f"Loss should be finite, got {loss.item()}"


def test_lm_compute_loss_backward():
    """Gradients must flow through NeuralODELanguageModel.compute_loss."""
    model = NeuralODELanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        n_ode_blocks=2, solver="rk4", n_steps=N_STEPS
    )
    ids = make_input_ids()
    loss = model.compute_loss(ids)
    loss.backward()
    # Check at least one parameter has a gradient
    has_grad = any(
        p.grad is not None and not torch.all(p.grad == 0)
        for p in model.parameters()
    )
    assert has_grad, "At least one parameter should have a non-zero gradient"


# ===========================================================================
# NeuralODEConfig tests
# ===========================================================================

def test_neural_ode_config_defaults():
    """NeuralODEConfig should have the expected default values."""
    cfg = NeuralODEConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_ode_blocks == 2
    assert cfg.hidden == 32
    assert cfg.solver == "rk4"
    assert cfg.n_steps == 4
    assert cfg.t_start == 0.0
    assert cfg.t_end == 1.0


def test_neural_ode_config_custom():
    """NeuralODEConfig should accept custom values."""
    cfg = NeuralODEConfig(d_model=64, vocab_size=128, solver="euler", n_steps=8)
    assert cfg.d_model == 64
    assert cfg.vocab_size == 128
    assert cfg.solver == "euler"
    assert cfg.n_steps == 8


# ===========================================================================
# Additional integration tests
# ===========================================================================

def test_euler_block_gradient_flow():
    """Gradients must flow through NeuralODEBlock with Euler solver."""
    block = NeuralODEBlock(D_MODEL, solver="euler", n_steps=2)
    x = make_z0().requires_grad_(True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_lm_euler_solver():
    """NeuralODELanguageModel works with Euler solver."""
    model = NeuralODELanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        n_ode_blocks=1, solver="euler", n_steps=N_STEPS
    )
    ids = make_input_ids()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE)


def test_rk4_nfe_count():
    """RK4Solver with n_steps uses 4*n_steps function evaluations."""
    func = ODEFunc(D_MODEL)
    solver = RK4Solver()
    z0 = make_z0()
    n = 3
    solver.solve(func, z0, (0.0, 1.0), n_steps=n)
    assert func.nfe == 4 * n, f"Expected {4*n} NFE for RK4 with n_steps={n}, got {func.nfe}"


def test_euler_nfe_count():
    """EulerSolver with n_steps uses exactly n_steps function evaluations."""
    func = ODEFunc(D_MODEL)
    solver = EulerSolver()
    z0 = make_z0()
    n = 5
    solver.solve(func, z0, (0.0, 1.0), n_steps=n)
    assert func.nfe == n, f"Expected {n} NFE for Euler with n_steps={n}, got {func.nfe}"
