"""Tests for the SOAP optimizer (Vyas et al. 2024)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.soap import SOAP

# ---------------------------------------------------------------------------
# 1. Loss decreases after a few steps on a simple quadratic
# ---------------------------------------------------------------------------


def test_soap_step_reduces_loss():
    """SOAP should reduce a simple quadratic loss within 5 steps."""
    torch.manual_seed(0)
    W = nn.Parameter(torch.randn(8, 8))
    target = torch.zeros(8, 8)
    optimizer = SOAP([W], lr=1e-2, weight_decay=0.0, precond_freq=2)

    initial_loss = (W - target).pow(2).mean().item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = (W - target).pow(2).mean()
        loss.backward()
        optimizer.step()

    final_loss = (W - target).pow(2).mean().item()
    assert final_loss < initial_loss, (
        f"Expected loss to decrease, got {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. 1D parameters use plain Adam — no Kronecker factors in state
# ---------------------------------------------------------------------------


def test_soap_1d_params_plain_adam():
    """1D parameters must not have L, R, Q_L, Q_R in their optimizer state."""
    param = nn.Parameter(torch.randn(16))
    optimizer = SOAP([param], lr=1e-3, weight_decay=0.0)
    param.grad = torch.randn_like(param)
    optimizer.step()

    state = optimizer.state[param]
    for key in ("L", "R", "Q_L", "Q_R"):
        assert key not in state, f"Unexpected Kronecker key '{key}' in 1D param state"

    # Sanity: Adam keys are present
    assert "exp_avg" in state
    assert "exp_avg_sq" in state


# ---------------------------------------------------------------------------
# 3. 2D parameters have Kronecker factors after first step
# ---------------------------------------------------------------------------


def test_soap_2d_params_have_kronecker_factors():
    """After the first step on a 2D param, state must contain L, R, Q_L, Q_R."""
    param = nn.Parameter(torch.randn(8, 16))
    optimizer = SOAP([param], lr=1e-3, weight_decay=0.0)
    param.grad = torch.randn_like(param)
    optimizer.step()

    state = optimizer.state[param]
    for key in ("step", "exp_avg", "exp_avg_sq", "L", "R", "Q_L", "Q_R"):
        assert key in state, f"Missing expected state key '{key}'"

    assert state["L"].shape == (8, 8)
    assert state["R"].shape == (16, 16)
    assert state["Q_L"].shape == (8, 8)
    assert state["Q_R"].shape == (16, 16)


# ---------------------------------------------------------------------------
# 4. Preconditioner updates exactly at precond_freq, not before
# ---------------------------------------------------------------------------


def test_soap_precond_update_frequency():
    """Q_L should stay at the identity until step == precond_freq."""
    torch.manual_seed(42)
    freq = 4
    param = nn.Parameter(torch.randn(6, 6))
    optimizer = SOAP([param], lr=1e-3, weight_decay=0.0, precond_freq=freq)

    # Step 1 — initialises Q_L to identity, no eigendecomposition yet
    param.grad = torch.randn_like(param)
    optimizer.step()
    Q_L_after_step1 = optimizer.state[param]["Q_L"].clone()

    # Steps 2 … freq-1 — Q_L must remain unchanged
    for _ in range(freq - 2):
        param.grad = torch.randn_like(param)
        optimizer.step()

    Q_L_before_trigger = optimizer.state[param]["Q_L"].clone()
    assert torch.allclose(Q_L_after_step1, Q_L_before_trigger), (
        "Q_L changed before precond_freq was reached"
    )

    # Step freq — eigendecomposition should fire now
    param.grad = torch.randn_like(param)
    optimizer.step()
    Q_L_after_trigger = optimizer.state[param]["Q_L"].clone()

    assert not torch.allclose(Q_L_after_step1, Q_L_after_trigger, atol=1e-5), (
        "Q_L did not change at precond_freq step"
    )


# ---------------------------------------------------------------------------
# 5. Weight decay reduces L2 norm on zero gradient
# ---------------------------------------------------------------------------


def test_soap_weight_decay():
    """With weight_decay > 0 and zero gradient, param norm should shrink."""
    torch.manual_seed(7)
    param = nn.Parameter(torch.randn(8, 8))
    optimizer = SOAP([param], lr=1e-2, weight_decay=0.1, precond_freq=5)

    initial_norm = param.norm().item()
    for _ in range(30):
        param.grad = torch.zeros_like(param)  # zero gradient
        optimizer.step()

    final_norm = param.norm().item()
    assert final_norm < initial_norm, (
        f"Expected norm to shrink with weight decay; got {initial_norm:.4f} -> {final_norm:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Different matrix shapes are handled correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 8), (16, 32), (64, 64)])
def test_soap_different_matrix_shapes(shape):
    """SOAP should run without error for various 2D parameter shapes."""
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(*shape))
    optimizer = SOAP([param], lr=1e-3, weight_decay=0.0, precond_freq=2)

    for _ in range(3):
        param.grad = torch.randn_like(param)
        optimizer.step()

    state = optimizer.state[param]
    m, n = shape
    assert state["L"].shape == (m, m)
    assert state["R"].shape == (n, n)
    assert state["Q_L"].shape == (m, m)
    assert state["Q_R"].shape == (n, n)


# ---------------------------------------------------------------------------
# 7. Eigenvectors are orthonormal: Q_L^T @ Q_L ≈ I
# ---------------------------------------------------------------------------


def test_soap_eigenspace_projection_roundtrip():
    """After a preconditioner update, Q_L and Q_R must be orthonormal matrices."""
    torch.manual_seed(3)
    param = nn.Parameter(torch.randn(8, 12))
    # Set precond_freq=1 so eigendecomposition fires on every step after step 0
    optimizer = SOAP([param], lr=1e-3, weight_decay=0.0, precond_freq=1)

    # Step 1 initialises (no eig yet); step 2 triggers first eig decomposition
    for _ in range(2):
        param.grad = torch.randn_like(param)
        optimizer.step()

    state = optimizer.state[param]
    Q_L = state["Q_L"]
    Q_R = state["Q_R"]

    eye_m = torch.eye(Q_L.shape[0], device=Q_L.device, dtype=Q_L.dtype)
    eye_n = torch.eye(Q_R.shape[0], device=Q_R.device, dtype=Q_R.dtype)

    assert torch.allclose(Q_L.T @ Q_L, eye_m, atol=1e-5), "Q_L is not orthonormal"
    assert torch.allclose(Q_R.T @ Q_R, eye_n, atol=1e-5), "Q_R is not orthonormal"


# ---------------------------------------------------------------------------
# 8. State is initialised correctly after one step
# ---------------------------------------------------------------------------


def test_soap_state_initialized_correctly():
    """After exactly one step, verify all expected state keys exist with correct dtypes."""
    torch.manual_seed(1)
    param = nn.Parameter(torch.randn(10, 20))
    optimizer = SOAP([param], lr=1e-3, weight_decay=0.0)
    param.grad = torch.randn_like(param)
    optimizer.step()

    state = optimizer.state[param]
    expected_keys = {"step", "exp_avg", "exp_avg_sq", "L", "R", "Q_L", "Q_R"}
    assert expected_keys.issubset(state.keys()), (
        f"Missing keys: {expected_keys - set(state.keys())}"
    )

    assert state["step"] == 1
    assert state["exp_avg"].shape == (10, 20)
    assert state["exp_avg_sq"].shape == (10, 20)
    assert state["L"].shape == (10, 10)
    assert state["R"].shape == (20, 20)
    assert state["Q_L"].shape == (10, 10)
    assert state["Q_R"].shape == (20, 20)

    # Q_L and Q_R must still be identity on step 1 (no eigen update yet)
    eye10 = torch.eye(10, dtype=param.dtype)
    eye20 = torch.eye(20, dtype=param.dtype)
    assert torch.allclose(state["Q_L"], eye10, atol=1e-6), "Q_L should be identity after step 1"
    assert torch.allclose(state["Q_R"], eye20, atol=1e-6), "Q_R should be identity after step 1"
