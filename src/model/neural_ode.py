"""Neural ODE for sequence modeling.

Provides a continuous-depth alternative to discrete transformer layers via
neural ordinary differential equations. Supports Euler, RK4, and midpoint
integration schemes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ODEConfig:
    """Configuration for a NeuralODEBlock."""

    n_steps: int = 4          # number of integration steps
    step_size: float = 0.25   # dt per step (t spans [0, 1])
    d_model: int = 64
    solver: str = "euler"     # "euler" | "rk4" | "midpoint"
    augment_dim: int = 0      # extra dims appended to state before integration


# ---------------------------------------------------------------------------
# ODE dynamics function
# ---------------------------------------------------------------------------

class ODEFunction(nn.Module):
    """The right-hand-side dynamics f(t, h) of the neural ODE.

    Architecture: Linear → SiLU → Linear (small MLP with time concatenated).
    """

    def __init__(self, d_model: int, augment_dim: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.augment_dim = augment_dim
        state_dim = d_model + augment_dim
        # +1 for the time scalar concatenated to each token
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )

    def forward(self, t: float, h: Tensor) -> Tensor:
        """Compute dhdt = f(t, h).

        Args:
            t: Current time scalar.
            h: Hidden state of shape (B, T, D) where D = d_model + augment_dim.

        Returns:
            dhdt with the same shape as h.
        """
        # Broadcast time scalar to (B, T, 1) and concatenate
        t_tensor = h.new_full((*h.shape[:-1], 1), t)
        h_t = torch.cat([h, t_tensor], dim=-1)  # (B, T, D+1)
        return self.net(h_t)


# ---------------------------------------------------------------------------
# ODE solvers (standalone functions)
# ---------------------------------------------------------------------------

def euler_solve(
    f: Callable[[float, Tensor], Tensor],
    h0: Tensor,
    t0: float,
    t1: float,
    n_steps: int,
) -> Tensor:
    """Euler integration of dh/dt = f(t, h) from t0 to t1.

    Args:
        f: Dynamics function f(t, h) -> dhdt.
        h0: Initial state.
        t0: Start time.
        t1: End time.
        n_steps: Number of Euler steps.

    Returns:
        h at time t1, same shape as h0.
    """
    dt = (t1 - t0) / n_steps
    h = h0
    t = t0
    for _ in range(n_steps):
        h = h + dt * f(t, h)
        t = t + dt
    return h


def rk4_solve(
    f: Callable[[float, Tensor], Tensor],
    h0: Tensor,
    t0: float,
    t1: float,
    n_steps: int,
) -> Tensor:
    """Fourth-order Runge-Kutta integration from t0 to t1.

    Args:
        f: Dynamics function f(t, h) -> dhdt.
        h0: Initial state.
        t0: Start time.
        t1: End time.
        n_steps: Number of RK4 steps.

    Returns:
        h at time t1, same shape as h0.
    """
    dt = (t1 - t0) / n_steps
    h = h0
    t = t0
    for _ in range(n_steps):
        k1 = f(t, h)
        k2 = f(t + dt / 2, h + dt / 2 * k1)
        k3 = f(t + dt / 2, h + dt / 2 * k2)
        k4 = f(t + dt, h + dt * k3)
        h = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
    return h


def midpoint_solve(
    f: Callable[[float, Tensor], Tensor],
    h0: Tensor,
    t0: float,
    t1: float,
    n_steps: int,
) -> Tensor:
    """Explicit midpoint (Runge-Kutta 2) integration from t0 to t1.

    Args:
        f: Dynamics function f(t, h) -> dhdt.
        h0: Initial state.
        t0: Start time.
        t1: End time.
        n_steps: Number of midpoint steps.

    Returns:
        h at time t1, same shape as h0.
    """
    dt = (t1 - t0) / n_steps
    h = h0
    t = t0
    for _ in range(n_steps):
        k1 = f(t, h)
        k2 = f(t + dt / 2, h + dt / 2 * k1)
        h = h + dt * k2
        t = t + dt
    return h


# ---------------------------------------------------------------------------
# NeuralODEBlock
# ---------------------------------------------------------------------------

class NeuralODEBlock(nn.Module):
    """Integrates hidden states through learned continuous dynamics.

    Wraps an ODEFunction with a chosen solver, optionally augmenting the
    state with extra zero-padded dimensions before integration.
    """

    _SOLVERS: dict[str, Callable] = {
        "euler": euler_solve,
        "rk4": rk4_solve,
        "midpoint": midpoint_solve,
    }

    def __init__(self, config: ODEConfig) -> None:
        super().__init__()
        self.config = config
        self.ode_func = ODEFunction(config.d_model, config.augment_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Integrate x from t=0 to t=1.

        Args:
            x: Input of shape (B, T, d_model).

        Returns:
            Integrated hidden states, same shape as x.
        """
        solve = self._SOLVERS[self.config.solver]

        if self.config.augment_dim > 0:
            aug = x.new_zeros(*x.shape[:-1], self.config.augment_dim)
            h = torch.cat([x, aug], dim=-1)
        else:
            h = x

        h_out = solve(self.ode_func, h, t0=0.0, t1=1.0, n_steps=self.config.n_steps)

        # Strip augmented dimensions
        if self.config.augment_dim > 0:
            h_out = h_out[..., : self.config.d_model]

        return h_out


# ---------------------------------------------------------------------------
# ODETransformerBlock
# ---------------------------------------------------------------------------

class ODETransformerBlock(nn.Module):
    """Transformer block that replaces depth-wise layers with a Neural ODE.

    Applies LayerNorm → NeuralODE → residual add → linear projection.
    """

    def __init__(self, d_model: int, config: ODEConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ode = NeuralODEBlock(config)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ODE block with pre-norm and residual connection.

        Args:
            x: Input of shape (B, T, d_model).

        Returns:
            Output of shape (B, T, d_model).
        """
        normed = self.norm(x)
        ode_out = self.ode(normed)
        return x + self.ff(ode_out)


# ---------------------------------------------------------------------------
# Trajectory utility
# ---------------------------------------------------------------------------

def compute_trajectory(
    ode_block: NeuralODEBlock,
    x: Tensor,
    n_snapshots: int = 4,
) -> list[Tensor]:
    """Capture hidden states at evenly spaced time points during integration.

    Args:
        ode_block: A NeuralODEBlock to integrate through.
        x: Input of shape (B, T, d_model).
        n_snapshots: Number of time snapshots to capture (including t=1).

    Returns:
        List of n_snapshots tensors, each of shape (B, T, d_model).
    """
    config = ode_block.config
    solve = NeuralODEBlock._SOLVERS[config.solver]

    if config.augment_dim > 0:
        aug = x.new_zeros(*x.shape[:-1], config.augment_dim)
        h = torch.cat([x, aug], dim=-1)
    else:
        h = x

    snapshots: list[Tensor] = []
    t_points = [i / n_snapshots for i in range(1, n_snapshots + 1)]
    t_prev = 0.0

    for t_next in t_points:
        # Steps proportional to fraction of total interval
        steps = max(1, round(config.n_steps * (t_next - t_prev)))
        h = solve(ode_block.ode_func, h, t0=t_prev, t1=t_next, n_steps=steps)
        t_prev = t_next

        snap = h[..., : config.d_model] if config.augment_dim > 0 else h
        snapshots.append(snap)

    return snapshots
