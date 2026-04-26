"""Flow Matching for continuous token embedding generation.

Implements Conditional Flow Matching (CFM) from Lipman et al. 2022 and
Albergo & Viale 2023. Instead of learning to denoise (diffusion), we learn
a velocity field v_theta(x_t, t) that interpolates between noise x_0 ~ N(0,I)
and data x_1 via a straight-line (linear) path:

    x_t = (1 - (1 - sigma_min)*t) * x_0 + t * x_1
    v* = x_1 - x_0

At inference, integrate dx/dt = v_theta(x_t, t) from t=0 to t=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FlowMatchingConfig:
    n_timesteps: int = 100  # ODE solver steps at inference
    sigma_min: float = 1e-4  # minimum noise floor
    solver: str = "euler"  # "euler" | "midpoint" | "rk4"
    embed_dim: int = 64  # embedding dimension to flow over


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


class FlowMatchingSchedule:
    """Linear interpolation schedule for conditional flow matching.

    x_t = (1 - (1 - sigma_min)*t)*x_0 + t*x_1    where t in [0, 1]
    target_velocity = x_1 - x_0

    Also supports cosine-time sampling for training.
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t, target_velocity).

        x_t = (1 - (1-sigma_min)*t)*x_0 + t*x_1
        target_velocity = x_1 - x_0
        """
        while t.dim() < x_0.dim():
            t = t.unsqueeze(-1)

        alpha = 1.0 - (1.0 - self.sigma_min) * t
        x_t = alpha * x_0 + t * x_1
        target_velocity = x_1 - x_0
        return x_t, target_velocity

    def sample_timesteps(
        self,
        n: int,
        device: torch.device,
        mode: str = "uniform",
    ) -> torch.Tensor:
        """Sample n timesteps in [0, 1].

        mode='uniform': t ~ Uniform(0, 1)
        mode='cosine':  t = (1 - cos(pi*u/2))^2  where u ~ Uniform(0, 1)
        """
        u = torch.rand(n, device=device)
        if mode == "uniform":
            return u
        elif mode == "cosine":
            return (1.0 - torch.cos(math.pi * u / 2.0)) ** 2
        else:
            raise ValueError(f"Unknown timestep sampling mode: {mode!r}")


# ---------------------------------------------------------------------------
# Velocity Network
# ---------------------------------------------------------------------------


class VelocityNetwork(nn.Module):
    """MLP that predicts the velocity field v_theta(x_t, t).

    Time embedding: sinusoidal positional encoding with k=16 frequencies
        te(t) = [sin(2^0*pi*t), cos(2^0*pi*t), ..., sin(2^15*pi*t), cos(2^15*pi*t)]
        time_embed_dim = 32

    MLP: Linear(D+32, hidden) -> SiLU -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, D)

    Handles both (B, D) and (B, T, D) inputs.
    """

    TIME_EMBED_DIM: int = 32  # 2 * k, k=16

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        k = self.TIME_EMBED_DIM // 2  # 16
        freqs = (2.0 ** torch.arange(k)) * math.pi
        self.register_buffer("freqs", freqs)

        in_dim = embed_dim + self.TIME_EMBED_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _time_embedding(self, t: torch.Tensor, leading_shape: tuple) -> torch.Tensor:
        """Build sinusoidal time embedding expanded to leading_shape + (TIME_EMBED_DIM,)."""
        if t.dim() == 0:
            t = t.unsqueeze(0)

        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        te = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        for _ in range(len(leading_shape) - 1):
            te = te.unsqueeze(1)
        te = te.expand(*leading_shape, self.TIME_EMBED_DIM)
        return te

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, D) or (B, T, D)
        t:   (B,) or scalar
        Returns velocity of the same shape as x_t.
        """
        leading_shape = x_t.shape[:-1]
        te = self._time_embedding(t, leading_shape)
        x_in = torch.cat([x_t, te], dim=-1)
        return self.net(x_in)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def flow_matching_loss(
    velocity_network: VelocityNetwork,
    x_1: torch.Tensor,
    schedule: FlowMatchingSchedule,
) -> torch.Tensor:
    """Conditional Flow Matching (CFM) training loss.

    1. Sample noise x_0 ~ N(0, I)
    2. Sample t ~ Uniform(0, 1)
    3. Interpolate: x_t, v_target = schedule.interpolate(x_0, x_1, t)
    4. Predict:  v_pred = velocity_network(x_t, t)
    5. Loss = MSE(v_pred, v_target)

    Returns scalar loss.
    """
    B = x_1.shape[0]
    device = x_1.device

    x_0 = torch.randn_like(x_1)
    t = schedule.sample_timesteps(B, device=device, mode="uniform")

    x_t, v_target = schedule.interpolate(x_0, x_1, t)
    v_pred = velocity_network(x_t, t)

    return F.mse_loss(v_pred, v_target)


# ---------------------------------------------------------------------------
# ODE Solver
# ---------------------------------------------------------------------------


class ODESolver:
    """Integrate dx/dt = v_theta(x_t, t) from t=0 (noise) to t=1 (data).

    Solvers:
        "euler":    x_{t+h} = x_t + h * v(x_t, t)
        "midpoint": two-stage midpoint method
        "rk4":      standard 4th-order Runge-Kutta
    """

    def __init__(
        self,
        velocity_network: VelocityNetwork,
        n_steps: int = 100,
        solver: str = "euler",
    ):
        self.velocity_network = velocity_network
        self.n_steps = n_steps
        self.solver = solver

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from the flow: x_0 ~ N(0,I) -> integrate -> return x_1."""
        x = torch.randn(*shape, device=device)
        h = 1.0 / self.n_steps
        B = shape[0]

        for i in range(self.n_steps):
            t_val = i * h
            t = torch.full((B,), t_val, device=device, dtype=x.dtype)

            if self.solver == "euler":
                v = self.velocity_network(x, t)
                x = x + h * v

            elif self.solver == "midpoint":
                v1 = self.velocity_network(x, t)
                x_mid = x + (h / 2.0) * v1
                t_mid = torch.full((B,), t_val + h / 2.0, device=device, dtype=x.dtype)
                v2 = self.velocity_network(x_mid, t_mid)
                x = x + h * v2

            elif self.solver == "rk4":
                k1 = self.velocity_network(x, t)
                t_half = torch.full((B,), t_val + h / 2.0, device=device, dtype=x.dtype)
                k2 = self.velocity_network(x + (h / 2.0) * k1, t_half)
                k3 = self.velocity_network(x + (h / 2.0) * k2, t_half)
                t_next = torch.full((B,), t_val + h, device=device, dtype=x.dtype)
                k4 = self.velocity_network(x + h * k3, t_next)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(f"Unknown solver: {self.solver!r}")

        return x


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FlowMatchingTrainer:
    """Trainer for flow matching over token embeddings."""

    def __init__(
        self,
        velocity_network: VelocityNetwork,
        schedule: FlowMatchingSchedule,
        optimizer: torch.optim.Optimizer,
    ):
        self.velocity_network = velocity_network
        self.schedule = schedule
        self.optimizer = optimizer

    def train_step(self, embeddings: torch.Tensor) -> dict:
        """embeddings: (B, D). Returns {'loss': float}."""
        self.velocity_network.train()
        self.optimizer.zero_grad()

        loss = flow_matching_loss(self.velocity_network, embeddings, self.schedule)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def generate(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate n_samples embedding vectors using ODESolver. Returns (n_samples, embed_dim)."""
        self.velocity_network.eval()
        solver = ODESolver(
            self.velocity_network,
            n_steps=50,
            solver="euler",
        )
        shape = (n_samples, self.velocity_network.embed_dim)
        return solver.sample(shape, device)
