"""Flow matching (conditional flow matching / continuous normalizing flows).

Implements conditional flow matching for generative modeling over embeddings.
The model learns a vector field that transports samples from a source distribution
x0 to a target distribution x1 via a simple linear interpolation path.

Objective:
    min E_{t, x0, x1} || v_theta(x_t, t) - u_t(x0, x1) ||^2

where x_t = (1-t)*x0 + t*x1 and u_t = x1 - x0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlowConfig:
    """Configuration for flow matching."""

    sigma: float = 0.1  # noise level (unused in basic CFM but available for extensions)
    n_timesteps: int = 100  # number of ODE integration steps by default
    ode_method: str = "euler"  # "euler" | "midpoint"
    d_model: int = 64  # embedding dimensionality


def sample_xt(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Linear interpolation between x0 and x1 at time t.

    Args:
        x0: Source samples shape (B, D).
        x1: Target samples shape (B, D).
        t:  Time values in [0, 1] shape (B,).

    Returns:
        Interpolated samples x_t = (1-t)*x0 + t*x1 shape (B, D).
    """
    t = t.view(-1, 1)  # (B, 1) for broadcasting
    return (1.0 - t) * x0 + t * x1


def compute_ut(x0: Tensor, x1: Tensor) -> Tensor:
    """Compute the target (conditional) vector field u_t = x1 - x0.

    The field is constant along the linear interpolation path.

    Args:
        x0: Source samples shape (B, D).
        x1: Target samples shape (B, D).

    Returns:
        Target vector field shape (B, D).
    """
    return x1 - x0


class VectorFieldNet(nn.Module):
    """MLP that predicts the vector field v_theta(x, t).

    Time t is embedded as [sin(2*pi*t), cos(2*pi*t)] and concatenated with x,
    so the MLP input dimension is D+2.

    Args:
        d_model:    Dimensionality of the embedding space D.
        hidden_dim: Width of hidden layers.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        input_dim = d_model + 2  # x (D) + sinusoidal t embedding (2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict vector field at (x, t).

        Args:
            x: Embeddings shape (B, D).
            t: Timesteps in [0, 1] shape (B,).

        Returns:
            Predicted vector field shape (B, D).
        """
        t = t.view(-1, 1)  # (B, 1)
        t_embed = torch.cat(
            [torch.sin(2.0 * math.pi * t), torch.cos(2.0 * math.pi * t)], dim=-1
        )  # (B, 2)
        h = torch.cat([x, t_embed], dim=-1)  # (B, D+2)
        return self.net(h)


def flow_matching_loss(
    vf_net: VectorFieldNet,
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
) -> Tensor:
    """Compute the conditional flow matching loss.

    Loss = E [ || v_theta(x_t, t) - u_t ||^2 ]

    Args:
        vf_net: The vector field network.
        x0:     Source samples shape (B, D).
        x1:     Target samples shape (B, D).
        t:      Timesteps in [0, 1] shape (B,).

    Returns:
        Scalar MSE loss.
    """
    x_t = sample_xt(x0, x1, t)  # (B, D)
    u_t = compute_ut(x0, x1)  # (B, D)
    v_pred = vf_net(x_t, t)  # (B, D)
    return F.mse_loss(v_pred, u_t)


def euler_solve(
    vf_net: VectorFieldNet,
    x0: Tensor,
    n_steps: int,
    dt: float | None = None,
) -> Tensor:
    """Integrate the ODE from t=0 to t=1 using the Euler method.

    x_{t+dt} = x_t + dt * v_theta(x_t, t)

    Args:
        vf_net:  Trained vector field network.
        x0:      Starting samples shape (B, D).
        n_steps: Number of Euler steps.
        dt:      Step size (defaults to 1/n_steps).

    Returns:
        Final integrated samples shape (B, D).
    """
    if dt is None:
        dt = 1.0 / n_steps

    x = x0.clone()
    B = x.shape[0]
    device = x.device

    with torch.no_grad():
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, dtype=x.dtype, device=device)
            v = vf_net(x, t)
            x = x + dt * v

    return x


class FlowMatchingTrainer:
    """Trains a VectorFieldNet using conditional flow matching.

    Args:
        vf_net:    The vector field network.
        config:    Flow matching configuration.
        optimizer: PyTorch optimizer bound to vf_net parameters.
    """

    def __init__(
        self,
        vf_net: VectorFieldNet,
        config: FlowConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.vf_net = vf_net
        self.config = config
        self.optimizer = optimizer

    def train_step(self, x0: Tensor, x1: Tensor) -> dict:
        """Perform one gradient update step.

        Samples t ~ Uniform(0, 1) per sample, computes flow matching loss,
        runs backward, and steps the optimizer.

        Args:
            x0: Source samples shape (B, D).
            x1: Target samples shape (B, D).

        Returns:
            Dictionary with key "loss" mapping to a Python float.
        """
        self.vf_net.train()
        self.optimizer.zero_grad()

        B = x0.shape[0]
        t = torch.rand(B, dtype=x0.dtype, device=x0.device)
        loss = flow_matching_loss(self.vf_net, x0, x1, t)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def generate(self, x0: Tensor, n_steps: int = 50) -> Tensor:
        """Generate samples by integrating the ODE from x0.

        Args:
            x0:      Starting noise samples shape (B, D).
            n_steps: Number of Euler integration steps.

        Returns:
            Generated samples shape (B, D).
        """
        self.vf_net.eval()
        return euler_solve(self.vf_net, x0, n_steps=n_steps)
