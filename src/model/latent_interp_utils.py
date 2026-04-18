"""
src/model/latent_interpolation.py

Spherical and linear interpolation utilities for transformer hidden states.
Useful for latent space navigation, representation mixing, and smooth decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


def slerp(v0: Tensor, v1: Tensor, t: float | Tensor) -> Tensor:
    """Spherical linear interpolation between two vectors.

    Handles parallel vectors (dot product >= 1 - eps) by falling back to lerp.

    Args:
        v0: Source tensor of shape (..., D).
        v1: Target tensor of shape (..., D).
        t:  Interpolation factor(s). Scalar or (B,) tensor.

    Returns:
        Interpolated tensor of same shape as v0.
    """
    eps = 1e-8

    # Normalize
    v0_norm = v0 / (v0.norm(dim=-1, keepdim=True).clamp(min=eps))
    v1_norm = v1 / (v1.norm(dim=-1, keepdim=True).clamp(min=eps))

    dot = (v0_norm * v1_norm).sum(dim=-1)  # (...) or scalar

    # Handle scalar t or batch t
    if isinstance(t, Tensor) and t.dim() > 0:
        # batch t: (B,) -> (B, 1) for broadcasting
        t_broadcast = t.unsqueeze(-1)
    else:
        t_broadcast = t

    # Where vectors are nearly parallel, fall back to lerp
    parallel = dot >= (1.0 - eps)

    # Compute slerp
    theta = torch.acos(dot.clamp(-1.0 + eps, 1.0 - eps))  # (...)

    if isinstance(theta, Tensor) and theta.dim() > 0:
        sin_theta = torch.sin(theta).unsqueeze(-1)  # (..., 1)
        theta_unsq = theta.unsqueeze(-1)  # (..., 1)
    else:
        sin_theta = torch.sin(theta)
        theta_unsq = theta

    sin_1mt = torch.sin((1.0 - t_broadcast) * theta_unsq)
    sin_t = torch.sin(t_broadcast * theta_unsq)

    slerp_result = (sin_1mt * v0_norm + sin_t * v1_norm) / sin_theta.clamp(min=eps)

    # Linear fallback for parallel case
    lerp_result = (1.0 - t_broadcast) * v0_norm + t_broadcast * v1_norm

    if isinstance(parallel, Tensor) and parallel.dim() > 0:
        parallel_expanded = parallel.unsqueeze(-1)
        result = torch.where(parallel_expanded, lerp_result, slerp_result)
    else:
        result = lerp_result if bool(parallel) else slerp_result

    # Rescale to mean magnitude of inputs
    mag0 = v0.norm(dim=-1, keepdim=True)
    mag1 = v1.norm(dim=-1, keepdim=True)
    mag = (mag0 + mag1) / 2.0

    return result * mag


def lerp_states(
    states_a: list[Tensor],
    states_b: list[Tensor],
    t: float,
) -> list[Tensor]:
    """Linear interpolation over a list of hidden state tensors.

    Args:
        states_a: List of tensors, one per layer.
        states_b: List of tensors matching states_a in length and shape.
        t: Interpolation factor in [0, 1].

    Returns:
        List of interpolated tensors of the same shapes.
    """
    assert len(states_a) == len(states_b), "states_a and states_b must have same length"
    return [(1.0 - t) * a + t * b for a, b in zip(states_a, states_b)]


def slerp_states(
    states_a: list[Tensor],
    states_b: list[Tensor],
    t: float,
) -> list[Tensor]:
    """Spherical interpolation over a list of hidden state tensors.

    Args:
        states_a: List of tensors, one per layer.
        states_b: List of tensors matching states_a.
        t: Interpolation factor in [0, 1].

    Returns:
        List of slerp-interpolated tensors.
    """
    assert len(states_a) == len(states_b), "states_a and states_b must have same length"
    return [slerp(a, b, t) for a, b in zip(states_a, states_b)]


class LatentMixer(nn.Module):
    """Learnable convex combination of two hidden state tensors.

    Uses a single scalar parameter alpha: output = sigmoid(alpha) * h_a + (1-sigmoid(alpha)) * h_b.

    Args:
        None. The module has a single learnable parameter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        """Blend h_a and h_b with a learned weight.

        Args:
            h_a: Tensor of any shape.
            h_b: Tensor of same shape as h_a.

        Returns:
            Blended tensor of same shape.
        """
        w = torch.sigmoid(self.alpha)
        return w * h_a + (1.0 - w) * h_b


@dataclass
class InterpolationPath:
    """Configuration and generator for an interpolation path.

    Attributes:
        steps: Number of points along the path (including endpoints).
        mode:  Interpolation mode: "slerp" or "lerp".
    """

    steps: int
    mode: str = "slerp"

    def generate_path(self, v0: Tensor, v1: Tensor) -> list[Tensor]:
        """Generate a list of interpolated tensors from v0 to v1.

        Args:
            v0: Source tensor of shape (..., D).
            v1: Target tensor of shape (..., D).

        Returns:
            List of `steps` tensors evenly spaced from v0 to v1.
        """
        ts = torch.linspace(0.0, 1.0, self.steps, dtype=v0.dtype, device=v0.device)
        path = []
        for t_val in ts:
            t = float(t_val.item())
            if self.mode == "slerp":
                point = slerp(v0, v1, t)
            else:
                point = (1.0 - t) * v0 + t * v1
            path.append(point)
        return path
