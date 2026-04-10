"""
Hyperbolic (Poincaré ball) embeddings for hierarchical representation learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class HyperbolicConfig:
    dim: int = 64
    curvature: float = 1.0
    clip_r: float = 2.3
    eps: float = 1e-5


def poincare_distance(u: Tensor, v: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Poincaré ball distance using the arccosh formula.

    d_c(u, v) = arccosh(1 + 2c * ||u - v||^2 / ((1 - c*||u||^2)(1 - c*||v||^2))) / sqrt(c)

    Args:
        u: (..., D) tensor
        v: (..., D) tensor
        c: curvature (positive)
        eps: numerical stability floor

    Returns:
        (...,) distance tensor
    """
    sqrt_c = math.sqrt(c)
    diff_sq = (u - v).pow(2).sum(dim=-1)  # (...,)
    u_sq = u.pow(2).sum(dim=-1).clamp(max=1.0 - eps)  # inside ball
    v_sq = v.pow(2).sum(dim=-1).clamp(max=1.0 - eps)

    numerator = 2.0 * c * diff_sq
    denominator = (1.0 - c * u_sq) * (1.0 - c * v_sq)
    denominator = denominator.clamp(min=eps)

    arg = 1.0 + numerator / denominator
    # arccosh domain: arg >= 1; clamp to 1.0 so same-point distance is 0
    arg = arg.clamp(min=1.0)
    dist = torch.acosh(arg) / sqrt_c
    return dist


def expmap0(v: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Exponential map at the origin: maps tangent space -> Poincaré ball.

    expmap0(v) = tanh(sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||)

    Args:
        v: (..., D) tensor in tangent space
        c: curvature
        eps: numerical floor for norm

    Returns:
        (..., D) tensor on Poincaré ball
    """
    sqrt_c = math.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    tanh_arg = (sqrt_c * norm / 2.0).clamp(max=15.0)  # prevent overflow in tanh
    return torch.tanh(tanh_arg) * v / (sqrt_c * norm)


def logmap0(y: Tensor, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    """Log map at the origin: inverse of expmap0.

    logmap0(y) = arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)

    Args:
        y: (..., D) tensor on Poincaré ball
        c: curvature
        eps: numerical floor for norm

    Returns:
        (..., D) tensor in tangent space
    """
    sqrt_c = math.sqrt(c)
    norm = y.norm(dim=-1, keepdim=True).clamp(min=eps)
    # arctanh requires input in (-1, 1)
    scaled_norm = (sqrt_c * norm).clamp(min=-1.0 + eps, max=1.0 - eps)
    # Factor 2 matches the /2 in expmap0
    return 2.0 * torch.arctanh(scaled_norm) * y / (sqrt_c * norm)


def project_to_ball(x: Tensor, c: float = 1.0, clip_r: float = 2.3) -> Tensor:
    """Project points to the interior of the Poincaré ball.

    Clips the norm to be strictly less than clip_r / sqrt(c).

    Args:
        x: (..., D) tensor
        c: curvature
        clip_r: clipping radius in the normalised ball

    Returns:
        (..., D) tensor with norm < clip_r / sqrt(c)
    """
    max_norm = clip_r / math.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # Scale down only those that exceed max_norm
    scale = (max_norm / norm).clamp(max=1.0)
    return x * scale


class PoincareEmbedding(nn.Module):
    """Embedding layer that lives in the Poincaré ball."""

    def __init__(self, n_embeddings: int, dim: int, config: HyperbolicConfig) -> None:
        super().__init__()
        self.config = config
        # Initialise small so that expmap0 keeps them inside the ball
        self.weight = nn.Parameter(torch.empty(n_embeddings, dim))
        nn.init.normal_(self.weight, mean=0.0, std=1e-2)

    def forward(self, idx: Tensor) -> Tensor:
        """Look up embeddings, map to ball, and project.

        Args:
            idx: (B,) or (B, T) long tensor of indices

        Returns:
            (..., dim) tensor on Poincaré ball
        """
        cfg = self.config
        embeds = self.weight[idx]  # (..., dim) — raw tangent-space vectors
        embeds = expmap0(embeds, c=cfg.curvature, eps=cfg.eps)
        embeds = project_to_ball(embeds, c=cfg.curvature, clip_r=cfg.clip_r)
        return embeds


class HyperbolicLinear(nn.Module):
    """Linear layer operating in hyperbolic space via tangent-space mappings.

    Pipeline: logmap0(x) -> linear -> expmap0 -> project_to_ball
    """

    def __init__(self, in_dim: int, out_dim: int, config: HyperbolicConfig) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        # Kaiming uniform for the linear part
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """Map x (on ball) through tangent space, apply linear, return to ball.

        Args:
            x: (..., in_dim) tensor on Poincaré ball

        Returns:
            (..., out_dim) tensor on Poincaré ball
        """
        cfg = self.config
        tangent = logmap0(x, c=cfg.curvature, eps=cfg.eps)
        out = self.linear(tangent)
        out = expmap0(out, c=cfg.curvature, eps=cfg.eps)
        out = project_to_ball(out, c=cfg.curvature, clip_r=cfg.clip_r)
        return out
