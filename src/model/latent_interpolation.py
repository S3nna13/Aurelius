"""Latent space interpolation methods for LLM representations.

Interpolates between hidden state vectors using various geometric methods:
- Linear (LERP): simple weighted average
- Spherical (SLERP): interpolation along the unit sphere
- Manifold: interpolation via geodesic approximation on learned manifold

Useful for concept blending, style mixing, and representation analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# LinearInterpolator
# ---------------------------------------------------------------------------


class LinearInterpolator:
    """Stateless linear (LERP) interpolator between latent vectors."""

    def __init__(self) -> None:
        pass

    def interpolate(self, z0: Tensor, z1: Tensor, t: float) -> Tensor:
        """Linearly interpolate between z0 and z1.

        Args:
            z0: Source tensor, shape ``(d,)`` or ``(B, d)``.
            z1: Target tensor, same shape as z0.
            t:  Interpolation factor in [0, 1].

        Returns:
            Interpolated tensor: ``(1-t) * z0 + t * z1``.
        """
        return (1.0 - t) * z0 + t * z1

    def path(self, z0: Tensor, z1: Tensor, n_steps: int) -> Tensor:
        """Return a sequence of interpolated points along the linear path.

        Args:
            z0: Source tensor, shape ``(d,)`` or ``(B, d)``.
            z1: Target tensor, same shape as z0.
            n_steps: Number of steps (including endpoints).

        Returns:
            For ``(d,)`` inputs: shape ``(n_steps, d)``.
            For ``(B, d)`` inputs: shape ``(n_steps, B, d)``.
        """
        ts = torch.linspace(0.0, 1.0, n_steps, dtype=z0.dtype, device=z0.device)
        # Build path by stacking interpolated points
        points = [self.interpolate(z0, z1, float(t)) for t in ts]
        return torch.stack(points, dim=0)


# ---------------------------------------------------------------------------
# SphericalInterpolator
# ---------------------------------------------------------------------------


class SphericalInterpolator:
    """Stateless spherical linear (SLERP) interpolator."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def interpolate(self, z0: Tensor, z1: Tensor, t: float) -> Tensor:
        """Spherically interpolate between z0 and z1.

        Vectors are normalized internally; the result is returned at the
        mean of the two input norms.

        Args:
            z0: Source vector, shape ``(d,)``.
            z1: Target vector, shape ``(d,)``.
            t:  Interpolation factor in [0, 1].

        Returns:
            Interpolated vector, shape ``(d,)``.
        """
        norm0 = z0.norm()
        norm1 = z1.norm()
        scale = (norm0 + norm1) / 2.0

        z0_n = z0 / (norm0 + self.eps)
        z1_n = z1 / (norm1 + self.eps)

        dot = torch.clamp(
            torch.dot(z0_n, z1_n),
            -1.0 + self.eps,
            1.0 - self.eps,
        )
        theta = torch.acos(dot)

        if theta.abs() < self.eps:
            # Parallel vectors — fall back to linear interpolation
            return ((1.0 - t) * z0_n + t * z1_n) * scale

        result = (torch.sin((1.0 - t) * theta) * z0_n + torch.sin(t * theta) * z1_n) / torch.sin(
            theta
        )
        return result * scale

    def path(self, z0: Tensor, z1: Tensor, n_steps: int) -> Tensor:
        """Return a SLERP path between z0 and z1.

        Args:
            z0: Source vector, shape ``(d,)``.
            z1: Target vector, shape ``(d,)``.
            n_steps: Number of steps (including endpoints).

        Returns:
            Tensor of shape ``(n_steps, d)``.
        """
        ts = torch.linspace(0.0, 1.0, n_steps, dtype=z0.dtype, device=z0.device)
        points = [self.interpolate(z0, z1, float(t)) for t in ts]
        return torch.stack(points, dim=0)


# ---------------------------------------------------------------------------
# ManifoldInterpolator
# ---------------------------------------------------------------------------


class ManifoldInterpolator(nn.Module):
    """Geodesic-approximating interpolator via a learned encoder-decoder.

    Encodes both endpoints into a lower-dimensional latent space, performs
    SLERP there, then decodes back to the original space.

    Args:
        d_model:  Dimensionality of the input/output space.
        d_latent: Dimensionality of the internal latent space.
    """

    def __init__(self, d_model: int, d_latent: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent

        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_model),
        )
        self.slerp = SphericalInterpolator()

    def encode(self, z: Tensor) -> Tensor:
        """Encode from model space to latent space.

        Args:
            z: Shape ``(B, d_model)``.

        Returns:
            Shape ``(B, d_latent)``.
        """
        return self.encoder(z)

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent space to model space.

        Args:
            z: Shape ``(B, d_latent)``.

        Returns:
            Shape ``(B, d_model)``.
        """
        return self.decoder(z)

    def interpolate(self, z0: Tensor, z1: Tensor, t: float) -> Tensor:
        """Interpolate via the learned manifold.

        Args:
            z0: Source tensor, shape ``(d_model,)`` or ``(B, d_model)``.
            z1: Target tensor, same shape as z0.
            t:  Interpolation factor in [0, 1].

        Returns:
            Interpolated tensor, same shape as input.
        """
        squeeze = z0.dim() == 1
        if squeeze:
            z0 = z0.unsqueeze(0)
            z1 = z1.unsqueeze(0)

        h0 = self.encode(z0)  # (B, d_latent)
        h1 = self.encode(z1)  # (B, d_latent)

        # SLERP per batch element
        h_mid = torch.stack(
            [self.slerp.interpolate(h0[i], h1[i], t) for i in range(h0.shape[0])],
            dim=0,
        )

        out = self.decode(h_mid)  # (B, d_model)
        if squeeze:
            out = out.squeeze(0)
        return out

    def path(self, z0: Tensor, z1: Tensor, n_steps: int) -> Tensor:
        """Return a manifold path between z0 and z1.

        Args:
            z0: Source tensor, shape ``(d_model,)`` or ``(B, d_model)``.
            z1: Target tensor, same shape as z0.
            n_steps: Number of steps (including endpoints).

        Returns:
            Shape ``(n_steps, d_model)``.
        """
        ts = torch.linspace(0.0, 1.0, n_steps, dtype=z0.dtype, device=z0.device)
        points = [self.interpolate(z0, z1, float(t)) for t in ts]
        return torch.stack(points, dim=0)


# ---------------------------------------------------------------------------
# InterpolationAnalyzer
# ---------------------------------------------------------------------------


class InterpolationAnalyzer:
    """Analyzes geometric properties of interpolation paths."""

    def __init__(self) -> None:
        pass

    def path_length(self, path: Tensor) -> float:
        """Compute total arc length of a path.

        Args:
            path: Tensor of shape ``(n_steps, d)``.

        Returns:
            Sum of L2 distances between consecutive points.
        """
        diffs = path[1:] - path[:-1]  # (n_steps-1, d)
        step_lengths = diffs.norm(dim=-1)  # (n_steps-1,)
        return float(step_lengths.sum().item())

    def midpoint_deviation(self, path: Tensor, z0: Tensor, z1: Tensor) -> float:
        """Distance from path midpoint to the linear midpoint.

        Args:
            path: Tensor of shape ``(n_steps, d)``.
            z0:   Start vector, shape ``(d,)``.
            z1:   End vector, shape ``(d,)``.

        Returns:
            L2 distance between ``path[n//2]`` and ``(z0 + z1) / 2``.
        """
        n = path.shape[0]
        midpoint_path = path[n // 2]
        midpoint_linear = (z0 + z1) / 2.0
        return float((midpoint_path - midpoint_linear).norm().item())

    def smoothness(self, path: Tensor) -> float:
        """Mean cosine similarity between consecutive step directions.

        A perfectly straight path returns 1.0.

        Args:
            path: Tensor of shape ``(n_steps, d)``.

        Returns:
            Mean cosine similarity of adjacent direction vectors.
        """
        dirs = path[1:] - path[:-1]  # (n_steps-1, d)
        # Need at least 2 direction vectors to compute pairwise similarity
        if dirs.shape[0] < 2:
            return 1.0

        d0 = dirs[:-1]  # (n_steps-2, d)
        d1 = dirs[1:]  # (n_steps-2, d)

        eps = 1e-8
        norm0 = d0.norm(dim=-1, keepdim=True).clamp(min=eps)
        norm1 = d1.norm(dim=-1, keepdim=True).clamp(min=eps)

        cos_sim = (d0 / norm0 * d1 / norm1).sum(dim=-1)  # (n_steps-2,)
        return float(cos_sim.mean().item())
