"""Loss-landscape utilities for interpolation and local sharpness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


def flatten_parameters(params: list[torch.Tensor]) -> torch.Tensor:
    """Flatten a parameter list to one vector."""
    if not params:
        return torch.empty(0)
    return torch.cat([param.reshape(-1) for param in params])


def unflatten_like(vector: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a flat vector into tensors matching reference shapes."""
    outputs: list[torch.Tensor] = []
    offset = 0
    for ref in like:
        numel = ref.numel()
        outputs.append(vector[offset : offset + numel].reshape_as(ref))
        offset += numel
    if offset != vector.numel():
        raise ValueError("Vector size does not match reference tensors")
    return outputs


def interpolate_parameters(
    params_a: list[torch.Tensor],
    params_b: list[torch.Tensor],
    alpha: float,
) -> list[torch.Tensor]:
    """Linearly interpolate two parameter sets."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if len(params_a) != len(params_b):
        raise ValueError("Parameter lists must have the same length")
    outputs = []
    for tensor_a, tensor_b in zip(params_a, params_b):
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Parameter shapes must match")
        outputs.append((1.0 - alpha) * tensor_a + alpha * tensor_b)
    return outputs


def random_direction_like(params: list[torch.Tensor], normalize: bool = True) -> list[torch.Tensor]:
    """Sample a random direction in parameter space."""
    direction = [torch.randn_like(param) for param in params]
    if normalize and direction:
        flat = flatten_parameters(direction)
        norm = flat.norm().clamp_min(1e-8)
        direction = [tensor / norm for tensor in direction]
    return direction


def perturb_parameters(
    params: list[torch.Tensor],
    direction: list[torch.Tensor],
    epsilon: float,
) -> list[torch.Tensor]:
    """Perturb parameters along a chosen direction."""
    if len(params) != len(direction):
        raise ValueError("params and direction must have the same length")
    return [param + epsilon * delta for param, delta in zip(params, direction)]


@dataclass(frozen=True)
class LandscapeSlice:
    alphas: torch.Tensor
    losses: torch.Tensor


def interpolation_slice(
    params_a: list[torch.Tensor],
    params_b: list[torch.Tensor],
    loss_fn,
    n_points: int = 5,
) -> LandscapeSlice:
    """Evaluate loss along a line segment in parameter space."""
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    alphas = torch.linspace(0.0, 1.0, n_points)
    losses = []
    for alpha in alphas:
        params = interpolate_parameters(params_a, params_b, float(alpha.item()))
        losses.append(loss_fn(params))
    return LandscapeSlice(alphas=alphas, losses=torch.stack(losses))


def local_sharpness(base_loss: torch.Tensor, perturbed_losses: torch.Tensor) -> torch.Tensor:
    """Compute max loss increase over a local perturbation set."""
    return (perturbed_losses - base_loss).max()


# ---------------------------------------------------------------------------
# LossLandscapeExplorer and LandscapeStats
# ---------------------------------------------------------------------------

class LossLandscapeExplorer:
    """
    Explore loss landscape by perturbing parameters along random/specific directions.

    Args:
        model: the model to analyze
        loss_fn: callable(model) -> scalar loss
        filter_normalization: normalize perturbation directions by parameter norms
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        filter_normalization: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.filter_normalization = filter_normalization

    def _get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a flat vector."""
        return torch.cat([p.detach().view(-1) for p in self.model.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Set all parameters from a flat vector."""
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(flat_params[offset:offset + n].view_as(p))
            offset += n

    def _random_direction(self) -> torch.Tensor:
        """
        Sample a random direction in parameter space.
        If filter_normalization: normalize direction to have same per-layer norm as params.
        Returns: flat direction vector.
        """
        if self.filter_normalization:
            # Filter normalization: scale each layer's direction to match param norm
            parts = []
            for p in self.model.parameters():
                d = torch.randn_like(p)
                p_norm = p.data.norm()
                d_norm = d.norm().clamp_min(1e-8)
                # Scale direction so its norm matches parameter norm
                d = d * (p_norm / d_norm)
                parts.append(d.view(-1))
            return torch.cat(parts)
        else:
            flat = self._get_flat_params()
            d = torch.randn_like(flat)
            d = d / d.norm().clamp_min(1e-8)
            return d

    def line_scan(
        self,
        direction: Optional[torch.Tensor] = None,
        alphas: Optional[torch.Tensor] = None,
        n_points: int = 20,
        alpha_range: tuple = (-1.0, 1.0),
    ) -> dict:
        """
        Scan loss along one direction from current parameters.

        Returns: {'alphas': tensor, 'losses': tensor}
        Restores original parameters after scan.
        """
        original_params = self._get_flat_params().clone()

        if direction is None:
            direction = self._random_direction()

        if alphas is None:
            alphas = torch.linspace(alpha_range[0], alpha_range[1], n_points)

        losses = []
        with torch.no_grad():
            for alpha in alphas:
                new_params = original_params + alpha.item() * direction
                self._set_flat_params(new_params)
                loss = self.loss_fn(self.model)
                if isinstance(loss, torch.Tensor):
                    losses.append(loss.detach().cpu())
                else:
                    losses.append(torch.tensor(float(loss)))

        # Restore original parameters
        self._set_flat_params(original_params)

        return {
            'alphas': alphas,
            'losses': torch.stack(losses),
        }

    def surface_scan(
        self,
        direction1: Optional[torch.Tensor] = None,
        direction2: Optional[torch.Tensor] = None,
        n_points: int = 10,
        alpha_range: tuple = (-1.0, 1.0),
    ) -> dict:
        """
        Scan loss over a 2D grid (two directions).

        Returns: {'alphas1': tensor, 'alphas2': tensor, 'losses': (n,n) tensor}
        Restores original parameters after scan.
        """
        original_params = self._get_flat_params().clone()

        if direction1 is None:
            direction1 = self._random_direction()
        if direction2 is None:
            direction2 = self._random_direction()

        alphas = torch.linspace(alpha_range[0], alpha_range[1], n_points)
        loss_grid = torch.zeros(n_points, n_points)

        with torch.no_grad():
            for i, a1 in enumerate(alphas):
                for j, a2 in enumerate(alphas):
                    new_params = original_params + a1.item() * direction1 + a2.item() * direction2
                    self._set_flat_params(new_params)
                    loss = self.loss_fn(self.model)
                    if isinstance(loss, torch.Tensor):
                        loss_grid[i, j] = loss.detach().cpu().item()
                    else:
                        loss_grid[i, j] = float(loss)

        # Restore original parameters
        self._set_flat_params(original_params)

        return {
            'alphas1': alphas,
            'alphas2': alphas,
            'losses': loss_grid,
        }

    def flatness_score(self, epsilon: float = 0.1, n_directions: int = 5) -> float:
        """
        Estimate flatness: mean loss increase over random perturbations of size epsilon.
        Lower = flatter = better generalization.
        Returns: mean(loss(perturbed) - loss(original))
        """
        original_params = self._get_flat_params().clone()

        with torch.no_grad():
            base_loss = self.loss_fn(self.model)
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.item()

        increases = []
        for _ in range(n_directions):
            direction = self._random_direction()
            # Normalize direction to unit norm then scale by epsilon
            direction = direction / direction.norm().clamp_min(1e-8)
            perturbed = original_params + epsilon * direction

            with torch.no_grad():
                self._set_flat_params(perturbed)
                pert_loss = self.loss_fn(self.model)
                if isinstance(pert_loss, torch.Tensor):
                    pert_loss = pert_loss.item()
                increases.append(pert_loss - base_loss)

        self._set_flat_params(original_params)
        return float(sum(increases) / len(increases)) if increases else 0.0

    def sharpness_ratio(self, epsilon: float = 0.1, n_directions: int = 10) -> float:
        """
        SAM-style sharpness: max loss increase in epsilon-ball.
        Samples multiple directions, returns max increase.
        """
        original_params = self._get_flat_params().clone()

        with torch.no_grad():
            base_loss = self.loss_fn(self.model)
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.item()

        max_increase = 0.0
        for _ in range(n_directions):
            direction = self._random_direction()
            direction = direction / direction.norm().clamp_min(1e-8)
            perturbed = original_params + epsilon * direction

            with torch.no_grad():
                self._set_flat_params(perturbed)
                pert_loss = self.loss_fn(self.model)
                if isinstance(pert_loss, torch.Tensor):
                    pert_loss = pert_loss.item()
                increase = pert_loss - base_loss
                if increase > max_increase:
                    max_increase = increase

        self._set_flat_params(original_params)
        return float(max_increase)


class LandscapeStats:
    """Statistics about the loss landscape."""

    def __init__(self, losses: torch.Tensor) -> None:
        self.losses = losses

    def local_minima_count(self, window: int = 3) -> int:
        """Count local minima in 1D loss profile (for line_scan output)."""
        losses = self.losses
        if losses.numel() < 3:
            return 0

        losses_1d = losses.view(-1)
        n = losses_1d.numel()
        count = 0
        half = window // 2

        for i in range(half, n - half):
            # Check if losses[i] is less than all neighbors within window
            window_vals = losses_1d[max(0, i - half):min(n, i + half + 1)]
            if losses_1d[i] == window_vals.min() and (window_vals < losses_1d[i]).sum() == 0:
                # Strictly less than all other values in window
                others = torch.cat([losses_1d[max(0, i - half):i], losses_1d[i + 1:min(n, i + half + 1)]])
                if len(others) > 0 and (losses_1d[i] < others).all():
                    count += 1

        return max(0, count)

    def curvature_at_center(self) -> float:
        """Estimate curvature via finite differences at center point."""
        losses_1d = self.losses.view(-1).float()
        n = losses_1d.numel()
        if n < 3:
            return 0.0

        center = n // 2
        # Second-order finite difference: f''(x) ≈ f(x+h) - 2f(x) + f(x-h)
        curvature = (losses_1d[center + 1] - 2.0 * losses_1d[center] + losses_1d[center - 1]).item()
        return float(curvature)

    def is_convex(self) -> bool:
        """Check if loss profile is convex (losses form a bowl shape)."""
        losses_1d = self.losses.view(-1).float()
        n = losses_1d.numel()
        if n < 3:
            return True

        # A 1D function is convex if second differences are >= 0
        for i in range(1, n - 1):
            second_diff = (losses_1d[i + 1] - 2.0 * losses_1d[i] + losses_1d[i - 1]).item()
            if second_diff < -1e-6:
                return False
        return True
