"""Loss-landscape utilities for interpolation and local sharpness analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
