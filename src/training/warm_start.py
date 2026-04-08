"""Warm-start utilities for partially loading model weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WarmStartReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatch_keys: tuple[str, ...]


def warm_start_state(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], WarmStartReport]:
    """Load matching source tensors into a target state dict copy."""
    updated: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    missing_keys: list[str] = []
    shape_mismatch_keys: list[str] = []

    for key, target_tensor in target_state.items():
        if key not in source_state:
            updated[key] = target_tensor.clone()
            missing_keys.append(key)
            continue
        source_tensor = source_state[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            updated[key] = target_tensor.clone()
            shape_mismatch_keys.append(key)
            continue
        updated[key] = source_tensor.detach().clone()
        loaded_keys.append(key)

    return updated, WarmStartReport(
        loaded_keys=tuple(loaded_keys),
        missing_keys=tuple(missing_keys),
        shape_mismatch_keys=tuple(shape_mismatch_keys),
    )


def interpolation_warm_start(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend target and source tensors for gentle warm-starting."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if target_tensor.shape != source_tensor.shape:
        raise ValueError("target_tensor and source_tensor must match")
    return (1.0 - alpha) * target_tensor + alpha * source_tensor


def prefix_warm_start(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Copy the largest matching prefix from source into target."""
    if target_tensor.dim() != source_tensor.dim():
        raise ValueError("target_tensor and source_tensor must have the same rank")
    if dim < 0 or dim >= target_tensor.dim():
        raise ValueError(f"dim must be in [0, {target_tensor.dim() - 1}], got {dim}")
    result = target_tensor.clone()
    copy_shape = list(target_tensor.shape)
    copy_shape[dim] = min(target_tensor.shape[dim], source_tensor.shape[dim])
    slices = tuple(slice(0, size) for size in copy_shape)
    result[slices] = source_tensor[slices]
    return result
