"""Advanced Weight Merging: TIES, DARE, and SLERP.

Implements three model merging strategies for combining fine-tuned models:

TIES (Trim, Elect Sign, Merge) — Yadav et al. 2023:
    1. Trim: zero out smallest (1-density) fraction of task vector values.
    2. Elect sign: majority vote on sign across task vectors per position.
    3. Merge: average aligned values, scale by alpha, add to base.

DARE (Drop And REscale) — Yu et al. 2023:
    Randomly drop (1-density) fraction of delta values, rescale remaining
    by 1/density to preserve expected magnitude, then average and add to base.

SLERP (Spherical Linear Interpolation):
    Interpolate two weight sets along the great-circle arc on the unit
    hypersphere, preserving angular relationships.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MergeConfig:
    """Configuration for advanced model merging."""

    method: str = "ties"  # "ties" | "dare" | "slerp"
    density: float = 0.9
    alpha: float = 0.5
    seed: int = 42


# ---------------------------------------------------------------------------
# Task vectors
# ---------------------------------------------------------------------------


def compute_task_vector(
    base_weights: dict[str, torch.Tensor],
    finetuned_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return {name: finetuned - base} for all matching keys."""
    tv: dict[str, torch.Tensor] = {}
    for name in base_weights:
        if name in finetuned_weights and finetuned_weights[name].shape == base_weights[name].shape:
            tv[name] = finetuned_weights[name].float() - base_weights[name].float()
    return tv


# ---------------------------------------------------------------------------
# TIES helpers
# ---------------------------------------------------------------------------


def ties_trim(
    task_vectors: list[dict[str, torch.Tensor]],
    density: float,
) -> list[dict[str, torch.Tensor]]:
    """Zero out the smallest (1-density) fraction by magnitude per param."""
    trimmed = []
    for tv in task_vectors:
        new_tv: dict[str, torch.Tensor] = {}
        for name, delta in tv.items():
            flat = delta.abs().flatten()
            k = max(1, int(math.ceil(density * flat.numel())))
            if k >= flat.numel():
                new_tv[name] = delta.clone()
            else:
                threshold = flat.kthvalue(flat.numel() - k + 1).values.item()
                mask = delta.abs() >= threshold
                new_tv[name] = delta * mask.to(delta.dtype)
        trimmed.append(new_tv)
    return trimmed


def ties_elect_sign(
    task_vectors: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Majority-vote on sign across task vectors for each position.

    Returns a dict of sign tensors (+1 or -1) per parameter name.
    """
    signs: dict[str, torch.Tensor] = {}
    all_keys = task_vectors[0].keys() if task_vectors else []
    for name in all_keys:
        # Sum of signs across task vectors; tie goes to positive
        sign_sum = torch.zeros_like(task_vectors[0][name])
        for tv in task_vectors:
            sign_sum = sign_sum + torch.sign(tv[name])
        elected = torch.sign(sign_sum)
        # Where sum is exactly 0, default to +1
        elected[elected == 0] = 1.0
        signs[name] = elected
    return signs


def ties_merge(
    base_weights: dict[str, torch.Tensor],
    task_vectors: list[dict[str, torch.Tensor]],
    density: float,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Full TIES merge: trim, elect sign, average aligned, add to base."""
    trimmed = ties_trim(task_vectors, density)
    elected_signs = ties_elect_sign(trimmed)

    merged: dict[str, torch.Tensor] = {}
    for name in base_weights:
        if name not in elected_signs:
            merged[name] = base_weights[name].clone()
            continue

        sign = elected_signs[name]
        # Average only values whose sign matches the elected sign
        acc = torch.zeros_like(base_weights[name], dtype=torch.float32)
        count = torch.zeros_like(base_weights[name], dtype=torch.float32)
        for tv in trimmed:
            delta = tv[name]
            aligned = (torch.sign(delta) == sign) & (delta != 0)
            acc = acc + delta * aligned.to(delta.dtype)
            count = count + aligned.float()

        count = count.clamp(min=1.0)
        avg_delta = acc / count
        merged[name] = base_weights[name].float() + alpha * avg_delta
        merged[name] = merged[name].to(base_weights[name].dtype)

    return merged


# ---------------------------------------------------------------------------
# DARE helpers
# ---------------------------------------------------------------------------


def dare_mask(
    task_vector: dict[str, torch.Tensor],
    density: float,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Randomly drop (1-density) fraction, rescale remaining by 1/density."""
    masked: dict[str, torch.Tensor] = {}
    for name, delta in task_vector.items():
        gen = torch.Generator()
        gen.manual_seed(seed + hash(name) % (2**31))
        mask = torch.bernoulli(torch.full_like(delta, density, dtype=torch.float32), generator=gen)
        rescaled = delta * mask.to(delta.dtype) / max(density, 1e-8)
        masked[name] = rescaled
    return masked


def dare_merge(
    base_weights: dict[str, torch.Tensor],
    task_vectors: list[dict[str, torch.Tensor]],
    density: float,
    alpha: float,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """DARE merge: mask each task vector, average, add to base."""
    masked_tvs = [dare_mask(tv, density, seed=seed + i) for i, tv in enumerate(task_vectors)]

    merged: dict[str, torch.Tensor] = {}
    for name in base_weights:
        deltas = [tv[name] for tv in masked_tvs if name in tv]
        if not deltas:
            merged[name] = base_weights[name].clone()
            continue
        avg_delta = torch.stack(deltas).float().mean(dim=0)
        merged[name] = base_weights[name].float() + alpha * avg_delta
        merged[name] = merged[name].to(base_weights[name].dtype)

    return merged


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------


def slerp_merge(
    weights_a: dict[str, torch.Tensor],
    weights_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Spherical linear interpolation per tensor between two state dicts."""
    merged: dict[str, torch.Tensor] = {}
    eps = 1e-8
    for name in weights_a:
        if name not in weights_b:
            merged[name] = weights_a[name].clone()
            continue

        va = weights_a[name].float().flatten()
        vb = weights_b[name].float().flatten()

        norm_a = va.norm()
        norm_b = vb.norm()

        if norm_a < eps or norm_b < eps:
            # Fallback to linear interpolation
            merged[name] = (
                (1 - alpha) * weights_a[name].float() + alpha * weights_b[name].float()
            ).to(weights_a[name].dtype)
            continue

        va_unit = va / norm_a
        vb_unit = vb / norm_b
        cos_omega = torch.clamp(torch.dot(va_unit, vb_unit), -1.0, 1.0)
        omega = torch.acos(cos_omega)

        if omega.abs() < eps:
            # Nearly parallel, linear interpolation
            result = (1 - alpha) * va + alpha * vb
        else:
            sin_omega = torch.sin(omega)
            result = (torch.sin((1 - alpha) * omega) / sin_omega) * va + (
                torch.sin(alpha * omega) / sin_omega
            ) * vb

        merged[name] = result.view(weights_a[name].shape).to(weights_a[name].dtype)

    return merged


# ---------------------------------------------------------------------------
# ModelMerger
# ---------------------------------------------------------------------------


class ModelMerger:
    """Dispatch-based model merger supporting TIES, DARE, and SLERP."""

    def __init__(self, config: MergeConfig) -> None:
        self.config = config

    def merge(
        self,
        base_model: nn.Module,
        finetuned_models: Sequence[nn.Module],
    ) -> dict[str, torch.Tensor]:
        """Merge finetuned models into a single state dict.

        Args:
            base_model: The base (pre-trained) model.
            finetuned_models: One or more fine-tuned model variants.

        Returns:
            Merged state dict.
        """
        base_weights = {k: v.clone() for k, v in base_model.state_dict().items()}

        if self.config.method == "slerp":
            if len(finetuned_models) != 2:
                raise ValueError("SLERP requires exactly 2 finetuned models")
            wa = finetuned_models[0].state_dict()
            wb = finetuned_models[1].state_dict()
            return slerp_merge(wa, wb, self.config.alpha)

        # Compute task vectors for TIES and DARE
        task_vectors = [
            compute_task_vector(base_weights, {k: v.clone() for k, v in m.state_dict().items()})
            for m in finetuned_models
        ]

        if self.config.method == "ties":
            return ties_merge(base_weights, task_vectors, self.config.density, self.config.alpha)
        elif self.config.method == "dare":
            return dare_merge(
                base_weights, task_vectors, self.config.density, self.config.alpha, self.config.seed
            )
        else:
            raise ValueError(f"Unknown merge method: {self.config.method}")
