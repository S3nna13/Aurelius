"""
model_merging.py — Model merging utilities for Aurelius LLM.

Supported methods:
  - linear : weighted average of parameters
  - slerp  : spherical linear interpolation (two models)
  - ties   : TIES merging via majority-sign election
  - dare   : DARE random masking of task vectors before merging
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """Configuration for model merging."""

    method: str = "linear"
    """One of: 'linear', 'slerp', 'ties', 'dare'."""

    weights: Optional[List[float]] = None
    """Per-model weights that must sum to 1. None => uniform."""

    density: float = 1.0
    """Fraction of task-vector parameters to keep (used by DARE)."""

    lambda_coeff: float = 1.0
    """Scaling coefficient applied to the merged delta (used by TIES)."""


# ---------------------------------------------------------------------------
# Core merge functions
# ---------------------------------------------------------------------------

def linear_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Weighted average of all parameters across *state_dicts*.

    Args:
        state_dicts: List of state dicts (must share the same keys).
        weights: Per-model scalar weights; must sum to 1.

    Returns:
        Merged state dict.
    """
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"Number of state_dicts ({len(state_dicts)}) must match "
            f"number of weights ({len(weights)})."
        )
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-5:
        raise ValueError(f"Weights must sum to 1, got {weight_sum:.6f}.")

    keys = list(state_dicts[0].keys())
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = weights[0] * state_dicts[0][k].float()
        for sd, w in zip(state_dicts[1:], weights[1:]):
            acc = acc + w * sd[k].float()
        merged[k] = acc.to(state_dicts[0][k].dtype)
    return merged


def slerp_merge(
    sd1: Dict[str, torch.Tensor],
    sd2: Dict[str, torch.Tensor],
    t: float,
) -> Dict[str, torch.Tensor]:
    """Spherical linear interpolation between two state dicts.

    For each parameter tensor the function computes SLERP(p1, p2, t).
    Falls back to linear interpolation when the tensors are 0-D/1-D, or
    when either norm is near zero, or when the angle between them is
    near zero (nearly parallel / anti-parallel).

    Args:
        sd1: First model's state dict (t=0 endpoint).
        sd2: Second model's state dict (t=1 endpoint).
        t:   Interpolation factor in [0, 1].

    Returns:
        Merged state dict.
    """
    merged: Dict[str, torch.Tensor] = {}
    for k in sd1:
        p1 = sd1[k].float()
        p2 = sd2[k].float()
        original_dtype = sd1[k].dtype

        # Scalars or 1-D tensors: plain linear interpolation
        if p1.ndim <= 1:
            merged[k] = ((1.0 - t) * p1 + t * p2).to(original_dtype)
            continue

        # Flatten to vectors for the SLERP computation
        shape = p1.shape
        v1 = p1.reshape(-1)
        v2 = p2.reshape(-1)

        n1 = torch.norm(v1)
        n2 = torch.norm(v2)

        # Fall back to linear if either norm is near zero
        if n1 < 1e-8 or n2 < 1e-8:
            merged[k] = ((1.0 - t) * p1 + t * p2).to(original_dtype)
            continue

        v1_norm = v1 / n1
        v2_norm = v2 / n2

        dot = torch.dot(v1_norm, v2_norm).clamp(-1.0, 1.0)
        omega = torch.acos(dot)

        sin_omega = torch.sin(omega)

        # Near-zero angle: fall back to linear
        if sin_omega.abs() < 1e-8:
            merged[k] = ((1.0 - t) * p1 + t * p2).to(original_dtype)
            continue

        coeff1 = torch.sin((1.0 - t) * omega) / sin_omega
        coeff2 = torch.sin(t * omega) / sin_omega

        # SLERP operates on unit vectors; scale result by interpolated norm
        interp_norm = (1.0 - t) * n1 + t * n2
        result = (coeff1 * v1_norm + coeff2 * v2_norm) * interp_norm
        merged[k] = result.reshape(shape).to(original_dtype)

    return merged


# ---------------------------------------------------------------------------
# Task-vector helpers
# ---------------------------------------------------------------------------

def compute_task_vector(
    base_sd: Dict[str, torch.Tensor],
    finetuned_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Element-wise difference: finetuned − base.

    Args:
        base_sd:      State dict of the pre-trained base model.
        finetuned_sd: State dict of the fine-tuned model.

    Returns:
        Task-vector dict with the same keys.
    """
    return {
        k: (finetuned_sd[k].float() - base_sd[k].float()).to(base_sd[k].dtype)
        for k in base_sd
    }


def apply_task_vector(
    base_sd: Dict[str, torch.Tensor],
    task_vector: Dict[str, torch.Tensor],
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Apply a task vector to the base model: base + scale * task_vector.

    Args:
        base_sd:     State dict of the base model.
        task_vector: Task vector produced by :func:`compute_task_vector`.
        scale:       Scalar multiplier (0 → base, 1 → fine-tuned).

    Returns:
        Merged state dict.
    """
    return {
        k: (base_sd[k].float() + scale * task_vector[k].float()).to(base_sd[k].dtype)
        for k in base_sd
    }


# ---------------------------------------------------------------------------
# TIES merging
# ---------------------------------------------------------------------------

def ties_merge(
    base_sd: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    lambda_coeff: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """TIES merging (Trim, Elect Sign, Disjoint Merge).

    Algorithm per parameter tensor:
      1. Compute the per-element sign of each task vector.
      2. Elect the majority sign (ties broken by positive).
      3. For each task vector, zero out elements that disagree with the
         majority sign.
      4. Average the remaining (non-zero) values; scale by lambda_coeff.
      5. Add the scaled delta to the base model.

    Args:
        base_sd:      Base model state dict.
        task_vectors: List of task-vector dicts (same keys as base_sd).
        lambda_coeff: Scaling factor for the merged delta.

    Returns:
        Merged state dict.
    """
    n = len(task_vectors)
    merged: Dict[str, torch.Tensor] = {}

    for k in base_sd:
        base_param = base_sd[k].float()

        # Stack task vectors for this parameter: shape (n, *param_shape)
        stacked = torch.stack([tv[k].float() for tv in task_vectors], dim=0)

        # Step 1 & 2: majority sign election
        signs = torch.sign(stacked)               # (n, *param_shape)
        sign_sum = signs.sum(dim=0)               # (*param_shape)
        majority_sign = torch.sign(sign_sum)      # (*param_shape)
        # Break ties (sign_sum == 0) → positive
        majority_sign = torch.where(
            majority_sign == 0,
            torch.ones_like(majority_sign),
            majority_sign,
        )

        # Step 3: mask out disagreeing elements
        agree_mask = signs == majority_sign.unsqueeze(0)  # (n, *param_shape)
        masked = stacked * agree_mask.float()

        # Step 4: average non-zero contributions
        count = agree_mask.float().sum(dim=0).clamp(min=1.0)
        delta = masked.sum(dim=0) / count

        # Step 5: add scaled delta to base
        merged[k] = (base_param + lambda_coeff * delta).to(base_sd[k].dtype)

    return merged


# ---------------------------------------------------------------------------
# DARE masking
# ---------------------------------------------------------------------------

def dare_mask(
    task_vector: Dict[str, torch.Tensor],
    density: float,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Randomly drop (1 − density) of each task-vector parameter, then rescale.

    Dropped entries are set to zero; surviving entries are divided by
    *density* so the expected value is preserved.

    Args:
        task_vector: Task vector to mask.
        density:     Fraction of elements to *keep* in [0, 1].
        seed:        RNG seed for reproducibility.

    Returns:
        Masked (and rescaled) task vector with the same keys.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0, 1], got {density}.")

    rng = torch.Generator()
    rng.manual_seed(seed)

    masked: Dict[str, torch.Tensor] = {}
    for k, v in task_vector.items():
        v_f = v.float()
        mask = torch.bernoulli(
            torch.full(v_f.shape, density, dtype=torch.float32),
            generator=rng,
        )
        if density > 0.0:
            masked[k] = (v_f * mask / density).to(v.dtype)
        else:
            masked[k] = torch.zeros_like(v)
    return masked


# ---------------------------------------------------------------------------
# ModelMerger class
# ---------------------------------------------------------------------------

class ModelMerger:
    """High-level interface for merging multiple :class:`nn.Module` instances.

    Usage::

        config = MergeConfig(method="linear")
        merger = ModelMerger(config)
        merged_sd = merger.merge([model_a, model_b])
        model_a = merger.load_merged(model_a, merged_sd)
    """

    def __init__(self, config: MergeConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, models: List[nn.Module]) -> Dict[str, torch.Tensor]:
        """Merge a list of models and return the merged state dict.

        Args:
            models: List of :class:`nn.Module` instances to merge.

        Returns:
            Merged state dict (plain Python dict of tensors).
        """
        if len(models) < 2:
            raise ValueError("At least two models are required for merging.")

        state_dicts = [m.state_dict() for m in models]
        cfg = self.config

        # Resolve weights
        weights = cfg.weights
        if weights is None:
            w = 1.0 / len(models)
            weights = [w] * len(models)

        method = cfg.method.lower()

        if method == "linear":
            return linear_merge(state_dicts, weights)

        elif method == "slerp":
            if len(models) != 2:
                raise ValueError("SLERP merging requires exactly 2 models.")
            t = weights[1]  # interpret second weight as the interpolation factor
            return slerp_merge(state_dicts[0], state_dicts[1], t)

        elif method == "ties":
            base_sd = state_dicts[0]
            task_vectors = [
                compute_task_vector(base_sd, sd) for sd in state_dicts[1:]
            ]
            return ties_merge(base_sd, task_vectors, lambda_coeff=cfg.lambda_coeff)

        elif method == "dare":
            base_sd = state_dicts[0]
            task_vectors = [
                dare_mask(compute_task_vector(base_sd, sd), cfg.density)
                for sd in state_dicts[1:]
            ]
            # After DARE masking, apply linear merge of the task vectors
            n_tv = len(task_vectors)
            tv_weight = 1.0 / n_tv
            averaged_tv: Dict[str, torch.Tensor] = {}
            for k in base_sd:
                acc = task_vectors[0][k].float()
                for tv in task_vectors[1:]:
                    acc = acc + tv[k].float()
                averaged_tv[k] = (acc * tv_weight).to(base_sd[k].dtype)
            return apply_task_vector(base_sd, averaged_tv, scale=cfg.lambda_coeff)

        else:
            raise ValueError(
                f"Unknown merge method '{cfg.method}'. "
                "Choose one of: 'linear', 'slerp', 'ties', 'dare'."
            )

    def load_merged(
        self,
        model: nn.Module,
        merged_sd: Dict[str, torch.Tensor],
    ) -> nn.Module:
        """Load a merged state dict into *model* (in-place) and return it.

        Args:
            model:     Target :class:`nn.Module`.
            merged_sd: Merged state dict produced by :meth:`merge`.

        Returns:
            The same *model* with updated weights.
        """
        model.load_state_dict(merged_sd)
        return model
