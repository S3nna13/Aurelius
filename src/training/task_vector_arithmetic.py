"""Task Vector Arithmetic for multi-task model composition.

Implements task vectors (θ_ft - θ_base) with arithmetic operations for
gradient-free model editing, composition, and forgetting.

Reference: Ilharco et al. 2023, "Editing Models with Task Arithmetic", arXiv:2212.04089
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TaskVectorConfig:
    """Configuration for task vector arithmetic operations."""

    scaling_coef: float = 1.0
    """λ for task vector application: θ_new = θ_base + λ * τ"""

    normalize: bool = False
    """Normalize individual vectors to unit norm before combining."""


class TaskVector:
    """Represents the task-specific weight delta: θ_ft - θ_base.

    Supports arithmetic operations (+, -, *, negation) to enable
    multi-task composition, task forgetting, and interpolation.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        finetuned_model: Optional[nn.Module] = None,
        vector: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        """Initialize from (base, finetuned) pair or directly from vector dict.

        Args:
            base_model: The pretrained base model.
            finetuned_model: The fine-tuned model (same architecture as base).
            vector: Pre-computed {param_name: finetuned_param - base_param} dict.
        """
        if vector is not None:
            self.vector: Dict[str, Tensor] = vector
        elif base_model is not None and finetuned_model is not None:
            self.vector = {
                name: ft_param.data.clone() - base_param.data.clone()
                for (name, base_param), (_, ft_param)
                in zip(base_model.named_parameters(), finetuned_model.named_parameters())
            }
        else:
            raise ValueError(
                "Must provide either (base_model, finetuned_model) pair or a vector dict."
            )

    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Add two task vectors element-wise."""
        new_vector = {
            name: self.vector[name] + other.vector[name]
            for name in self.vector
            if name in other.vector
        }
        return TaskVector(vector=new_vector)

    def __neg__(self) -> "TaskVector":
        """Negate task vector (for task forgetting)."""
        return self.__mul__(-1.0)

    def __sub__(self, other: "TaskVector") -> "TaskVector":
        """Subtract two task vectors."""
        new_vector = {
            name: self.vector[name] - other.vector[name]
            for name in self.vector
            if name in other.vector
        }
        return TaskVector(vector=new_vector)

    def __mul__(self, scalar: float) -> "TaskVector":
        """Scale task vector by scalar."""
        new_vector = {
            name: self.vector[name] * scalar
            for name in self.vector
        }
        return TaskVector(vector=new_vector)

    def __rmul__(self, scalar: float) -> "TaskVector":
        """Right multiply: scalar * task_vector."""
        return self.__mul__(scalar)

    def norm(self) -> float:
        """L2 norm of all concatenated task vector parameters."""
        total_sq = sum(
            param.float().reshape(-1).norm().item() ** 2
            for param in self.vector.values()
        )
        return math.sqrt(total_sq)

    def apply(
        self,
        base_model: nn.Module,
        scaling_coef: float = 1.0,
    ) -> nn.Module:
        """Apply task vector to base model: θ_new = θ_base + λ * τ.

        Returns a new model (deep copy of base_model + task vector applied).
        Does NOT modify the input base_model.

        Args:
            base_model: The base model to apply the task vector to.
            scaling_coef: λ scaling coefficient.

        Returns:
            New nn.Module with updated weights.
        """
        new_model = copy.deepcopy(base_model)
        base_params = dict(base_model.named_parameters())

        for name, param in new_model.named_parameters():
            if name in self.vector:
                delta = scaling_coef * self.vector[name].to(param.dtype)
                param.data.copy_(base_params[name].data + delta)

        return new_model

    def cosine_similarity(self, other: "TaskVector") -> float:
        """Cosine similarity between two task vectors (flat parameter vectors).

        Args:
            other: Another TaskVector to compare against.

        Returns:
            Cosine similarity in [-1, 1].
        """
        shared_keys = sorted(set(self.vector.keys()) & set(other.vector.keys()))
        if not shared_keys:
            return 0.0

        flat_self = torch.cat([self.vector[k].reshape(-1).float() for k in shared_keys])
        flat_other = torch.cat([other.vector[k].reshape(-1).float() for k in shared_keys])

        norm_self = flat_self.norm()
        norm_other = flat_other.norm()

        if norm_self == 0 or norm_other == 0:
            return 0.0

        dot = (flat_self * flat_other).sum()
        return (dot / (norm_self * norm_other)).item()

    def sparsify(self, fraction: float) -> "TaskVector":
        """Zero out (1 - fraction) of parameters with smallest absolute value.

        Implements magnitude-based pruning (DARE-style): keeps only the top
        `fraction` of parameters by absolute magnitude.

        Args:
            fraction: Fraction of parameters to keep (0.0 to 1.0).

        Returns:
            New TaskVector with low-magnitude parameters zeroed out.
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")

        new_vector: Dict[str, Tensor] = {}
        for name, param in self.vector.items():
            flat = param.reshape(-1).float()
            n_keep = max(1, int(round(flat.numel() * fraction)))
            threshold_idx = flat.abs().topk(n_keep, largest=True, sorted=False).indices
            mask = torch.zeros_like(flat)
            mask[threshold_idx] = 1.0
            mask = mask.reshape(param.shape).to(param.dtype)
            new_vector[name] = param * mask

        return TaskVector(vector=new_vector)


def multi_task_compose(
    base_model: nn.Module,
    task_vectors: List[TaskVector],
    weights: Optional[List[float]] = None,
    config: Optional[TaskVectorConfig] = None,
) -> nn.Module:
    """Compose multiple task vectors via weighted sum.

    θ = θ_base + λ * sum(w_i * τ_i)

    Args:
        base_model: The base model to apply vectors to.
        task_vectors: List of TaskVector objects to combine.
        weights: Per-vector weights (uniform 1.0 if None).
        config: TaskVectorConfig controlling scaling_coef and normalize.

    Returns:
        New nn.Module with composed weights.
    """
    if config is None:
        config = TaskVectorConfig()

    if not task_vectors:
        return copy.deepcopy(base_model)

    n = len(task_vectors)
    if weights is None:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError(
            f"len(weights)={len(weights)} must match len(task_vectors)={n}"
        )

    # Optionally normalize each task vector to unit norm
    vecs: List[TaskVector] = []
    for tv, w in zip(task_vectors, weights):
        if config.normalize:
            tv_norm = tv.norm()
            if tv_norm > 0:
                tv = tv * (1.0 / tv_norm)
        vecs.append(tv * w)

    # Sum all weighted task vectors
    combined = vecs[0]
    for tv in vecs[1:]:
        combined = combined + tv

    return combined.apply(base_model, scaling_coef=config.scaling_coef)


def task_negation(
    base_model: nn.Module,
    task_vector: TaskVector,
    scaling_coef: float = 1.0,
) -> nn.Module:
    """Apply negated task vector to forget a task.

    θ_new = θ_base - λ * τ

    Args:
        base_model: The base model.
        task_vector: Task vector representing the capability to forget.
        scaling_coef: λ scaling coefficient.

    Returns:
        New nn.Module with the task capability reduced.
    """
    return (-task_vector).apply(base_model, scaling_coef=scaling_coef)


def interpolate_models(
    model_a: nn.Module,
    model_b: nn.Module,
    alpha: float,
) -> nn.Module:
    """Linear interpolation between two models.

    θ = alpha * θ_a + (1 - alpha) * θ_b

    Args:
        model_a: First model (weight alpha).
        model_b: Second model (weight 1 - alpha).
        alpha: Interpolation coefficient in [0, 1].
            alpha=1.0 → model_a weights, alpha=0.0 → model_b weights.

    Returns:
        New nn.Module with interpolated weights.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    new_model = copy.deepcopy(model_a)
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    for name, param in new_model.named_parameters():
        if name in params_a and name in params_b:
            interp = alpha * params_a[name].data.float() + (1.0 - alpha) * params_b[name].data.float()
            param.data.copy_(interp.to(param.dtype))

    return new_model
