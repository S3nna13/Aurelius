"""Task vector arithmetic for gradient-free model editing.

Task vectors are the diff between fine-tuned and pretrained weights.
Arithmetic on task vectors enables capability editing, negation, and composition.

Reference: Ilharco et al. 2022, arXiv:2212.04089
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


class TaskVector:
    """Represents the weight diff between a fine-tuned and pretrained model.

    Supports arithmetic: +, -, * (scalar), negation.
    """

    def __init__(
        self,
        pretrained: nn.Module | None = None,
        finetuned: nn.Module | None = None,
        vector: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Create from model pair OR from pre-computed vector dict.

        Args:
            pretrained: Base model
            finetuned: Fine-tuned model (same architecture)
            vector: Pre-computed {param_name: diff_tensor} dict
        """
        if vector is not None:
            self.vector = vector
        elif pretrained is not None and finetuned is not None:
            self.vector = {
                name: finetuned_param.data.clone() - pretrained_param.data.clone()
                for (name, pretrained_param), (_, finetuned_param)
                in zip(pretrained.named_parameters(), finetuned.named_parameters())
            }
        else:
            raise ValueError("Must provide either (pretrained, finetuned) or vector")

    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Element-wise sum of two task vectors (for capability composition)."""
        new_vector = {
            name: self.vector[name] + other.vector[name]
            for name in self.vector
        }
        return TaskVector(vector=new_vector)

    def __sub__(self, other: "TaskVector") -> "TaskVector":
        """Element-wise difference of task vectors."""
        new_vector = {
            name: self.vector[name] - other.vector[name]
            for name in self.vector
        }
        return TaskVector(vector=new_vector)

    def __mul__(self, scalar: float) -> "TaskVector":
        """Scale task vector by a scalar."""
        new_vector = {
            name: self.vector[name] * scalar
            for name in self.vector
        }
        return TaskVector(vector=new_vector)

    def __rmul__(self, scalar: float) -> "TaskVector":
        return self.__mul__(scalar)

    def __neg__(self) -> "TaskVector":
        """Negate task vector (for capability removal)."""
        return self.__mul__(-1.0)

    def norm(self) -> float:
        """Total L2 norm across all parameters."""
        total_sq = sum(
            param.float().norm().item() ** 2
            for param in self.vector.values()
        )
        return total_sq ** 0.5

    def apply(
        self,
        pretrained: nn.Module,
        scaling_coef: float = 1.0,
    ) -> nn.Module:
        """Apply task vector to pretrained model.

        Returns new model (deep copy) with θ = θ_pretrained + scaling_coef * τ.
        Does NOT modify the input model.
        """
        new_model = copy.deepcopy(pretrained)

        # Modify parameters in-place via named_parameters() to correctly handle
        # tied weights (e.g. embed.weight == lm_head.weight). load_state_dict()
        # would load tied keys sequentially and the last write would win, so we
        # use the parameter dict directly instead.
        pretrained_params = dict(pretrained.named_parameters())
        for name, param in new_model.named_parameters():
            if name in self.vector:
                delta = scaling_coef * self.vector[name].to(param.dtype)
                param.data.copy_(pretrained_params[name].data + delta)

        return new_model


def extract_task_vector(
    pretrained: nn.Module,
    finetuned: nn.Module,
) -> TaskVector:
    """Convenience function to extract task vector from model pair."""
    return TaskVector(pretrained, finetuned)


def apply_task_vectors(
    pretrained: nn.Module,
    task_vectors: list[TaskVector],
    scaling_coef: float = 1.0,
) -> nn.Module:
    """Apply multiple task vectors simultaneously.

    θ_new = θ_pretrained + scaling_coef * sum(τ_i)
    """
    if not task_vectors:
        return copy.deepcopy(pretrained)

    combined = task_vectors[0]
    for tv in task_vectors[1:]:
        combined = combined + tv

    return combined.apply(pretrained, scaling_coef=scaling_coef)


def negation_edit(
    pretrained: nn.Module,
    finetuned: nn.Module,
    scaling_coef: float = 1.0,
) -> nn.Module:
    """Remove a capability by negating its task vector.

    θ_new = θ_pretrained - scaling_coef * (θ_finetuned - θ_pretrained)
    """
    tv = TaskVector(pretrained, finetuned)
    return (-tv).apply(pretrained, scaling_coef=scaling_coef)
