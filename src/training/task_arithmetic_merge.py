"""Task Arithmetic model merging (Ilharco et al. 2023, arXiv:2212.04089).

Enables multi-task model merging by operating directly on task vectors
(differences between fine-tuned and pretrained weights) without joint training.

Supported conflict-resolution strategies:
  - mean: simple (weighted) average
  - ties: TIES — zero out parameters where sign is not majority-consistent
  - dare: DARE — randomly drop values then rescale to preserve expected magnitude
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Type alias: parameter name → delta tensor
TaskVector = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Core functional operations
# ---------------------------------------------------------------------------

def extract_task_vector(
    pretrained_state: Dict[str, torch.Tensor],
    finetuned_state: Dict[str, torch.Tensor],
) -> TaskVector:
    """Compute τ = θ_ft - θ_pre for all shared parameters.

    Args:
        pretrained_state: state_dict of the base (pretrained) model.
        finetuned_state:  state_dict of the fine-tuned model.

    Returns:
        Dict mapping param name → delta tensor.
    """
    shared_keys = set(pretrained_state.keys()) & set(finetuned_state.keys())
    return {
        k: finetuned_state[k].clone() - pretrained_state[k].clone()
        for k in shared_keys
    }


def apply_task_vector(
    pretrained_state: Dict[str, torch.Tensor],
    task_vector: TaskVector,
    scaling: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Apply θ_merged = θ_pre + scaling * τ.

    Args:
        pretrained_state: Base weights (state_dict).
        task_vector:      Delta to apply.
        scaling:          λ coefficient.

    Returns:
        New state_dict with merged weights (pretrained_state is not mutated).
    """
    merged: Dict[str, torch.Tensor] = {}
    for k, v in pretrained_state.items():
        if k in task_vector:
            merged[k] = v.clone() + scaling * task_vector[k].to(v.dtype)
        else:
            merged[k] = v.clone()
    return merged


def add_task_vectors(
    task_vectors: List[TaskVector],
    weights: Optional[List[float]] = None,
) -> TaskVector:
    """Weighted sum of task vectors: sum_i w_i * τ_i.

    Args:
        task_vectors: List of task vector dicts.
        weights:      Per-vector weights; uniform (1/n) if None.

    Returns:
        Aggregated task vector.
    """
    if not task_vectors:
        return {}

    n = len(task_vectors)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"len(weights)={len(weights)} != len(task_vectors)={n}"
            )

    all_keys: set = set()
    for tv in task_vectors:
        all_keys |= set(tv.keys())

    result: TaskVector = {}
    for k in all_keys:
        acc: Optional[torch.Tensor] = None
        for w, tv in zip(weights, task_vectors):
            if k not in tv:
                continue
            delta = w * tv[k]
            acc = delta if acc is None else acc + delta
        if acc is not None:
            result[k] = acc
    return result


def negate_task_vector(task_vector: TaskVector) -> TaskVector:
    """Return -τ (useful for forgetting a task).

    Args:
        task_vector: Input task vector.

    Returns:
        Negated task vector.
    """
    return {k: -v.clone() for k, v in task_vector.items()}


def scale_task_vector(task_vector: TaskVector, scale: float) -> TaskVector:
    """Return scale * τ.

    Args:
        task_vector: Input task vector.
        scale:       Scalar multiplier.

    Returns:
        Scaled task vector.
    """
    return {k: scale * v.clone() for k, v in task_vector.items()}


def resolve_conflicts(
    task_vectors: List[TaskVector],
    method: str = "mean",
) -> TaskVector:
    """Resolve sign conflicts between multiple task vectors.

    Args:
        task_vectors: List of task vector dicts.
        method: One of:
            "mean" — simple mean (no conflict resolution, uniform weights).
            "ties" — TIES-style: keep only parameters where the sign is
                     consistent across the majority of vectors; zero others.
            "dare" — DARE-style: randomly drop values with p=0.5, then
                     rescale by 1/(1-p) to preserve expected magnitude.

    Returns:
        Merged task vector dict.
    """
    if not task_vectors:
        return {}

    if method == "mean":
        return add_task_vectors(task_vectors, weights=None)

    if method == "ties":
        n = len(task_vectors)
        all_keys: set = set()
        for tv in task_vectors:
            all_keys |= set(tv.keys())

        result: TaskVector = {}
        for k in all_keys:
            deltas = [tv[k] for tv in task_vectors if k in tv]
            if not deltas:
                continue
            # Stack: shape (n_present, *param_shape)
            stacked = torch.stack(deltas, dim=0)  # (n_p, ...)
            # Sign of sum determines majority sign
            sum_sign = torch.sign(stacked.sum(dim=0))  # (...,)
            # For each element, count how many vectors agree with majority sign
            # A vector "agrees" if sign(delta) == sum_sign (both non-zero)
            # Zero out entries where no majority agreement
            # TIES: keep param only when majority of vectors have the same sign
            signs = torch.sign(stacked)            # (n_p, ...)
            # majority threshold: more than half
            agreement = (signs == sum_sign.unsqueeze(0)).float().sum(dim=0)
            majority_mask = (agreement > (len(deltas) / 2)).float()
            # Mean over agreeing vectors, zeroed where no majority
            masked = stacked * (signs == sum_sign.unsqueeze(0)).float()
            n_agree = (signs == sum_sign.unsqueeze(0)).float().sum(dim=0).clamp(min=1)
            mean_val = masked.sum(dim=0) / n_agree
            result[k] = mean_val * majority_mask
        return result

    if method == "dare":
        # First compute the mean task vector, then apply DARE dropout
        mean_tv = add_task_vectors(task_vectors, weights=None)
        p = 0.5  # dropout probability
        result = {}
        for k, v in mean_tv.items():
            # Random binary mask: keep with probability (1 - p)
            mask = (torch.rand_like(v) > p).float()
            # Rescale to preserve expected magnitude
            result[k] = v * mask / (1.0 - p)
        return result

    raise ValueError(f"Unknown conflict resolution method: {method!r}. "
                     f"Choose from 'mean', 'ties', 'dare'.")


def task_vector_similarity(tv1: TaskVector, tv2: TaskVector) -> float:
    """Cosine similarity between two task vectors (flattened).

    Args:
        tv1, tv2: Task vector dicts sharing at least some keys.

    Returns:
        Scalar cosine similarity in [-1, 1].
    """
    shared = sorted(set(tv1.keys()) & set(tv2.keys()))
    if not shared:
        return 0.0

    flat1 = torch.cat([tv1[k].reshape(-1).float() for k in shared])
    flat2 = torch.cat([tv2[k].reshape(-1).float() for k in shared])

    dot = (flat1 * flat2).sum()
    norm1 = flat1.norm()
    norm2 = flat2.norm()

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return (dot / (norm1 * norm2)).item()


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """Configuration for TaskArithmeticMerger."""

    scaling: float = 1.0
    """λ applied to the merged (combined) task vector."""

    weights: Optional[List[float]] = None
    """Per-task weights; uniform if None."""

    conflict_resolution: str = "mean"
    """One of 'mean', 'ties', 'dare'."""

    dare_dropout: float = 0.5
    """Dropout probability for DARE conflict resolution."""


# ---------------------------------------------------------------------------
# High-level merger class
# ---------------------------------------------------------------------------

class TaskArithmeticMerger:
    """Merge multiple fine-tuned models into one via task arithmetic.

    Usage::

        merger = TaskArithmeticMerger(pretrained_model)
        merger.add_finetuned(ft_model_a)
        merger.add_finetuned(ft_model_b, weight=0.5)
        merged_model = merger.merge()
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        config: Optional[MergeConfig] = None,
    ) -> None:
        self._pretrained = pretrained_model
        self._pretrained_state: Dict[str, torch.Tensor] = {
            k: v.clone() for k, v in pretrained_model.state_dict().items()
        }
        self._config = config if config is not None else MergeConfig()
        self._task_vectors: List[TaskVector] = []
        self._task_weights: List[float] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_finetuned(self, finetuned_model: nn.Module, weight: float = 1.0) -> None:
        """Register a fine-tuned model; extract its task vector internally.

        Args:
            finetuned_model: A model with the same architecture as pretrained.
            weight:          Scalar weight for this task vector.
        """
        ft_state = finetuned_model.state_dict()
        tv = extract_task_vector(self._pretrained_state, ft_state)
        self._task_vectors.append(tv)
        self._task_weights.append(weight)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self) -> nn.Module:
        """Merge all registered task vectors and return a new model.

        Does NOT modify the original pretrained_model.

        Returns:
            Deep-copied pretrained model with merged weights loaded.
        """
        if not self._task_vectors:
            return copy.deepcopy(self._pretrained)

        cfg = self._config

        # Build effective per-task weights
        if cfg.weights is not None and len(cfg.weights) == len(self._task_vectors):
            eff_weights = [
                w * tw for w, tw in zip(cfg.weights, self._task_weights)
            ]
        else:
            eff_weights = self._task_weights

        # Normalize so they sum to 1 before conflict resolution
        total = sum(eff_weights)
        if total == 0:
            norm_weights = [1.0 / len(eff_weights)] * len(eff_weights)
        else:
            norm_weights = [w / total for w in eff_weights]

        # Apply scaled individual vectors before resolution
        scaled_tvs = [
            scale_task_vector(tv, w * total)
            for tv, w in zip(self._task_vectors, norm_weights)
        ]

        merged_tv = resolve_conflicts(scaled_tvs, method=cfg.conflict_resolution)
        merged_state = apply_task_vector(
            self._pretrained_state, merged_tv, scaling=cfg.scaling
        )

        new_model = copy.deepcopy(self._pretrained)
        new_model.load_state_dict(merged_state)
        return new_model

    def forget(self, finetuned_model: nn.Module, scaling: float = 1.0) -> nn.Module:
        """Return a model with a specific task forgotten (negated task vector).

        θ_result = θ_pre - scaling * τ_ft

        Args:
            finetuned_model: The fine-tuned model whose capability to forget.
            scaling:         Magnitude of negation.

        Returns:
            New model with capability removed.
        """
        ft_state = finetuned_model.state_dict()
        tv = extract_task_vector(self._pretrained_state, ft_state)
        neg_tv = negate_task_vector(tv)
        forgotten_state = apply_task_vector(
            self._pretrained_state, neg_tv, scaling=scaling
        )
        new_model = copy.deepcopy(self._pretrained)
        new_model.load_state_dict(forgotten_state)
        return new_model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_tasks(self) -> int:
        """Number of registered fine-tuned models."""
        return len(self._task_vectors)
