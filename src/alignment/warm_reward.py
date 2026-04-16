"""WARM: Weight-Averaged Reward Models (Weymar et al., 2024).

WARM averages the weights of multiple reward model checkpoints (trained with
different random seeds or from different checkpoints) into a single merged
model.  This eliminates inference-time ensemble cost while empirically reducing
reward hacking / overoptimization.

Key insight: weight averaging in parameter space achieves the benefits of
ensemble disagreement without the inference cost of running multiple models.

References:
    Weymar et al. (2024), "WARM: On the Benefits of Weight Averaged Reward
    Models"
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WARMConfig:
    """Configuration for WARM ensembling.

    Attributes:
        n_models:     Number of reward model checkpoints to merge.
        merge_method: One of ``'linear'``, ``'slerp'``, ``'dare'``.
        temperature:  Softmax temperature applied before scoring (unused in
                      the functional helpers, kept for downstream use).
        dare_density: Fraction of delta weights to keep in DARE merge.
    """

    n_models: int = 4
    merge_method: str = "linear"
    temperature: float = 1.0
    dare_density: float = 0.7


# ---------------------------------------------------------------------------
# Functional merge helpers
# ---------------------------------------------------------------------------


def linear_merge(
    state_dicts: List[Dict[str, Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Tensor]:
    """Weighted average of all state dicts.

    Args:
        state_dicts: List of K state dicts, all with the same keys/shapes.
        weights:     Optional list of K non-negative floats.  Equal weights
                     (1/K each) are used when *None*.

    Returns:
        A single merged state dict with the same keys and shapes.

    Raises:
        ValueError: If ``state_dicts`` is empty or ``weights`` has wrong length.
    """
    K = len(state_dicts)
    if K == 0:
        raise ValueError("state_dicts must be non-empty")

    if weights is None:
        weights = [1.0 / K] * K
    else:
        if len(weights) != K:
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(state_dicts)={K}"
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        weights = [w / total for w in weights]

    merged: Dict[str, Tensor] = {}
    for key in state_dicts[0]:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += w * sd[key].float()
        merged[key] = acc.to(state_dicts[0][key].dtype)

    return merged


def dare_merge(
    state_dicts: List[Dict[str, Tensor]],
    density: float = 0.7,
    seed: int = 42,
) -> Dict[str, Tensor]:
    """DARE: randomly drop (1-density) fraction of delta weights before merging.

    Algorithm per key:
        1. Compute base as the first state_dict (treated as the anchor/base model).
        2. For each subsequent state_dict compute delta = state_dict[key] - base[key].
        3. Sample a Bernoulli(density) mask; zero out dropped delta entries.
        4. Rescale kept deltas by 1/density so the expected magnitude is preserved.
        5. Average all (masked) deltas and add back to base.

    When ``density=1.0`` this is equivalent to :func:`linear_merge`.
    When ``density=0.0`` all deltas are dropped and the base is returned.

    Args:
        state_dicts: List of K state dicts (first is the base/anchor model).
        density:     Fraction of delta parameters to retain in [0, 1].
        seed:        Random seed for reproducibility.

    Returns:
        Merged state dict.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0, 1], got {density}")
    K = len(state_dicts)
    if K == 0:
        raise ValueError("state_dicts must be non-empty")
    if K == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    base_sd = state_dicts[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    merged: Dict[str, Tensor] = {}
    for key in base_sd:
        base = base_sd[key].float()
        acc_delta = torch.zeros_like(base)

        for sd in state_dicts[1:]:
            delta = sd[key].float() - base
            if density == 0.0:
                # All deltas dropped — contribute nothing.
                masked_delta = torch.zeros_like(delta)
            elif density == 1.0:
                masked_delta = delta
            else:
                mask = torch.bernoulli(
                    torch.full(delta.shape, density), generator=rng
                ).bool()
                # Rescale to keep expected magnitude
                masked_delta = torch.where(mask, delta / density, torch.zeros_like(delta))
            acc_delta += masked_delta

        # Average over (K-1) fine-tuned models
        avg_delta = acc_delta / (K - 1)
        merged[key] = (base + avg_delta).to(base_sd[key].dtype)

    return merged


# ---------------------------------------------------------------------------
# Functional scoring helpers
# ---------------------------------------------------------------------------


def compute_reward_margin(
    rewards: Tensor,
    chosen_idx: int,
    rejected_idx: int,
) -> tuple[Tensor, Tensor]:
    """Compute reward margin between chosen and rejected responses.

    Args:
        rewards:      ``(n_models, batch)`` reward predictions from each model.
        chosen_idx:   Batch index of the chosen (preferred) response.
        rejected_idx: Batch index of the rejected response.

    Returns:
        Tuple ``(mean_margin, std_margin)`` both scalar tensors.
        ``mean_margin`` is the mean margin across models;
        ``std_margin`` is the std across models.
    """
    # rewards: (n_models, batch)
    chosen_rewards = rewards[:, chosen_idx]    # (n_models,)
    rejected_rewards = rewards[:, rejected_idx]  # (n_models,)
    margin = chosen_rewards - rejected_rewards   # (n_models,)

    mean_margin = margin.mean()
    if margin.shape[0] > 1:
        std_margin = margin.std(correction=0)
    else:
        std_margin = torch.zeros_like(mean_margin)

    return mean_margin, std_margin


def warm_reward_score(
    models: List[nn.Module],
    input_ids: Tensor,
    aggregate: str = "mean",
) -> Tensor:
    """Run all models and aggregate reward scores.

    Each model in ``models`` is called with ``input_ids`` and expected to
    return a ``(batch,)`` scalar reward tensor.

    Args:
        models:    List of reward model nn.Modules.
        input_ids: ``(batch, seq_len)`` token ids (or any tensor the models
                   accept).
        aggregate: Aggregation strategy: ``'mean'``, ``'min'``, ``'max'``,
                   or ``'vote'``.

    Returns:
        ``(batch,)`` aggregated reward tensor.

    Raises:
        ValueError: If ``models`` is empty or ``aggregate`` is unknown.
    """
    if not models:
        raise ValueError("models list must be non-empty")

    all_rewards: List[Tensor] = []
    for model in models:
        with torch.no_grad():
            r = model(input_ids)
        # Allow models that return (mean, std) tuples (e.g. RewardEnsemble)
        if isinstance(r, (tuple, list)):
            r = r[0]
        r = r.squeeze(-1)  # ensure (batch,)
        all_rewards.append(r)

    stacked = torch.stack(all_rewards, dim=0)  # (n_models, batch)

    if aggregate == "mean":
        return stacked.mean(dim=0)
    elif aggregate == "min":
        return stacked.min(dim=0).values
    elif aggregate == "max":
        return stacked.max(dim=0).values
    elif aggregate == "vote":
        # Majority vote: 1 if majority of models give positive reward, else 0
        votes = (stacked > 0).float().mean(dim=0)
        return (votes >= 0.5).float()
    else:
        raise ValueError(
            f"Unknown aggregate '{aggregate}'. Choose from 'mean', 'min', 'max', 'vote'."
        )


# ---------------------------------------------------------------------------
# WARMEnsemble
# ---------------------------------------------------------------------------


class WARMEnsemble:
    """Weight-Averaged Reward Model ensemble.

    Stores a collection of checkpoint state dicts and merges them into a
    single set of weights using a chosen merge method.

    Args:
        base_model_factory: Callable that returns a fresh ``nn.Module`` with
                            the architecture to merge into.
        n_models:           Expected number of checkpoints (informational).
        merge_method:       One of ``'linear'``, ``'slerp'``, ``'dare'``.
    """

    def __init__(
        self,
        base_model_factory: Callable[[], nn.Module],
        n_models: int = 4,
        merge_method: str = "linear",
    ) -> None:
        self.base_model_factory = base_model_factory
        self.n_models = n_models
        self.merge_method = merge_method

        # List of (state_dict, weight) tuples
        self._checkpoints: List[tuple[Dict[str, Tensor], float]] = []

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def add_checkpoint(self, state_dict: Dict[str, Tensor], weight: float = 1.0) -> None:
        """Add a checkpoint state dict to the ensemble.

        Args:
            state_dict: Model state dict to add.
            weight:     Relative weight for this checkpoint in the merge.
                        Values are renormalized at merge time.
        """
        # Store a deep copy so subsequent mutations don't affect stored state
        self._checkpoints.append(
            ({k: v.clone() for k, v in state_dict.items()}, float(weight))
        )

    def merge(self) -> Dict[str, Tensor]:
        """Merge all stored checkpoints into one state dict.

        Returns:
            Merged state dict using :attr:`merge_method`.

        Raises:
            RuntimeError: If no checkpoints have been added.
        """
        if not self._checkpoints:
            raise RuntimeError("No checkpoints added. Call add_checkpoint() first.")

        state_dicts = [sd for sd, _ in self._checkpoints]
        weights = [w for _, w in self._checkpoints]

        if self.merge_method == "linear":
            return linear_merge(state_dicts, weights)
        elif self.merge_method == "slerp":
            # Delegate to WARP's SLERP merge if available; otherwise fall back
            # to linear (pure PyTorch, no HuggingFace dependency).
            try:
                from src.alignment.warp import merge_policies_slerp  # type: ignore
                return merge_policies_slerp(state_dicts, weights)
            except ImportError:
                return linear_merge(state_dicts, weights)
        elif self.merge_method == "dare":
            return dare_merge(state_dicts)
        else:
            raise ValueError(
                f"Unknown merge_method '{self.merge_method}'. "
                "Choose from 'linear', 'slerp', 'dare'."
            )

    def get_merged_model(self, model: nn.Module) -> nn.Module:
        """Apply merged weights to a model and return it.

        Args:
            model: An ``nn.Module`` whose architecture matches the stored
                   checkpoints.

        Returns:
            The same ``model`` instance with weights updated in-place.
        """
        merged_sd = self.merge()
        model.load_state_dict(merged_sd)
        return model

    # ------------------------------------------------------------------
    # Disagreement / uncertainty
    # ------------------------------------------------------------------

    def compute_disagreement(self, rewards: List[Tensor]) -> Tensor:
        """Compute per-sample variance across models.

        Args:
            rewards: List of length n_models, each a ``(batch,)`` reward
                     tensor predicted by the corresponding model.

        Returns:
            ``(batch,)`` variance tensor.  Higher values indicate greater
            disagreement between models.
        """
        if not rewards:
            raise ValueError("rewards list must be non-empty")

        stacked = torch.stack(rewards, dim=0)  # (n_models, batch)
        if stacked.shape[0] == 1:
            return torch.zeros(stacked.shape[1], dtype=stacked.dtype)

        variance = stacked.var(dim=0, correction=0)  # (batch,)
        return variance
