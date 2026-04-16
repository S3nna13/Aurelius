"""Reward Model Soup — weight-average multiple reward models to reduce variance/reward hacking.

Based on Rame et al. 2024: averaging model weights in parameter space produces a better model
than any single model while reducing reward hacking.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RewardSoupConfig:
    aggregation: str = "mean"           # "mean", "weighted_mean", "min", "max", "median"
    normalize_before_agg: bool = False  # z-score normalize each model's rewards first
    temperature: float = 1.0           # softmax temperature for computing ensemble weights
    weights: Optional[List[float]] = None  # per-model weights for weighted_mean


def weight_average_models(
    models: list[nn.Module],
    weights: list[float] | None = None,
) -> nn.Module:
    """Create a new model with weight-averaged parameters.

    theta_soup = sum(w_i * theta_i)  where w_i = 1/N if weights=None.
    Returns a deep copy of models[0] with averaged weights.
    """
    if not models:
        raise ValueError("models list must not be empty")

    n = len(models)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"len(weights)={len(weights)} must equal len(models)={n}")
        total = sum(weights)
        weights = [w / total for w in weights]

    # Deep copy the first model to get the averaged result
    soup = copy.deepcopy(models[0])

    # Zero out all parameters in the copy
    with torch.no_grad():
        for param in soup.parameters():
            param.zero_()

        # Weighted sum of all models' parameters
        for model, w in zip(models, weights):
            for p_soup, p_src in zip(soup.parameters(), model.parameters()):
                p_soup.add_(p_src * w)

    return soup


def _maybe_normalize(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Z-score normalize a (N,) tensor."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def aggregate_rewards(
    reward_lists: list[Tensor],  # list of (N,) tensors, one per model
    config: RewardSoupConfig,
) -> Tensor:
    """Aggregate reward predictions from multiple models.

    - "mean": element-wise mean
    - "weighted_mean": requires weights in config (use uniform if not set)
    - "min": element-wise minimum (conservative)
    - "max": element-wise maximum
    - "median": element-wise median
    Returns (N,) aggregated rewards.
    """
    if not reward_lists:
        raise ValueError("reward_lists must not be empty")

    # Optionally z-score normalize each model's rewards
    if config.normalize_before_agg:
        reward_lists = [_maybe_normalize(r) for r in reward_lists]

    # Stack to (M, N) where M = number of models
    stacked = torch.stack(reward_lists, dim=0)  # (M, N)

    agg = config.aggregation
    if agg == "mean":
        return stacked.mean(dim=0)
    elif agg == "weighted_mean":
        m = stacked.shape[0]
        if config.weights is not None:
            w = torch.tensor(config.weights, dtype=stacked.dtype, device=stacked.device)
            w = w / w.sum()
        else:
            w = torch.full((m,), 1.0 / m, dtype=stacked.dtype, device=stacked.device)
        # w: (M,) -> (M, 1) for broadcasting
        return (stacked * w.unsqueeze(1)).sum(dim=0)
    elif agg == "min":
        return stacked.min(dim=0).values
    elif agg == "max":
        return stacked.max(dim=0).values
    elif agg == "median":
        return stacked.median(dim=0).values
    else:
        raise ValueError(f"Unknown aggregation mode: {agg!r}")


class RewardSoup:
    """Ensemble of reward models with configurable aggregation."""

    def __init__(self, models: list[nn.Module], config: RewardSoupConfig | None = None):
        self.models = models
        self.config = config or RewardSoupConfig()
        self._weights: list[float] | None = None

    def set_weights(self, weights: list[float]) -> None:
        """Set model weights for weighted_mean aggregation."""
        if len(weights) != len(self.models):
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(models)={len(self.models)}"
            )
        self._weights = weights
        self.config.weights = weights

    @torch.no_grad()
    def score(self, token_ids: Tensor) -> Tensor:
        """Score input with all models, return aggregated reward (N,)."""
        reward_lists = []
        for model in self.models:
            model.eval()
            out = model(token_ids)
            # Handle (loss, logits, ...) tuple output from backbone
            if isinstance(out, tuple):
                rewards = out[0]
            else:
                rewards = out
            reward_lists.append(rewards)

        return aggregate_rewards(reward_lists, self.config)

    def calibrate_weights(
        self,
        val_inputs: Tensor,   # (N, T) validation inputs
        val_labels: Tensor,   # (N,) binary preference labels
    ) -> list[float]:
        """Compute per-model accuracy on validation set.

        Use softmax(accuracy / temperature) as model weights.
        Returns and sets the weights.
        """
        accuracies = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                out = model(val_inputs)
                if isinstance(out, tuple):
                    rewards = out[0]
                else:
                    rewards = out
                # Treat reward > 0 as positive prediction; compare to val_labels
                preds = (rewards > 0).float()
                acc = (preds == val_labels.float()).float().mean().item()
                accuracies.append(acc)

        # Softmax with temperature
        acc_tensor = torch.tensor(accuracies, dtype=torch.float32)
        logits = acc_tensor / max(self.config.temperature, 1e-8)
        weights_tensor = torch.softmax(logits, dim=0)
        weights = weights_tensor.tolist()

        self.set_weights(weights)
        return weights

    def distill_to_single(self) -> nn.Module:
        """Return a weight-averaged single model (model soup).

        Uses uniform averaging across all models.
        """
        return weight_average_models(self.models, weights=None)


def evaluate_reward_diversity(
    models: list[nn.Module],
    token_ids: Tensor,  # (N, T) test inputs
) -> dict[str, float]:
    """Measure disagreement between reward models.

    Returns: {'std_across_models': float, 'max_disagreement': float, 'mean_reward': float}
    """
    reward_lists = []
    with torch.no_grad():
        for model in models:
            model.eval()
            out = model(token_ids)
            if isinstance(out, tuple):
                rewards = out[0]
            else:
                rewards = out
            reward_lists.append(rewards)

    # stacked: (M, N) where M = num models, N = batch size
    stacked = torch.stack(reward_lists, dim=0)  # (M, N)

    # Per-sample std across models, then average
    std_per_sample = stacked.std(dim=0)           # (N,)
    std_across_models = std_per_sample.mean().item()

    # Max disagreement: max range across models for any single sample
    max_reward = stacked.max(dim=0).values        # (N,)
    min_reward = stacked.min(dim=0).values        # (N,)
    max_disagreement = (max_reward - min_reward).max().item()

    mean_reward = stacked.mean().item()

    return {
        "std_across_models": std_across_models,
        "max_disagreement": max_disagreement,
        "mean_reward": mean_reward,
    }
