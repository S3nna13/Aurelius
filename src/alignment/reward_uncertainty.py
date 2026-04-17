"""Uncertainty-Aware Reward Modeling with MC-Dropout and Deep Ensembles.

Provides two orthogonal uncertainty estimation approaches:
- MC-Dropout: single model, multiple stochastic forward passes
- Deep Ensemble: K independent models, aggregate predictions

Also includes UncertaintyFilter for filtering high-uncertainty samples and
RewardUncertaintyTrainer for Bradley-Terry preference training.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# MCDropoutReward
# ---------------------------------------------------------------------------


class MCDropoutReward(nn.Module):
    """Reward model with MC-Dropout uncertainty estimation.

    Architecture: Linear(d_model, 128) -> ReLU -> Dropout(p) -> Linear(128, 1)

    Args:
        d_model:   Input feature dimension.
        dropout_p: Dropout probability (default 0.1).
    """

    def __init__(self, d_model: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute reward scores.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            ``(B,)`` reward scores.
        """
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h.squeeze(-1)

    def predict_with_uncertainty(
        self, x: Tensor, n_samples: int = 20
    ) -> Tuple[Tensor, Tensor]:
        """Estimate reward and uncertainty via MC-Dropout.

        Enables dropout at inference by temporarily setting the model to train
        mode, runs n_samples forward passes, and returns mean and std.

        Args:
            x:         ``(B, d_model)`` input features.
            n_samples: Number of stochastic forward passes.

        Returns:
            Tuple (mean_reward, std_reward) both shape (B,).
        """
        was_training = self.training
        self.train()

        with torch.no_grad():
            samples = torch.stack(
                [self.forward(x) for _ in range(n_samples)], dim=0
            )

        if not was_training:
            self.eval()

        mean_reward = samples.mean(dim=0)
        std_reward = samples.std(dim=0, correction=0)
        return mean_reward, std_reward


# ---------------------------------------------------------------------------
# DeepEnsembleReward
# ---------------------------------------------------------------------------


class DeepEnsembleReward:
    """Ensemble of K independent reward models.

    Args:
        models: List of nn.Module reward models. Each must accept
                (B, d_model) tensors and return (B,) rewards.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        if not models:
            raise ValueError("models list must be non-empty")
        self.models = list(models)

    def forward(self, x: Tensor) -> Tensor:
        """Return mean reward across ensemble members.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            ``(B,)`` mean reward.
        """
        rewards = torch.stack([m(x) for m in self.models], dim=0)
        return rewards.mean(dim=0)

    def predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return mean and std across ensemble members.

        Args:
            x: ``(B, d_model)`` input features.

        Returns:
            Tuple (mean_reward, std_reward) both shape (B,).
        """
        rewards = torch.stack([m(x) for m in self.models], dim=0)
        mean_reward = rewards.mean(dim=0)
        if rewards.shape[0] > 1:
            std_reward = rewards.std(dim=0, correction=0)
        else:
            std_reward = torch.zeros_like(mean_reward)
        return mean_reward, std_reward

    def update_member(self, idx: int, model: nn.Module) -> None:
        """Replace ensemble member at index idx.

        Args:
            idx:   Index of the model to replace (0-based).
            model: New nn.Module to insert.
        """
        if idx < 0 or idx >= len(self.models):
            raise IndexError(
                f"Index {idx} out of range for ensemble of size {len(self.models)}"
            )
        self.models[idx] = model


# ---------------------------------------------------------------------------
# UncertaintyFilter
# ---------------------------------------------------------------------------


class UncertaintyFilter:
    """Filter samples whose uncertainty exceeds a threshold.

    Args:
        threshold: Maximum allowed uncertainty. Samples with
                   uncertainty <= threshold are kept.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def filter(
        self, rewards: Tensor, uncertainties: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Keep samples with low uncertainty.

        Args:
            rewards:       ``(N,)`` reward values.
            uncertainties: ``(N,)`` uncertainty estimates.

        Returns:
            Tuple (kept_rewards, kept_mask) where kept_mask is bool (N,).
        """
        kept_mask = uncertainties <= self.threshold
        kept_rewards = rewards[kept_mask]
        return kept_rewards, kept_mask

    def calibrate_threshold(
        self, uncertainties: Tensor, percentile: float = 90
    ) -> float:
        """Set and return the p-th percentile of uncertainties as threshold.

        Args:
            uncertainties: ``(N,)`` uncertainty values.
            percentile:    Percentile to use as threshold (default 90).

        Returns:
            The p-th percentile value as a Python float.
        """
        if uncertainties.numel() == 0:
            return 0.0
        k = max(1, int(uncertainties.numel() * percentile / 100.0))
        k = min(k, uncertainties.numel())
        sorted_u, _ = torch.sort(uncertainties.flatten())
        threshold = sorted_u[k - 1].item()
        self.threshold = threshold
        return float(threshold)


# ---------------------------------------------------------------------------
# RewardUncertaintyTrainer
# ---------------------------------------------------------------------------


class RewardUncertaintyTrainer:
    """Train a reward model with Bradley-Terry preference loss.

    Args:
        model:     Reward model (nn.Module) accepting (B, d_model) -> (B,).
        optimizer: Any torch.optim.Optimizer.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def train_step(
        self, x_w: Tensor, x_l: Tensor
    ) -> Dict[str, float]:
        """One gradient update on a preference pair batch.

        Bradley-Terry loss: -logsigmoid(r_w - r_l)

        Args:
            x_w: ``(B, d_model)`` preferred (winner) features.
            x_l: ``(B, d_model)`` rejected (loser) features.

        Returns:
            Dict with keys 'loss', 'reward_margin', 'mean_reward_w', 'mean_reward_l'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        r_w = self.model(x_w)
        r_l = self.model(x_l)

        loss = -F.logsigmoid(r_w - r_l).mean()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            reward_margin = (r_w - r_l).mean().item()
            mean_reward_w = r_w.mean().item()
            mean_reward_l = r_l.mean().item()

        return {
            "loss": loss.item(),
            "reward_margin": reward_margin,
            "mean_reward_w": mean_reward_w,
            "mean_reward_l": mean_reward_l,
        }
