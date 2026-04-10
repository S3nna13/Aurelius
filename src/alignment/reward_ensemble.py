"""Reward model ensemble for uncertainty-aware RLHF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Configuration for the reward ensemble."""

    n_models: int = 5
    aggregation: str = "mean"       # "mean" | "min" | "ucb"
    ucb_beta: float = 1.0
    dropout_rate: float = 0.1       # for MC dropout baseline
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# RewardHead
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Scalar reward head: Linear -> GELU -> Dropout -> Linear -> scalar.

    Args:
        d_model:  Input feature dimension.
        dropout:  Dropout probability (default 0.1).
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute scalar reward from hidden states.

        Args:
            hidden_states: ``(B, T, D)`` transformer hidden states.

        Returns:
            ``(B,)`` scalar reward per sample.
        """
        x = hidden_states[:, -1, :]   # pool last token: (B, D)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)          # (B,)


# ---------------------------------------------------------------------------
# RewardEnsemble
# ---------------------------------------------------------------------------

class RewardEnsemble(nn.Module):
    """Ensemble of reward models sharing one backbone.

    The backbone is run once; n_models separate RewardHeads each produce an
    independent reward estimate, enabling uncertainty quantification.

    Args:
        base_model:  Shared AureliusTransformer backbone.
        config:      Ensemble hyperparameters.
    """

    def __init__(self, base_model: nn.Module, config: EnsembleConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        d_model: int = base_model.config.d_model
        self.reward_heads = nn.ModuleList(
            [RewardHead(d_model, config.dropout_rate) for _ in range(config.n_models)]
        )

    def _get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Run the backbone once and capture final hidden states via a hook.

        Returns:
            ``(B, T, D)`` hidden states from the final transformer layer.
        """
        captured: list[Tensor] = []

        def hook(module: nn.Module, inp: tuple, out: Tensor) -> None:
            captured.append(out if self.training else out.detach())

        # Hook onto the final norm which produces hidden states fed to lm_head
        handle = self.base_model.norm.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(self.training):
                self.base_model(input_ids)
        finally:
            handle.remove()

        return captured[0]  # (B, T, D)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Score a batch of sequences.

        Args:
            input_ids: ``(B, T)`` token ids.

        Returns:
            Tuple ``(mean_reward, std_reward)`` both shape ``(B,)``.
        """
        hidden = self._get_hidden_states(input_ids)  # (B, T, D)

        per_head = torch.stack(
            [head(hidden) for head in self.reward_heads], dim=0
        )  # (n_models, B)

        mean_reward = per_head.mean(dim=0)   # (B,)
        if per_head.shape[0] > 1:
            std_reward = per_head.std(dim=0, correction=0)
        else:
            std_reward = torch.zeros_like(mean_reward)

        return mean_reward, std_reward

    def aggregate(self, rewards: Tensor) -> Tensor:
        """Aggregate per-model rewards into a single score.

        Args:
            rewards: ``(n_models, B)`` raw rewards from each head.

        Returns:
            ``(B,)`` aggregated reward.
        """
        mode = self.config.aggregation
        if mode == "mean":
            return rewards.mean(dim=0)
        elif mode == "min":
            return rewards.min(dim=0).values
        elif mode == "ucb":
            mean = rewards.mean(dim=0)
            if rewards.shape[0] > 1:
                std = rewards.std(dim=0, correction=0)
            else:
                std = torch.zeros_like(mean)
            return mean + self.config.ucb_beta * std
        else:
            raise ValueError(
                f"Unknown aggregation '{mode}'. Choose from 'mean', 'min', 'ucb'."
            )

    def uncertainty(self, input_ids: Tensor) -> dict[str, Tensor]:
        """Return uncertainty estimates for a batch.

        Args:
            input_ids: ``(B, T)`` token ids.

        Returns:
            Dict with keys: ``"mean"``, ``"std"``, ``"epistemic"``,
            ``"lower_bound"``, all shape ``(B,)``.
        """
        hidden = self._get_hidden_states(input_ids)  # (B, T, D)

        per_head = torch.stack(
            [head(hidden) for head in self.reward_heads], dim=0
        )  # (n_models, B)

        mean = per_head.mean(dim=0)
        if per_head.shape[0] > 1:
            std = per_head.std(dim=0, correction=0)
        else:
            std = torch.zeros_like(mean)
        epistemic = std
        lower_bound = mean - std

        return {
            "mean": mean,
            "std": std,
            "epistemic": epistemic,
            "lower_bound": lower_bound,
        }


# ---------------------------------------------------------------------------
# MCDropoutReward
# ---------------------------------------------------------------------------

class MCDropoutReward:
    """MC-Dropout baseline: single head run multiple times in train mode.

    Args:
        reward_head:  A :class:`RewardHead` instance.
        n_forward:    Number of stochastic forward passes (default 10).
    """

    def __init__(self, reward_head: RewardHead, n_forward: int = 10) -> None:
        self.reward_head = reward_head
        self.n_forward = n_forward

    def predict(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Run reward_head n_forward times with dropout enabled.

        Args:
            hidden: ``(B, T, D)`` hidden states.

        Returns:
            Tuple ``(mean, std)`` both shape ``(B,)``.
        """
        self.reward_head.train()   # enable dropout
        samples = torch.stack(
            [self.reward_head(hidden) for _ in range(self.n_forward)], dim=0
        )  # (n_forward, B)

        mean = samples.mean(dim=0)
        if self.n_forward > 1:
            std = samples.std(dim=0, correction=0)
        else:
            std = torch.zeros_like(mean)
        return mean, std


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def conservative_reward(mean: Tensor, std: Tensor, beta: float = 1.0) -> Tensor:
    """Lower confidence bound: mean - beta * std.

    Args:
        mean: ``(B,)`` mean reward estimates.
        std:  ``(B,)`` reward std estimates.
        beta: Penalty strength (default 1.0).

    Returns:
        ``(B,)`` conservative reward.
    """
    return mean - beta * std


# ---------------------------------------------------------------------------
# RewardEnsembleTrainer
# ---------------------------------------------------------------------------

class RewardEnsembleTrainer:
    """Train a RewardEnsemble on preference pairs with Bradley-Terry loss.

    Args:
        ensemble:   The :class:`RewardEnsemble` to train.
        optimizer:  Any ``torch.optim.Optimizer``.
        config:     Ensemble configuration.
    """

    def __init__(
        self,
        ensemble: RewardEnsemble,
        optimizer: torch.optim.Optimizer,
        config: EnsembleConfig,
    ) -> None:
        self.ensemble = ensemble
        self.optimizer = optimizer
        self.config = config

    def _get_per_head_rewards(self, input_ids: Tensor) -> Tensor:
        """Return per-head rewards without aggregation.

        Returns:
            ``(n_models, B)``
        """
        hidden = self.ensemble._get_hidden_states(input_ids)
        return torch.stack(
            [head(hidden) for head in self.ensemble.reward_heads], dim=0
        )

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """One gradient update on a preference pair batch.

        Bradley-Terry loss per head:
            loss_k = -log(sigmoid(r_chosen_k - r_rejected_k))

        Total loss is the mean over all heads.

        Args:
            chosen_ids:   ``(B, T)`` preferred response token ids.
            rejected_ids: ``(B, T)`` rejected response token ids.

        Returns:
            Dict with keys ``"loss"``, ``"mean_margin"``, ``"agreement"``.
        """
        self.ensemble.train()
        self.optimizer.zero_grad()

        chosen_rewards = self._get_per_head_rewards(chosen_ids)     # (n_models, B)
        rejected_rewards = self._get_per_head_rewards(rejected_ids)  # (n_models, B)

        margin = chosen_rewards - rejected_rewards                   # (n_models, B)
        loss = -F.logsigmoid(margin).mean()

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            mean_margin = margin.mean().item()
            agreement = (margin > 0).float().mean().item()

        return {
            "loss": loss.item(),
            "mean_margin": mean_margin,
            "agreement": agreement,
        }

    def evaluate(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """Evaluate on a preference pair batch without gradient updates.

        Args:
            chosen_ids:   ``(B, T)`` preferred response token ids.
            rejected_ids: ``(B, T)`` rejected response token ids.

        Returns:
            Dict with keys ``"loss"``, ``"mean_margin"``, ``"agreement"``.
        """
        self.ensemble.eval()

        with torch.no_grad():
            chosen_rewards = self._get_per_head_rewards(chosen_ids)
            rejected_rewards = self._get_per_head_rewards(rejected_ids)

            margin = chosen_rewards - rejected_rewards
            loss = -F.logsigmoid(margin).mean()
            mean_margin = margin.mean().item()
            agreement = (margin > 0).float().mean().item()

        return {
            "loss": loss.item(),
            "mean_margin": mean_margin,
            "agreement": agreement,
        }
