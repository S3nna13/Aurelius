"""Reward model training with Bradley-Terry preference learning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RewardModelConfig:
    d_model: int = 64
    dropout: float = 0.1
    label_smoothing: float = 0.0
    margin: float = 0.0


class RewardHead(nn.Module):
    """Scalar reward head: projects last-token hidden state to a scalar reward."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: (B, T, d_model) -> reward: (B,)"""
        last = hidden[:, -1, :]
        last = self.dropout(last)
        return self.proj(last).squeeze(-1)


class RewardModel(nn.Module):
    """backbone_fn + RewardHead for preference learning.

    backbone_fn: Callable[[Tensor], Tensor]
        Takes token_ids (B, T) and returns hidden states (B, T, d_model).
    """

    def __init__(
        self,
        backbone_fn: Callable[[Tensor], Tensor],
        config: RewardModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone_fn = backbone_fn
        self.config = config if config is not None else RewardModelConfig()
        self.reward_head = RewardHead(self.config.d_model, self.config.dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids (B, T) -> scalar rewards (B,)."""
        out = self.backbone_fn(token_ids)
        # Handle models that return (loss, logits, ...) tuples
        hidden = out[1] if isinstance(out, tuple) else out
        return self.reward_head(hidden)

    @torch.no_grad()
    def get_reward(self, token_ids: Tensor) -> Tensor:
        """Alias for forward with no grad."""
        return self.forward(token_ids)


def bradley_terry_loss(
    rewards_chosen: Tensor,
    rewards_rejected: Tensor,
    margin: float = 0.0,
) -> Tensor:
    """Bradley-Terry pairwise ranking loss.

    loss = -mean(log(sigmoid(r_chosen - r_rejected - margin)))
    Returns a scalar tensor.
    """
    gap = rewards_chosen - rewards_rejected - margin
    return -F.logsigmoid(gap).mean()


def compute_reward_accuracy(
    rewards_chosen: Tensor,
    rewards_rejected: Tensor,
) -> float:
    """Fraction of pairs where chosen reward > rejected reward."""
    return (rewards_chosen > rewards_rejected).float().mean().item()


class RewardTrainer:
    """Trains a RewardModel on preference pairs using Bradley-Terry loss."""

    def __init__(
        self,
        model: RewardModel,
        optimizer: torch.optim.Optimizer,
        config: RewardModelConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else RewardModelConfig()

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """Forward both, compute BT loss, backward, step. Returns metrics dict."""
        self.model.train()
        self.optimizer.zero_grad()

        r_chosen = self.model(chosen_ids)
        r_rejected = self.model(rejected_ids)

        loss = bradley_terry_loss(r_chosen, r_rejected, margin=self.config.margin)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = compute_reward_accuracy(r_chosen.detach(), r_rejected.detach())
            mean_c = r_chosen.detach().mean().item()
            mean_r = r_rejected.detach().mean().item()

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "mean_chosen_reward": mean_c,
            "mean_rejected_reward": mean_r,
        }

    def evaluate(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """Evaluate without gradient. Same metrics as train_step."""
        self.model.eval()
        with torch.no_grad():
            r_chosen = self.model(chosen_ids)
            r_rejected = self.model(rejected_ids)
            loss = bradley_terry_loss(r_chosen, r_rejected, margin=self.config.margin)
            acc = compute_reward_accuracy(r_chosen, r_rejected)
            mean_c = r_chosen.mean().item()
            mean_r = r_rejected.mean().item()

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "mean_chosen_reward": mean_c,
            "mean_rejected_reward": mean_r,
        }


def normalize_rewards(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Z-score normalization: (r - mean) / (std + eps)."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)
