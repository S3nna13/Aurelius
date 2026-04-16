"""Reward model v2 — Bradley-Terry preference learning for RLHF.

Trains scalar reward models from pairwise human preference data.
Supports running-stat normalization and reward clipping for stable RL training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Configuration for RewardModel v2."""
    d_model: int = 512
    dropout: float = 0.0
    normalize_rewards: bool = True
    reward_clip: float = 5.0


# ---------------------------------------------------------------------------
# RewardHead
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Projects a hidden state to a scalar reward.

    Accepts either:
      - (B, T, d_model) — takes last token hidden[:, -1, :]
      - (B, d_model)    — uses directly
    Returns (B,) scalar rewards.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: (B, T, d_model) or (B, d_model) -> (B,)"""
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]      # last-token pooling
        hidden = self.drop(hidden)
        return self.proj(hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def compute_preference_loss(
    chosen_rewards: Tensor,
    rejected_rewards: Tensor,
) -> Tensor:
    """Bradley-Terry pairwise loss.

    loss = -mean(log sigmoid(chosen - rejected))
    Returns a scalar tensor.
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def compute_reward_accuracy(
    chosen_rewards: Tensor,
    rejected_rewards: Tensor,
) -> float:
    """Fraction of pairs where chosen_reward > rejected_reward."""
    return (chosen_rewards > rejected_rewards).float().mean().item()


def normalize_rewards(
    rewards: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Update running statistics and return zero-mean, unit-variance rewards.

    Uses exponential moving average of mean and variance.

    Args:
        rewards:      (N,) batch of raw rewards.
        running_mean: scalar running mean tensor (updated in-place style).
        running_var:  scalar running variance tensor (updated in-place style).
        momentum:     EMA momentum for stat update.

    Returns:
        (normalized_rewards, new_mean, new_var)
    """
    batch_mean = rewards.mean()
    batch_var = rewards.var(unbiased=False)

    new_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
    new_var  = (1.0 - momentum) * running_var  + momentum * batch_var

    # Normalize with updated stats
    std = (new_var + 1e-8).sqrt()
    normalized = (rewards - new_mean) / std

    return normalized, new_mean, new_var


def clip_rewards(rewards: Tensor, clip_value: float) -> Tensor:
    """Clamp rewards to [-clip_value, clip_value]."""
    return rewards.clamp(-clip_value, clip_value)


def compute_reward_stats(rewards: Tensor) -> Dict[str, float]:
    """Return descriptive statistics for a batch of rewards.

    Returns dict with keys: "mean", "std", "min", "max".
    """
    return {
        "mean": rewards.mean().item(),
        "std":  rewards.std().item(),
        "min":  rewards.min().item(),
        "max":  rewards.max().item(),
    }


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Full reward model: backbone + scalar RewardHead.

    Args:
        backbone_fn: Callable that maps input_ids (B, T) -> hidden_states (B, T, d_model).
        config:      RewardConfig instance.
    """

    def __init__(
        self,
        backbone_fn: Callable[[Tensor], Tensor],
        config: RewardConfig,
    ) -> None:
        super().__init__()
        self.backbone_fn = backbone_fn
        self.config = config
        self.head = RewardHead(config.d_model, config.dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids (B, T) -> scalar rewards (B,)."""
        hidden = self.backbone_fn(input_ids)   # (B, T, d_model)
        return self.head(hidden)               # (B,)

    def score_pair(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward both sequences and return (chosen_rewards, rejected_rewards).

        Both outputs are (B,).
        """
        chosen_rewards   = self.forward(chosen_ids)
        rejected_rewards = self.forward(rejected_ids)
        return chosen_rewards, rejected_rewards
