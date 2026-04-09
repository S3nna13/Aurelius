"""Reward model training pipeline: preference pair training, ELO ranking, and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RMTrainerConfig:
    """Configuration for the RewardModelTrainer."""

    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0
    margin: float = 0.0        # minimum margin for preference loss
    use_elo: bool = False       # track ELO ratings for responses
    elo_k: float = 32.0        # ELO K-factor


# ---------------------------------------------------------------------------
# Loss and accuracy helpers
# ---------------------------------------------------------------------------

def preference_loss(
    chosen_rewards: Tensor,
    rejected_rewards: Tensor,
    margin: float = 0.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Bradley-Terry preference loss over (chosen, rejected) reward pairs.

    Args:
        chosen_rewards:   (B,) scalar rewards for preferred responses.
        rejected_rewards: (B,) scalar rewards for dispreferred responses.
        margin:           minimum margin; loss = -log(sigmoid(delta - margin)).
        label_smoothing:  mix with uniform 0.5 label: (1-ls)*loss + ls*0.5

    Returns:
        Scalar loss.
    """
    delta = chosen_rewards - rejected_rewards - margin
    loss = -F.logsigmoid(delta).mean()
    if label_smoothing > 0.0:
        loss = (1.0 - label_smoothing) * loss + label_smoothing * 0.5
    return loss


def compute_accuracy(chosen_rewards: Tensor, rejected_rewards: Tensor) -> float:
    """Fraction of pairs where chosen reward > rejected reward.

    Args:
        chosen_rewards:   (B,) rewards for preferred responses.
        rejected_rewards: (B,) rewards for dispreferred responses.

    Returns:
        Float in [0, 1].
    """
    correct = (chosen_rewards > rejected_rewards).float().mean()
    return correct.item()


# ---------------------------------------------------------------------------
# ELO Rating System
# ---------------------------------------------------------------------------

class ELORatingSystem:
    """Track ELO ratings for responses or models.

    Args:
        initial_rating: Starting ELO for any new id (default 1000.0).
        k:              K-factor controlling update magnitude (default 32.0).
    """

    def __init__(self, initial_rating: float = 1000.0, k: float = 32.0) -> None:
        self.initial_rating = initial_rating
        self.k = k
        self.ratings: dict[str, float] = {}

    def _ensure(self, response_id: str) -> None:
        """Initialize rating for id if not present."""
        if response_id not in self.ratings:
            self.ratings[response_id] = self.initial_rating

    def update(self, winner_id: str, loser_id: str) -> None:
        """Apply standard ELO update after a head-to-head comparison.

        Args:
            winner_id: id of the winning response/model.
            loser_id:  id of the losing response/model.
        """
        self._ensure(winner_id)
        self._ensure(loser_id)

        r_winner = self.ratings[winner_id]
        r_loser = self.ratings[loser_id]

        expected = 1.0 / (1.0 + 10.0 ** ((r_loser - r_winner) / 400.0))

        self.ratings[winner_id] = r_winner + self.k * (1.0 - expected)
        self.ratings[loser_id] = r_loser + self.k * (0.0 - (1.0 - expected))

    def get_rating(self, response_id: str) -> float:
        """Return the current ELO rating for id (initializes if missing).

        Args:
            response_id: response/model identifier.

        Returns:
            Current ELO rating as float.
        """
        self._ensure(response_id)
        return self.ratings[response_id]

    def get_ranking(self) -> list[tuple[str, float]]:
        """Return all ratings sorted descending by ELO.

        Returns:
            List of (id, rating) tuples, highest rating first.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# RewardModelTrainer
# ---------------------------------------------------------------------------

class RewardModelTrainer:
    """Train a reward model backbone on (chosen, rejected) preference pairs.

    The backbone is expected to return (loss, logits, past_key_values) where
    logits is (B, T, vocab_size).  This trainer mean-pools the logits over the
    sequence dimension and projects to a scalar reward via a learnable linear
    head (score_head).

    Args:
        reward_model: nn.Module backbone whose forward returns (_, logits, _).
        config:       RMTrainerConfig.
    """

    def __init__(self, reward_model: nn.Module, config: RMTrainerConfig) -> None:
        self.reward_model = reward_model
        self.config = config

        # Infer vocab_size from the backbone config
        vocab_size = reward_model.config.vocab_size

        # Scalar projection head: vocab_size -> 1
        self.score_head = nn.Linear(vocab_size, 1, bias=True)
        nn.init.normal_(self.score_head.weight, std=0.02)
        nn.init.zeros_(self.score_head.bias)

        # Optimizer over all trainable parameters (backbone + score_head)
        all_params = list(reward_model.parameters()) + list(self.score_head.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

        # Optional ELO tracker
        self.elo: ELORatingSystem | None = (
            ELORatingSystem(k=config.elo_k) if config.use_elo else None
        )

    def compute_rewards(self, input_ids: Tensor) -> Tensor:
        """Compute scalar rewards for a batch of token sequences.

        Runs the backbone, mean-pools logits over the time dimension, then
        applies score_head to yield a (B,) reward tensor.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            (B,) scalar rewards.
        """
        _loss, logits, _pkv = self.reward_model(input_ids)
        # logits: (B, T, vocab_size) -> mean over T -> (B, vocab_size)
        pooled = logits.mean(dim=1)
        # score_head: (B, vocab_size) -> (B, 1) -> squeeze -> (B,)
        rewards = self.score_head(pooled).squeeze(-1)
        return rewards

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """One gradient update step on a batch of preference pairs.

        Args:
            chosen_ids:   (B, T) token ids for preferred responses.
            rejected_ids: (B, T) token ids for dispreferred responses.

        Returns:
            Dict with keys: "loss", "accuracy", "chosen_reward", "rejected_reward".
        """
        self.reward_model.train()
        self.score_head.train()
        self.optimizer.zero_grad()

        chosen_rewards = self.compute_rewards(chosen_ids)
        rejected_rewards = self.compute_rewards(rejected_ids)

        loss = preference_loss(
            chosen_rewards,
            rejected_rewards,
            margin=self.config.margin,
            label_smoothing=self.config.label_smoothing,
        )
        loss.backward()

        # Gradient clipping
        all_params = list(self.reward_model.parameters()) + list(self.score_head.parameters())
        nn.utils.clip_grad_norm_(all_params, self.config.gradient_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": compute_accuracy(chosen_rewards.detach(), rejected_rewards.detach()),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

    def evaluate(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """Evaluate on a batch of preference pairs without gradient updates.

        Args:
            chosen_ids:   (B, T) token ids for preferred responses.
            rejected_ids: (B, T) token ids for dispreferred responses.

        Returns:
            Dict with keys: "accuracy", "reward_margin".
        """
        self.reward_model.eval()
        self.score_head.eval()
        with torch.no_grad():
            chosen_rewards = self.compute_rewards(chosen_ids)
            rejected_rewards = self.compute_rewards(rejected_ids)

        accuracy = compute_accuracy(chosen_rewards, rejected_rewards)
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()

        return {
            "accuracy": accuracy,
            "reward_margin": reward_margin,
        }


# ---------------------------------------------------------------------------
# Convenience training loop
# ---------------------------------------------------------------------------

def train_reward_model(
    reward_model: nn.Module,
    preference_data: list[tuple[Tensor, Tensor]],
    config: RMTrainerConfig,
    n_epochs: int = 1,
) -> dict[str, list[float]]:
    """Train a reward model over multiple epochs of preference pairs.

    Args:
        reward_model:     nn.Module backbone.
        preference_data:  List of (chosen_ids, rejected_ids) tensor pairs.
        config:           RMTrainerConfig.
        n_epochs:         Number of full passes over preference_data.

    Returns:
        Dict with "losses" and "accuracies" lists (one entry per batch per epoch).
    """
    trainer = RewardModelTrainer(reward_model, config)
    losses: list[float] = []
    accuracies: list[float] = []

    for _ in range(n_epochs):
        for chosen_ids, rejected_ids in preference_data:
            metrics = trainer.train_step(chosen_ids, rejected_ids)
            losses.append(metrics["loss"])
            accuracies.append(metrics["accuracy"])

    return {"losses": losses, "accuracies": accuracies}
