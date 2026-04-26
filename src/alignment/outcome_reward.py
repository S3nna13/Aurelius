"""Outcome-supervised reward model (ORM): predict scalar reward from complete response."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ORMConfig:
    """Configuration for the OutcomeRewardModel."""

    d_model: int = 512
    n_layers_head: int = 2  # reward head MLP layers
    dropout: float = 0.1
    reward_scale: float = 1.0  # scale output reward
    use_mean_pooling: bool = True  # pool over sequence or use last token
    n_ensemble: int = 1  # number of ensemble heads


class RewardHead(nn.Module):
    """MLP projecting d_model → 1 (scalar reward).

    Architecture: Linear → ReLU → (repeat n_layers-1 times) → Linear → scalar
    The final linear has no activation.
    """

    def __init__(self, d_model: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.append(nn.Linear(d_model, d_model))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                # Final layer: d_model → 1, no activation
                layers.append(nn.Linear(d_model, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, d_model)
        Returns:
            (B, 1) scalar reward per sample
        """
        return self.mlp(x)


class OutcomeRewardModel(nn.Module):
    """Outcome-supervised reward model wrapping a frozen backbone + trainable reward head(s).

    Distinct from RewardModel (last-token hidden states, single linear head) and
    ProcessRewardModel (per-step scores). This ORM:
      - Operates on logits from the backbone (not hidden states).
      - Supports mean-pooling or last-token pooling over the sequence.
      - Projects vocab_size → d_model via a trainable linear (no bias).
      - Supports an ensemble of reward heads for uncertainty estimation.
    """

    def __init__(self, backbone: nn.Module, config: ORMConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone

        vocab_size = backbone.config.vocab_size
        d_model = config.d_model

        # Project backbone logits (vocab_size) → d_model
        self.proj = nn.Linear(vocab_size, d_model, bias=False)

        # Ensemble of reward heads
        self.heads = nn.ModuleList(
            [
                RewardHead(d_model, config.n_layers_head, config.dropout)
                for _ in range(config.n_ensemble)
            ]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute outcome reward scores.

        Args:
            input_ids: (B, T) token indices

        Returns:
            (B, 1) reward scores
        """
        # Run backbone with frozen weights
        with torch.no_grad():
            _loss, logits, _pkv = self.backbone(input_ids)
        # logits: (B, T, vocab_size)

        # Pool over sequence dimension
        if self.config.use_mean_pooling:
            pooled = logits.mean(dim=1)  # (B, vocab_size)
        else:
            pooled = logits[:, -1, :]  # (B, vocab_size)  — last token

        # Project vocab_size → d_model
        features = self.proj(pooled)  # (B, d_model)

        # Run each head, stack → (B, n_ensemble), mean → (B, 1)
        head_outputs = [head(features) for head in self.heads]  # each (B, 1)
        stacked = torch.stack(head_outputs, dim=1).squeeze(-1)  # (B, n_ensemble)
        reward = stacked.mean(dim=1, keepdim=True)  # (B, 1)

        return reward * self.config.reward_scale


def reward_loss(predicted: torch.Tensor, chosen_mask: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise loss for outcome reward training.

    For each consecutive pair (chosen_i, rejected_i):
        loss = -log(sigmoid(r_chosen - r_rejected))

    Args:
        predicted:    (B,) reward scores
        chosen_mask:  (B,) bool — True for chosen responses, False for rejected

    Returns:
        Scalar loss
    """
    # Drop last sample if B is odd
    B = predicted.shape[0]
    if B % 2 == 1:
        predicted = predicted[:-1]
        chosen_mask = chosen_mask[:-1]

    chosen_rewards = predicted[chosen_mask]
    rejected_rewards = predicted[~chosen_mask]

    # Pair up: assume interleaved or equal counts; zip to shortest
    n_pairs = min(chosen_rewards.shape[0], rejected_rewards.shape[0])
    chosen_rewards = chosen_rewards[:n_pairs]
    rejected_rewards = rejected_rewards[:n_pairs]

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def calibrate_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """Z-score normalize rewards to mean=0, std=1.

    Args:
        rewards: (N,) raw reward scores

    Returns:
        (N,) calibrated rewards; zeros if std < 1e-8
    """
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std.item() < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - mean) / std


def ensemble_uncertainty(rewards: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute ensemble statistics for uncertainty estimation.

    Args:
        rewards: (B, n_ensemble) reward scores from multiple heads

    Returns:
        dict with:
            "mean":        (B,) mean reward across ensemble
            "std":         (B,) standard deviation across ensemble
            "disagreement": scalar — mean of per-sample std across batch
    """
    mean = rewards.mean(dim=1)  # (B,)
    std = rewards.std(dim=1, unbiased=False)  # (B,)
    disagreement = std.mean()  # scalar
    return {"mean": mean, "std": std, "disagreement": disagreement}
