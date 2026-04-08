"""Reward model ensemble for robust RLHF (uncertainty-weighted scoring)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EnsembleConfig:
    """Configuration for EnsembleRewardModel."""

    n_members: int = 4
    aggregation: str = "mean"           # "mean" | "min" | "uncertainty_weighted"
    uncertainty_threshold: float = 0.5  # flag high-uncertainty samples above this std


class RewardHead(nn.Module):
    """Scalar reward head on top of a transformer's last hidden state.

    Args:
        d_model:  Input feature dimension.
        dropout:  Dropout probability (default 0.1).
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute scalar reward from hidden states.

        Args:
            hidden_states: ``(B, T, D)`` transformer hidden states.

        Returns:
            ``(B,)`` scalar reward per sample.
        """
        x = hidden_states[:, -1, :]        # take last token: (B, D)
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)               # (B,)


class EnsembleRewardModel(nn.Module):
    """Parameter-efficient reward ensemble sharing a single backbone.

    All ``n_members`` reward heads process the same backbone output, making
    this an ensemble via diverse head initialisation rather than multiple
    full models.

    Args:
        base_model:  Shared AureliusTransformer backbone.
        config:      Ensemble hyperparameters.
        d_model:     Hidden dimension used to size the reward heads.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: EnsembleConfig,
        d_model: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        # Obtain vocab_size from the backbone config so we can project logits
        vocab_size: int = base_model.config.vocab_size

        # Map logits (B, T, vocab_size) -> pseudo-hidden-states (B, T, d_model)
        self.hidden_extractor = nn.Linear(vocab_size, d_model)

        # One reward head per ensemble member (all share the backbone)
        self.reward_heads = nn.ModuleList(
            [RewardHead(d_model) for _ in range(config.n_members)]
        )

    def _get_pseudo_hidden(self, input_ids: Tensor) -> Tensor:
        """Run backbone once and project logits to pseudo-hidden-states.

        Returns:
            ``(B, T, d_model)``
        """
        with torch.no_grad():
            _, logits, _ = self.base_model(input_ids)   # logits: (B, T, V)
        # Project through a learnable linear to get pseudo-hidden-states
        return self.hidden_extractor(logits)             # (B, T, d_model)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Score a batch of sequences.

        Args:
            input_ids: ``(B, T)`` token ids.

        Returns:
            Tuple ``(rewards, uncertainty)``:
                - ``rewards``:     ``(B,)`` aggregated reward per sample.
                - ``uncertainty``: ``(B,)`` std across ensemble members.
        """
        pseudo_hidden = self._get_pseudo_hidden(input_ids)   # (B, T, d_model)

        all_rewards = torch.stack(
            [head(pseudo_hidden) for head in self.reward_heads], dim=1
        )  # (B, n_members)

        rewards, uncertainty = aggregate_rewards(all_rewards, self.config)
        return rewards, uncertainty

    def get_all_rewards(self, input_ids: Tensor) -> Tensor:
        """Return raw rewards from every head without aggregation.

        Returns:
            ``(B, n_members)``
        """
        pseudo_hidden = self._get_pseudo_hidden(input_ids)   # (B, T, d_model)
        return torch.stack(
            [head(pseudo_hidden) for head in self.reward_heads], dim=1
        )  # (B, n_members)


def aggregate_rewards(
    member_rewards: Tensor,
    config: EnsembleConfig,
) -> tuple[Tensor, Tensor]:
    """Aggregate per-member rewards into a single score with uncertainty.

    Args:
        member_rewards: ``(B, n_members)`` raw rewards from each head.
        config:         Ensemble config (selects aggregation strategy).

    Returns:
        ``(aggregated, uncertainty)`` both shape ``(B,)``.
    """
    std = member_rewards.std(dim=1, correction=0) if member_rewards.shape[1] > 1 \
        else torch.zeros(member_rewards.shape[0], device=member_rewards.device)

    mode = config.aggregation

    if mode == "mean":
        aggregated = member_rewards.mean(dim=1)

    elif mode == "min":
        aggregated = member_rewards.min(dim=1).values

    elif mode == "uncertainty_weighted":
        eps = 1e-8
        # Weight by inverse of each member's deviation from the ensemble mean
        # Simpler approach: weight each member equally but down-weight by std
        # Per-sample std is shared across members, so use 1/(std+eps) to
        # scale the mean: high uncertainty -> lower effective reward.
        # Actually compute per-sample weights over members using member deviations:
        # w_k = 1 / (|r_k - mean| + eps), normalised per sample.
        mean_over_members = member_rewards.mean(dim=1, keepdim=True)         # (B, 1)
        deviations = (member_rewards - mean_over_members).abs() + eps        # (B, K)
        weights = 1.0 / deviations                                           # (B, K)
        weights = weights / weights.sum(dim=1, keepdim=True)                 # normalise
        aggregated = (weights * member_rewards).sum(dim=1)                   # (B,)

    else:
        raise ValueError(
            f"Unknown aggregation '{mode}'. "
            "Choose from 'mean', 'min', 'uncertainty_weighted'."
        )

    return aggregated, std


def detect_ood_samples(uncertainty: Tensor, threshold: float) -> Tensor:
    """Return a boolean mask of out-of-distribution (high-uncertainty) samples.

    Args:
        uncertainty: ``(B,)`` uncertainty estimates (e.g. std across members).
        threshold:   Samples with uncertainty above this value are flagged.

    Returns:
        ``(B,)`` bool tensor — ``True`` where uncertainty > threshold.
    """
    return uncertainty > threshold


class EnsembleTrainer:
    """Train an EnsembleRewardModel on preference pairs.

    Args:
        ensemble:   The :class:`EnsembleRewardModel` to train.
        optimizer:  Any ``torch.optim.Optimizer`` wrapping the model's parameters.
        config:     Ensemble configuration.
    """

    def __init__(
        self,
        ensemble: EnsembleRewardModel,
        optimizer: torch.optim.Optimizer,
        config: EnsembleConfig,
    ) -> None:
        self.ensemble = ensemble
        self.optimizer = optimizer
        self.config = config

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict[str, float]:
        """One gradient update on a preference pair batch.

        For each head the Bradley-Terry loss is:
            loss_k = -log(sigmoid(r_chosen_k - r_rejected_k))

        The total loss is the mean over all heads.

        Args:
            chosen_ids:   ``(B, T)`` token ids of preferred responses.
            rejected_ids: ``(B, T)`` token ids of rejected responses.

        Returns:
            Dict with keys ``"loss"``, ``"mean_reward_gap"``,
            ``"mean_uncertainty"``.
        """
        self.ensemble.train()
        self.optimizer.zero_grad()

        chosen_all = self.ensemble.get_all_rewards(chosen_ids)    # (B, n_members)
        rejected_all = self.ensemble.get_all_rewards(rejected_ids) # (B, n_members)

        # Per-head Bradley-Terry loss, averaged over heads and batch
        gap = chosen_all - rejected_all                            # (B, n_members)
        loss_per_head = -F.logsigmoid(gap)                        # (B, n_members)
        loss = loss_per_head.mean()

        loss.backward()
        self.optimizer.step()

        # Diagnostics
        with torch.no_grad():
            mean_gap = gap.mean().item()
            _, uncertainty = self.ensemble(chosen_ids)
            mean_uncertainty = uncertainty.mean().item()

        return {
            "loss": loss.item(),
            "mean_reward_gap": mean_gap,
            "mean_uncertainty": mean_uncertainty,
        }
