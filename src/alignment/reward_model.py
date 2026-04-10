"""Reward model training with Bradley-Terry preference learning."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RewardModelConfig:
    d_model: int = 64
    dropout: float = 0.1
    margin: float = 0.0          # preference margin (loss = -log σ(r_w - r_l - margin))
    label_smoothing: float = 0.0


class RewardHead(nn.Module):
    """Scalar reward head on top of a backbone model.

    Takes the last-token hidden state and projects to a scalar reward.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: (B, T, d_model) -> reward: (B,) scalar per sequence."""
        # Take the last token's hidden state
        last = hidden[:, -1, :]        # (B, d_model)
        last = self.dropout(last)
        return self.proj(last).squeeze(-1)  # (B,)


class RewardModel(nn.Module):
    """Backbone + RewardHead for preference learning."""

    def __init__(self, backbone: nn.Module, config: RewardModelConfig | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config or RewardModelConfig(d_model=backbone.config.d_model)
        self.reward_head = RewardHead(self.config.d_model, self.config.dropout)

    def _hidden_states(self, input_ids: Tensor) -> Tensor:
        """Run backbone manually to extract hidden states before lm_head.

        Returns (B, T, d_model).
        """
        B, S = input_ids.shape
        x = self.backbone.embed(input_ids)
        freqs_cis = self.backbone.freqs_cis[:S]
        for layer in self.backbone.layers:
            x, _ = layer(x, freqs_cis, mask=None, past_kv=None)
        x = self.backbone.norm(x)   # (B, T, d_model)
        return x

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run backbone, extract last-token hidden state, compute scalar reward.

        Returns (B,) reward scores.
        """
        hidden = self._hidden_states(input_ids)   # (B, T, d_model)
        return self.reward_head(hidden)            # (B,)

    def score(self, input_ids: Tensor) -> float:
        """Score a single sequence. Returns a Python float."""
        self.eval()
        with torch.no_grad():
            scores = self.forward(input_ids)
        return scores[0].item()


def bradley_terry_loss(
    reward_chosen: Tensor,    # (B,)
    reward_rejected: Tensor,  # (B,)
    margin: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict]:
    """Bradley-Terry preference loss.

    loss = -mean(log σ(r_chosen - r_rejected - margin))
    With label smoothing: loss = -(1-ε)*log σ(gap) - ε*log σ(-gap)

    Returns (loss, metrics) where metrics has:
        "accuracy": float — fraction where r_chosen > r_rejected
        "mean_gap": float — mean(r_chosen - r_rejected)
        "mean_chosen_reward": float
        "mean_rejected_reward": float
    """
    gap = reward_chosen - reward_rejected - margin   # (B,)

    if label_smoothing > 0.0:
        eps = label_smoothing
        loss = -(
            (1.0 - eps) * F.logsigmoid(gap)
            + eps * F.logsigmoid(-gap)
        ).mean()
    else:
        loss = -F.logsigmoid(gap).mean()

    with torch.no_grad():
        accuracy = (reward_chosen > reward_rejected).float().mean().item()
        mean_gap = (reward_chosen - reward_rejected).mean().item()
        mean_chosen = reward_chosen.mean().item()
        mean_rejected = reward_rejected.mean().item()

    metrics = {
        "accuracy": accuracy,
        "mean_gap": mean_gap,
        "mean_chosen_reward": mean_chosen,
        "mean_rejected_reward": mean_rejected,
    }
    return loss, metrics


class RewardModelTrainer:
    """Trains a RewardModel on preference pairs."""

    def __init__(
        self,
        reward_model: RewardModel,
        config: RewardModelConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.reward_model = reward_model
        self.config = config or reward_model.config
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                [p for p in reward_model.parameters() if p.requires_grad],
                lr=1e-4,
            )
        else:
            self.optimizer = optimizer

    def train_step(
        self,
        chosen_ids: Tensor,    # (B, T) — preferred responses
        rejected_ids: Tensor,  # (B, T) — dispreferred responses
    ) -> dict:
        """Compute Bradley-Terry loss and update.

        Returns metrics dict with: loss, accuracy, mean_gap,
        mean_chosen_reward, mean_rejected_reward.
        """
        self.reward_model.train()
        self.optimizer.zero_grad()

        r_chosen = self.reward_model(chosen_ids)
        r_rejected = self.reward_model(rejected_ids)

        loss, metrics = bradley_terry_loss(
            r_chosen, r_rejected,
            margin=self.config.margin,
            label_smoothing=self.config.label_smoothing,
        )
        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.item()
        return metrics

    def evaluate(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> dict:
        """Evaluate without gradient. Same metrics as train_step."""
        self.reward_model.eval()
        with torch.no_grad():
            r_chosen = self.reward_model(chosen_ids)
            r_rejected = self.reward_model(rejected_ids)
            loss, metrics = bradley_terry_loss(
                r_chosen, r_rejected,
                margin=self.config.margin,
                label_smoothing=self.config.label_smoothing,
            )
        metrics["loss"] = loss.item()
        return metrics


class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation."""

    def __init__(self, models: list[RewardModel]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (mean_reward (B,), std_reward (B,)) across ensemble."""
        rewards = torch.stack(
            [m(input_ids) for m in self.models], dim=0
        )  # (num_models, B)
        mean = rewards.mean(dim=0)   # (B,)
        std = rewards.std(dim=0)     # (B,)
        return mean, std

    def uncertainty(self, input_ids: Tensor) -> Tensor:
        """Return std of reward estimates across ensemble members. Shape (B,)."""
        _, std = self.forward(input_ids)
        return std


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def build_reward_fn(reward_model: RewardModel, tokenizer, device: str = "cpu") -> Callable:
    """Build a reward_fn compatible with GRPOTrainer.

    Returns a callable: (prompt: str, response: str) -> float
    """
    def reward_fn(prompt: str, response: str) -> float:
        text = prompt + response
        ids = tokenizer.encode(text)
        if not ids:
            return 0.0
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        if input_ids.shape[1] > reward_model.backbone.config.max_seq_len:
            input_ids = input_ids[:, -reward_model.backbone.config.max_seq_len:]
        return reward_model.score(input_ids)

    return reward_fn
