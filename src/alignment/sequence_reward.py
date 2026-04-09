"""Sequence-level reward modeling for RLHF.

Train a reward model on preference data and use it to score completions.
Uses a pooling strategy over hidden states and Bradley-Terry preference loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SequenceRewardConfig:
    pooling: str = "last"           # "last" | "mean" | "max" -- how to pool hidden states
    reward_head_dim: int = 128      # hidden dim of reward MLP head
    margin: float = 0.5             # Bradley-Terry margin
    label_smoothing: float = 0.0
    normalize_rewards: bool = True   # z-score normalize reward outputs
    dropout: float = 0.1


class RewardHead(nn.Module):
    """MLP head mapping d_model to scalar reward."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> (B,) scalar rewards"""
        return self.net(x).squeeze(-1)


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask,
    method: str,
) -> torch.Tensor:
    """Pool (B, T, D) -> (B, D).

    last: take last non-padding token (or last token if no mask)
    mean: mean over non-padding tokens
    max: max over non-padding tokens
    """
    B, T, D = hidden_states.shape

    if method == "last":
        if attention_mask is None:
            return hidden_states[:, -1, :]
        mask = attention_mask.bool()
        lengths = mask.sum(dim=1)
        last_indices = (lengths - 1).clamp(min=0)
        idx = last_indices.unsqueeze(-1).unsqueeze(-1).expand(B, 1, D)
        return hidden_states.gather(1, idx).squeeze(1)

    elif method == "mean":
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.bool().float()
        mask_expanded = mask.unsqueeze(-1)
        summed = (hidden_states * mask_expanded).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return summed / counts

    elif method == "max":
        if attention_mask is None:
            return hidden_states.max(dim=1).values
        mask = attention_mask.bool()
        mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
        filled = hidden_states.masked_fill(~mask_expanded, float("-inf"))
        return filled.max(dim=1).values

    else:
        raise ValueError(f"Unknown pooling method: {method!r}. Choose 'last', 'mean', or 'max'.")


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 0.5,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry preference loss.

    loss = -log(sigmoid(chosen - rejected - margin))
    With label smoothing: mix with 0.5 constant.
    Returns mean loss scalar.
    """
    logits = chosen_rewards - rejected_rewards - margin
    loss = -F.logsigmoid(logits)

    if label_smoothing > 0.0:
        uniform_loss = -torch.log(torch.tensor(0.5, dtype=loss.dtype, device=loss.device))
        loss = (1.0 - label_smoothing) * loss + label_smoothing * uniform_loss

    return loss.mean()


def compute_reward_accuracy(chosen: torch.Tensor, rejected: torch.Tensor) -> float:
    """Fraction of pairs where chosen_reward > rejected_reward."""
    return (chosen > rejected).float().mean().item()


def compute_reward_margin(chosen: torch.Tensor, rejected: torch.Tensor) -> float:
    """Mean (chosen - rejected)."""
    return (chosen - rejected).mean().item()


class SequenceRewardModel(nn.Module):
    """Language model backbone + reward head for sequence scoring."""

    def __init__(
        self,
        backbone: nn.Module,
        cfg: SequenceRewardConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        d_model = backbone.config.d_model
        self.reward_head = RewardHead(d_model, cfg.reward_head_dim, cfg.dropout)

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone, extract last-layer hidden states via forward hook.

        Returns (B, T, D).
        """
        captured: list[torch.Tensor] = []

        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                captured.append(output[0].detach() if not output[0].requires_grad else output[0])
            else:
                captured.append(output)

        hook = self.backbone.layers[-1].register_forward_hook(hook_fn)
        try:
            _ = self.backbone(input_ids)
        finally:
            hook.remove()

        if not captured:
            raise RuntimeError("Forward hook did not capture any output.")

        return captured[0]

    def forward(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Returns (B,) reward scores."""
        hidden_states = self.get_hidden_states(input_ids)
        pooled = pool_hidden_states(hidden_states, attention_mask, self.cfg.pooling)
        return self.reward_head(pooled)

    def score_batch(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Same as forward but with no_grad and eval mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids)


class RewardModelTrainerV2:
    """Train SequenceRewardModel on preference pairs."""

    def __init__(
        self,
        reward_model: SequenceRewardModel,
        optimizer: torch.optim.Optimizer,
        cfg: SequenceRewardConfig,
    ) -> None:
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.cfg = cfg

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict:
        """Forward both, compute BT loss, backward, step.

        Returns dict with keys: loss, accuracy, mean_margin,
        mean_chosen_reward, mean_rejected_reward.
        """
        self.reward_model.train()
        self.optimizer.zero_grad()

        chosen_rewards = self.reward_model(chosen_ids)
        rejected_rewards = self.reward_model(rejected_ids)

        loss = bradley_terry_loss(
            chosen_rewards,
            rejected_rewards,
            margin=self.cfg.margin,
            label_smoothing=self.cfg.label_smoothing,
        )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": compute_reward_accuracy(chosen_rewards.detach(), rejected_rewards.detach()),
            "mean_margin": compute_reward_margin(chosen_rewards.detach(), rejected_rewards.detach()),
            "mean_chosen_reward": chosen_rewards.detach().mean().item(),
            "mean_rejected_reward": rejected_rewards.detach().mean().item(),
        }

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Z-score normalize: (r - mean) / (std + 1e-8)"""
        mean = rewards.mean()
        std = rewards.std()
        return (rewards - mean) / (std + 1e-8)
