"""Process Reward Model (PRM): assigns rewards at each reasoning step.

Based on Lightman et al., arXiv:2305.20050 "Let's Verify Step by Step".
PRMs provide fine-grained RLHF signals for chain-of-thought reasoning by
scoring individual steps rather than only the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class StepReward:
    """Reward assigned to a single reasoning step."""

    step_idx: int
    reward: float
    is_correct: bool | None = None


class PRMHead(nn.Module):
    """Reward head for step-level scoring.

    Projects each token's hidden state to a scalar reward value.
    """

    def __init__(self, d_model: int, n_steps_hint: int = 1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_steps_hint = n_steps_hint
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute per-token scalar rewards.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            (B, T) scalar reward for every token position.
        """
        # (B, T, 1) -> (B, T)
        return self.linear(hidden_states).squeeze(-1)


class ProcessRewardModel(nn.Module):
    """Wraps a backbone transformer with a PRM head for step-level reward scoring."""

    def __init__(self, backbone: nn.Module, d_model: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.prm_head = PRMHead(d_model)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute step rewards from pre-extracted hidden states.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            (B, T) step rewards.
        """
        return self.prm_head(hidden_states)

    def score_steps(self, hidden_states: Tensor, step_mask: Tensor) -> Tensor:
        """Score only step-boundary positions, padding with 0 for shorter sequences.

        Args:
            hidden_states: (B, T, d_model)
            step_mask:     (B, T) bool — True at step-boundary positions.

        Returns:
            (B, n_steps) rewards, where n_steps = max number of boundaries in batch.
            Positions beyond a batch item's step count are padded with 0.
        """
        all_rewards: Tensor = self.prm_head(hidden_states)  # (B, T)

        B = hidden_states.shape[0]
        steps_per_item = step_mask.sum(dim=1)  # (B,)
        max_steps = int(steps_per_item.max().item())
        if max_steps == 0:
            return torch.zeros(B, 1, device=hidden_states.device)

        output = torch.zeros(B, max_steps, device=hidden_states.device)
        for b in range(B):
            positions = step_mask[b].nonzero(as_tuple=False).squeeze(1)  # (n_steps,)
            if positions.numel() == 0:
                continue
            rewards_at_steps = all_rewards[b, positions]  # (n_steps,)
            n = rewards_at_steps.shape[0]
            output[b, :n] = rewards_at_steps

        return output


class PRMLoss:
    """Binary cross-entropy training loss for the PRM at step-boundary positions."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(
        self,
        predicted_rewards: Tensor,
        step_labels: Tensor,
        step_mask: Tensor,
    ) -> Tensor:
        """Compute masked BCE loss over step positions.

        Args:
            predicted_rewards: (B, T) raw logits from the PRM head.
            step_labels:       (B, T) binary labels - 1 for correct step, 0 for incorrect.
            step_mask:         (B, T) bool - True at positions to include in loss.

        Returns:
            Scalar loss. Returns 0.0 (requires_grad) when mask is all-False.
        """
        if not step_mask.any():
            return predicted_rewards.sum() * 0.0

        valid_preds = predicted_rewards[step_mask]
        valid_labels = step_labels[step_mask].float()
        return F.binary_cross_entropy_with_logits(
            valid_preds, valid_labels, reduction=self.reduction
        )


class PRMTrainer:
    """Handles training and evaluation of a ProcessRewardModel."""

    def __init__(self, model: ProcessRewardModel, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = PRMLoss()

    def train_step(
        self,
        hidden_states: Tensor,
        step_labels: Tensor,
        step_mask: Tensor,
    ) -> dict[str, float]:
        """Execute a single training step.

        Args:
            hidden_states: (B, T, d_model)
            step_labels:   (B, T) in {0, 1}
            step_mask:     (B, T) bool

        Returns:
            Dict with keys 'loss', 'accuracy', 'n_steps'.
        """
        self.model.prm_head.train()
        self.optimizer.zero_grad()

        predicted = self.model(hidden_states)  # (B, T)
        loss = self.loss_fn(predicted, step_labels, step_mask)
        loss.backward()
        self.optimizer.step()

        # Accuracy over masked positions
        n_steps = int(step_mask.sum().item())
        if n_steps > 0:
            with torch.no_grad():
                preds_binary = (predicted[step_mask] > 0).float()
                labels_binary = step_labels[step_mask].float()
                accuracy = (preds_binary == labels_binary).float().mean().item()
        else:
            accuracy = 0.0

        return {
            "loss": loss.item(),
            "accuracy": float(accuracy),
            "n_steps": float(n_steps),
        }

    def evaluate_chain(
        self,
        hidden_states: Tensor,
        step_mask: Tensor,
    ) -> list[StepReward]:
        """Evaluate a reasoning chain and return per-step rewards.

        Operates on the first batch item only.

        Args:
            hidden_states: (B, T, d_model)
            step_mask:     (B, T) bool

        Returns:
            List of StepReward objects, one per step boundary in item 0.
        """
        self.model.prm_head.train(False)
        with torch.no_grad():
            all_rewards = self.model(hidden_states)  # (B, T)

        positions = step_mask[0].nonzero(as_tuple=False).squeeze(1)  # (n_steps,)
        result: list[StepReward] = []
        for idx, pos in enumerate(positions.tolist()):
            reward_val = all_rewards[0, pos].item()
            result.append(StepReward(step_idx=idx, reward=reward_val))
        return result
