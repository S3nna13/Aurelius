"""DPO (Direct Preference Optimization) trainer — Rafailov et al. 2023 (2305.18290)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False
    loss_type: str = "sigmoid"  # sigmoid | hinge | ipo


class DPOLoss(nn.Module):
    """DPO loss module supporting sigmoid, hinge, and IPO variants."""

    def __init__(self, config: DPOConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        reference_chosen_logps: Optional[Tensor],
        reference_rejected_logps: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute DPO loss and implicit rewards.

        Args:
            policy_chosen_logps: (B,) log-probs of chosen under policy.
            policy_rejected_logps: (B,) log-probs of rejected under policy.
            reference_chosen_logps: (B,) log-probs under reference, or None.
            reference_rejected_logps: (B,) log-probs under reference, or None.

        Returns:
            (loss, chosen_rewards, rejected_rewards)
        """
        cfg = self.config

        if cfg.reference_free:
            chosen_log_ratio = policy_chosen_logps
            rejected_log_ratio = policy_rejected_logps
        else:
            chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
            rejected_log_ratio = policy_rejected_logps - reference_rejected_logps

        logits = cfg.beta * (chosen_log_ratio - rejected_log_ratio)

        if cfg.loss_type == "sigmoid":
            if cfg.label_smoothing == 0.0:
                loss = -F.logsigmoid(logits).mean()
            else:
                ls = cfg.label_smoothing
                loss = (
                    -(1.0 - ls) * F.logsigmoid(logits)
                    - ls * F.logsigmoid(-logits)
                ).mean()
        elif cfg.loss_type == "hinge":
            loss = F.relu(1.0 - logits).mean()
        elif cfg.loss_type == "ipo":
            loss = (logits - 1.0 / (2.0 * cfg.beta)).pow(2).mean()
        else:
            raise ValueError(
                f"Unknown loss_type '{cfg.loss_type}'. Use 'sigmoid', 'hinge', or 'ipo'."
            )

        chosen_rewards = cfg.beta * chosen_log_ratio.detach()
        rejected_rewards = cfg.beta * rejected_log_ratio.detach()

        return loss, chosen_rewards, rejected_rewards


class DPOTrainer:
    """Direct Preference Optimization trainer (lightweight, no model required)."""

    def __init__(self, config: Optional[DPOConfig] = None) -> None:
        self.config = config if config is not None else DPOConfig()
        self._loss_fn = DPOLoss(self.config)

    def compute_loss(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        reference_chosen_logps: Optional[Tensor] = None,
        reference_rejected_logps: Optional[Tensor] = None,
    ) -> dict:
        """Compute DPO loss and reward metrics.

        Returns:
            dict with keys: loss, chosen_reward, rejected_reward, reward_margin
        """
        loss, chosen_rewards, rejected_rewards = self._loss_fn(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

    def reward_accuracy(self, chosen_rewards: Tensor, rejected_rewards: Tensor) -> float:
        """Fraction of pairs where chosen reward > rejected reward."""
        return (chosen_rewards > rejected_rewards).float().mean().item()
