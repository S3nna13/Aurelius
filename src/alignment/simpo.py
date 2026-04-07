"""SimPO — Simple Preference Optimization (NeurIPS 2024).
Reference-free DPO variant using average log-prob reward.
No reference model or KL regularization needed.

Key differences from DPO:
- No reference model or KL term
- Reward = average log-probability (length-normalized), not sum
- Margin γ (gamma) enforces a target reward gap between chosen/rejected

Loss: -log sigmoid((beta * avg_logp_chosen - beta * avg_logp_rejected - gamma) / 1)
      = -log sigmoid(beta * (avg_logp_chosen - avg_logp_rejected) - gamma)

References:
    - SimPO: Simple Preference Optimization with a Reference-Free Reward (NeurIPS 2024)
      https://arxiv.org/abs/2405.14734
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    """SimPO loss module.

    Args:
        beta: Scaling factor for the reward signal. Default: 2.0.
        gamma: Target reward margin between chosen and rejected. Default: 0.5.
        label_smoothing: Label smoothing coefficient in [0, 1). Default: 0.0 (no smoothing).
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # shape: (batch,)
        policy_rejected_logps: torch.Tensor,  # shape: (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SimPO loss.

        Args:
            policy_chosen_logps: Average log-probabilities for chosen sequences. Shape: (batch,).
            policy_rejected_logps: Average log-probabilities for rejected sequences. Shape: (batch,).

        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards), where loss is a scalar
            and rewards have shape (batch,).
        """
        chosen_rewards = self.beta * policy_chosen_logps
        rejected_rewards = self.beta * policy_rejected_logps

        reward_margin = chosen_rewards - rejected_rewards - self.gamma

        # Primary NLL loss: -log sigmoid(reward_margin)
        nll_loss = -F.logsigmoid(reward_margin)

        if self.label_smoothing > 0.0:
            # Smoothed loss blends in the "wrong" direction
            smooth_loss = -F.logsigmoid(-reward_margin)
            loss = (
                (1.0 - self.label_smoothing) * nll_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            loss = nll_loss

        return loss.mean(), chosen_rewards, rejected_rewards


def compute_simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional SimPO loss. Returns (loss, chosen_rewards, rejected_rewards).

    Args:
        policy_chosen_logps: Average log-probabilities for chosen sequences. Shape: (batch,).
        policy_rejected_logps: Average log-probabilities for rejected sequences. Shape: (batch,).
        beta: Scaling factor for the reward signal.
        gamma: Target reward margin between chosen and rejected.

    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards), where loss is a scalar
        and rewards have shape (batch,).
    """
    loss_fn = SimPOLoss(beta=beta, gamma=gamma, label_smoothing=0.0)
    return loss_fn(policy_chosen_logps, policy_rejected_logps)
