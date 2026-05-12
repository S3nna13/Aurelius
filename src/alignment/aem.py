"""AEM: Adaptive Entropy Modulation for Multi-Turn Agentic RL.

Supervision-free credit assignment via entropy dynamics. Response-level
uncertainty proxy derived from interaction between sampled-response
advantage and its relative surprisal.

Paper: arXiv:2605.00425 — Zhao et al.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EntropyModulator(nn.Module):
    """Adaptively modulates entropy for exploration-exploitation balance."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.advantage_proj = nn.Linear(d_model, 1, bias=False)
        self.surprisal_proj = nn.Linear(d_model, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, response_hidden: Tensor, baseline_hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Compute entropy modulation signal and uncertainty proxy.

        Args:
            response_hidden: (B, S, d_model) current response hidden states
            baseline_hidden: (B, S, d_model) baseline/reference hidden states
        Returns:
            (entropy_modulation, uncertainty_proxy)
        """
        adv = self.advantage_proj(response_hidden - baseline_hidden).squeeze(-1)
        surprise = self.surprisal_proj(response_hidden).squeeze(-1)
        uncertainty = torch.sigmoid(adv / (surprise.abs() + 1e-8))
        entropy_mod = uncertainty * self.alpha
        return entropy_mod, uncertainty


def response_level_advantage(
    token_log_probs: Tensor,
    response_mask: Tensor,
    baseline_reward: Tensor,
    entropy_proxy: Tensor,
    gamma: float = 1.0,
) -> Tensor:
    """Compute response-level advantage using entropy-rescaled rewards.

    Args:
        token_log_probs: (B, S) log probs per token
        response_mask: (B, S) bool mask for valid token positions
        baseline_reward: (B,) scalar outcome reward
        entropy_proxy: (B,) response-level uncertainty from AEM
        gamma: discount factor (typically 1.0 for RLHF)
    Returns:
        advantages: (B, S) rescaled advantages
    """
    seq_rewards = token_log_probs.sum(dim=-1) * response_mask.float().sum(-1).clamp(min=1)
    reward_signal = seq_rewards + gamma * baseline_reward
    adv = reward_signal.unsqueeze(1) - reward_signal.mean()
    adv_std = adv.std().clamp(min=1e-8)
    normalized_adv = adv / adv_std
    return normalized_adv * entropy_proxy.unsqueeze(1)


def aem_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    entropy_proxy: Tensor,
    clip_ratio: float = 0.2,
) -> tuple[Tensor, dict]:
    """AEM-adaptive PPO-style loss with entropy-modulated advantages.

    Args:
        log_probs: (B, S) current policy log probs
        old_log_probs: (B, S) old policy log probs
        advantages: (B, S) entropy-rescaled advantages
        entropy_proxy: (B,) response-level uncertainty
        clip_ratio: PPO clipping epsilon
    Returns:
        (loss, metrics)
    """
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    loss = -torch.min(surr1, surr2).mean()

    entropy_bonus = entropy_proxy.mean() * 0.01
    total_loss = loss - entropy_bonus

    metrics = {
        "aem_loss": loss.item(),
        "mean_entropy_proxy": entropy_proxy.mean().item(),
        "clip_fraction": ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio))
        .float()
        .mean()
        .item(),
    }
    return total_loss, metrics


class AEMCreditAssigner:
    """Supervision-free credit assignment via response-level entropy modulation."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.modulator = EntropyModulator(model.config.d_model)

    def assign_credits(
        self, responses: list[Tensor], baseline_responses: list[Tensor]
    ) -> list[Tensor]:
        """Assign entropy-modulated credits to multi-turn interaction steps."""
        credits = []
        for resp, base in zip(responses, baseline_responses):
            mod, unc = self.modulator(resp, base)
            credits.append(unc)
        return credits


__all__ = ["AEMCreditAssigner", "aem_loss", "response_level_advantage", "EntropyModulator"]
