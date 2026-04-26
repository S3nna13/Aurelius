"""Token-level reward signals for dense RL fine-tuning.

Distinct from process_reward.py (step-level scoring) and reward_model.py
(sequence-level scalar). This module provides per-token dense rewards,
discounted return computation, GAE, reward shaping, and a REINFORCE trainer.
"""

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
class TokenRewardConfig:
    """Configuration for token-level dense reward signals."""

    reward_type: str = "dense"  # "dense" | "sparse" | "shaped"
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    normalize: bool = True  # normalize rewards per sequence
    clip_reward: float = 10.0  # clip rewards to [-clip_reward, clip_reward]


# ---------------------------------------------------------------------------
# TokenRewardModel
# ---------------------------------------------------------------------------


class TokenRewardModel(nn.Module):
    """Per-token reward model wrapping an AureliusTransformer backbone.

    Produces one scalar reward per token by hooking into the final
    layer-norm hidden states of the base model and projecting through
    a linear head.

    Args:
        base_model: AureliusTransformer (or compatible model with .norm layer).
        d_model: Hidden dimension of the base model.
    """

    def __init__(self, base_model: nn.Module, d_model: int = 64) -> None:
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute per-token rewards.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            (B, T) reward tensor — one scalar reward per token.
        """
        hidden_states: list[Tensor] = []

        hook = self.base_model.norm.register_forward_hook(lambda m, i, o: hidden_states.append(o))
        try:
            self.base_model(input_ids)
        finally:
            hook.remove()

        h = hidden_states[0]  # (B, T, d_model)
        rewards = self.reward_head(h).squeeze(-1)  # (B, T)
        return rewards


# ---------------------------------------------------------------------------
# Discounted returns
# ---------------------------------------------------------------------------


def compute_returns(rewards: Tensor, gamma: float = 0.99) -> Tensor:
    """Compute discounted returns for each token position.

    G_t = r_t + gamma * r_{t+1} + ... + gamma^{T-t-1} * r_{T-1}

    Args:
        rewards: (B, T) per-token rewards.
        gamma:   Discount factor.

    Returns:
        (B, T) discounted returns.
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running

    return returns


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tensor:
    """Compute Generalized Advantage Estimation (GAE).

    delta_t = r_t + gamma * V_{t+1} - V_t
    A_t = delta_t + (gamma * lam) * A_{t+1}

    The value at position T (beyond the last token) is treated as 0.

    Args:
        rewards: (B, T) per-token rewards.
        values:  (B, T) per-token value estimates.
        gamma:   Discount factor.
        lam:     GAE lambda.

    Returns:
        (B, T) advantage estimates.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_value = (
            values[:, t + 1]
            if t + 1 < T
            else torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        )
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * last_adv
        advantages[:, t] = last_adv

    return advantages


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


def shape_rewards(rewards: Tensor, config: TokenRewardConfig) -> Tensor:
    """Apply reward shaping: clipping and optional normalization.

    1. Clip to [-clip_reward, clip_reward].
    2. If normalize: standardize each sequence to zero mean, unit std.

    Args:
        rewards: (B, T) raw rewards.
        config:  TokenRewardConfig with clip_reward and normalize settings.

    Returns:
        (B, T) shaped rewards.
    """
    # Clip
    rewards = rewards.clamp(-config.clip_reward, config.clip_reward)

    if config.normalize:
        # Normalize per sequence (across T dimension)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        rewards = (rewards - mean) / std

    return rewards


# ---------------------------------------------------------------------------
# TokenRewardTrainer
# ---------------------------------------------------------------------------


class TokenRewardTrainer:
    """REINFORCE trainer with token-level dense reward signals.

    Args:
        policy:       Policy model (AureliusTransformer or compatible).
        reward_model: TokenRewardModel providing per-token rewards.
        optimizer:    PyTorch optimizer for the policy.
        config:       TokenRewardConfig.
    """

    def __init__(
        self,
        policy: nn.Module,
        reward_model: TokenRewardModel,
        optimizer: torch.optim.Optimizer,
        config: TokenRewardConfig,
    ) -> None:
        self.policy = policy
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.config = config

    def compute_policy_gradient_loss(
        self,
        input_ids: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """REINFORCE policy gradient loss at the token level.

        Loss = -mean(log_prob * advantages)

        where log_prob is the per-token log probability of each token
        given the preceding context.

        Args:
            input_ids:  (B, T) token indices.
            advantages: (B, T) advantage estimates.

        Returns:
            Scalar loss.
        """
        _, logits, _ = self.policy(input_ids)  # logits: (B, T, vocab_size)

        # Shift: predict token t from context [0..t-1]
        # log_probs for positions 1..T using logits 0..T-1
        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        target_ids = input_ids[:, 1:]  # (B, T-1)
        token_log_probs = log_probs_all.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # Align advantages: drop the last position (no prediction target for it)
        adv = advantages[:, :-1]  # (B, T-1)

        loss = -(token_log_probs * adv).mean()
        return loss

    def train_step(self, input_ids: Tensor) -> dict:
        """One full RL update step using token-level rewards.

        1. Obtain per-token rewards from reward_model.
        2. Compute discounted returns.
        3. Compute GAE using returns as both rewards and values (zero baseline).
        4. Shape rewards.
        5. Compute REINFORCE policy gradient loss.
        6. Backward + optimizer step.

        Args:
            input_ids: (B, T) token indices.

        Returns:
            dict with keys: "loss", "mean_reward", "mean_advantage", "mean_return".
        """
        self.policy.train()

        # Step 1: get per-token rewards (no grad through reward model)
        with torch.no_grad():
            raw_rewards = self.reward_model(input_ids)  # (B, T)

        # Step 2: compute discounted returns
        returns = compute_returns(raw_rewards, gamma=self.config.gamma)  # (B, T)

        # Step 3: compute GAE with zero value baseline
        values = torch.zeros_like(raw_rewards)
        advantages = compute_gae(
            raw_rewards,
            values,
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
        )  # (B, T)

        # Step 4: shape rewards
        shaped = shape_rewards(raw_rewards, self.config)  # (B, T)

        # Step 5: compute policy gradient loss using shaped rewards as advantages
        loss = self.compute_policy_gradient_loss(input_ids, shaped)

        # Step 6: backward + step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": raw_rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_return": returns.mean().item(),
        }
