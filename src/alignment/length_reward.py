"""Length-Conditional Reward for efficient thinking chains (ThinkPrune / 2025).

Motivation: In RL training for reasoning, penalize unnecessarily long thinking
chains. Correct rollouts within token budget get a bonus; correct rollouts that
exceed budget get penalized; incorrect rollouts always get the base penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LengthRewardConfig:
    """Configuration for length-conditional reward."""

    token_budget: int = 4096  # target thinking token budget
    length_penalty_weight: float = 0.1  # penalty per normalized token above budget
    length_bonus_weight: float = 0.05  # bonus per normalized token below budget
    correct_base_reward: float = 1.0  # base reward for correct answers
    incorrect_base_reward: float = 0.0  # base reward for incorrect answers
    min_length_ratio: float = 0.1  # don't bonus below 10% of budget (prevents trivial shortcuts)
    max_length_ratio: float = 3.0  # cap length consideration at 3× budget


# ---------------------------------------------------------------------------
# LengthReward
# ---------------------------------------------------------------------------


class LengthReward:
    """Length-conditional reward function for RL training of reasoning models.

    Applies bonuses for correct short-but-not-trivial responses and penalties
    for correct over-budget responses. Incorrect responses always receive the
    base incorrect reward regardless of length.

    Args:
        config: LengthRewardConfig instance. Defaults to LengthRewardConfig().
    """

    def __init__(self, config: LengthRewardConfig | None = None) -> None:
        self.config = config if config is not None else LengthRewardConfig()

    def compute(self, is_correct: bool, n_tokens: int) -> float:
        """Compute scalar reward for a single rollout.

        Args:
            is_correct: Whether the rollout produced a correct answer.
            n_tokens:   Number of tokens used in the thinking chain.

        Returns:
            Scalar float reward.
        """
        cfg = self.config

        # Incorrect rollouts always receive base incorrect reward
        if not is_correct:
            return cfg.incorrect_base_reward

        # Too-short responses are treated as cheating (trivial shortcuts)
        if n_tokens < cfg.token_budget * cfg.min_length_ratio:
            return cfg.incorrect_base_reward

        # Compute normalized length factor: (n_tokens - budget) / budget
        # Clamped to [-1, max_length_ratio - 1]
        length_factor = (n_tokens - cfg.token_budget) / cfg.token_budget
        length_factor = max(-1.0, min(length_factor, cfg.max_length_ratio - 1.0))

        if n_tokens <= cfg.token_budget:
            # Within budget: give a bonus proportional to how much shorter it is
            bonus = cfg.length_bonus_weight * abs(length_factor)
            return cfg.correct_base_reward + bonus
        else:
            # Over budget: apply a length penalty
            penalty = cfg.length_penalty_weight * length_factor
            return cfg.correct_base_reward - penalty

    def compute_batch(self, correctness: list[bool], token_counts: list[int]) -> list[float]:
        """Compute rewards for a batch of rollouts.

        Args:
            correctness:  List of bool indicating correctness per rollout.
            token_counts: List of int token counts per rollout.

        Returns:
            List of float rewards, one per rollout.
        """
        return [
            self.compute(is_correct, n_tokens)
            for is_correct, n_tokens in zip(correctness, token_counts)
        ]

    def compute_tensor(self, correctness: Tensor, token_counts: Tensor) -> Tensor:
        """Compute rewards as a tensor for a batch of rollouts.

        Args:
            correctness:  [B] bool tensor indicating correctness per rollout.
            token_counts: [B] int tensor of token counts per rollout.

        Returns:
            [B] float reward tensor.
        """
        cfg = self.config
        B = correctness.shape[0]

        # Work in float for arithmetic
        tokens_f = token_counts.float()
        budget = float(cfg.token_budget)

        # Normalized length factor, clamped
        length_factor = (tokens_f - budget) / budget
        length_factor = length_factor.clamp(-1.0, cfg.max_length_ratio - 1.0)

        # Masks
        correct_mask = correctness.bool()
        too_short_mask = tokens_f < budget * cfg.min_length_ratio
        within_budget_mask = tokens_f <= budget
        over_budget_mask = tokens_f > budget

        # Start from correct base reward everywhere
        rewards = torch.full((B,), cfg.correct_base_reward, dtype=torch.float32)

        # Bonus for correct, within-budget (not too short)
        bonus_mask = correct_mask & within_budget_mask & ~too_short_mask
        rewards = torch.where(
            bonus_mask,
            rewards + cfg.length_bonus_weight * length_factor.abs(),
            rewards,
        )

        # Penalty for correct, over-budget
        penalty_mask = correct_mask & over_budget_mask
        rewards = torch.where(
            penalty_mask,
            rewards - cfg.length_penalty_weight * length_factor,
            rewards,
        )

        # Too-short correct rollouts → incorrect base
        rewards = torch.where(
            correct_mask & too_short_mask,
            torch.full((B,), cfg.incorrect_base_reward, dtype=torch.float32),
            rewards,
        )

        # Incorrect rollouts → incorrect base
        rewards = torch.where(
            ~correct_mask,
            torch.full((B,), cfg.incorrect_base_reward, dtype=torch.float32),
            rewards,
        )

        return rewards

    def statistics(self, correctness: list[bool], token_counts: list[int]) -> dict:
        """Compute summary statistics over a batch of rollouts.

        Args:
            correctness:  List of bool per rollout.
            token_counts: List of int token counts per rollout.

        Returns:
            dict with keys:
                "mean_reward": float — average reward across all rollouts.
                "mean_tokens": float — average token count.
                "n_correct":   int   — number of correct rollouts.
                "n_penalized": int   — correct rollouts that exceeded budget.
                "n_bonus":     int   — correct rollouts within budget (not too short).
        """
        cfg = self.config
        rewards = self.compute_batch(correctness, token_counts)

        n_correct = sum(1 for c in correctness if c)
        n_penalized = sum(
            1 for c, t in zip(correctness, token_counts) if c and t > cfg.token_budget
        )
        n_bonus = sum(
            1
            for c, t in zip(correctness, token_counts)
            if c and t <= cfg.token_budget and t >= cfg.token_budget * cfg.min_length_ratio
        )

        return {
            "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "mean_tokens": float(sum(token_counts) / len(token_counts)) if token_counts else 0.0,
            "n_correct": n_correct,
            "n_penalized": n_penalized,
            "n_bonus": n_bonus,
        }
