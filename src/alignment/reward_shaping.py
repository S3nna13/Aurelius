"""Reward shaping: normalization, clipping, KL penalties, and composite reward functions."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from torch import Tensor


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping pipeline."""

    reward_clip: float = 5.0            # clip rewards to [-clip, clip]
    normalize_rewards: bool = True
    kl_penalty_coeff: float = 0.1
    length_penalty_coeff: float = 0.0  # negative = penalize long sequences
    diversity_bonus_coeff: float = 0.0
    running_stats_alpha: float = 0.01  # EMA alpha for running mean/std


class RunningMeanStd:
    """Online running mean and variance using Welford's algorithm.

    Uses exponential moving average (EMA) update controlled by alpha.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self._mean: float = 0.0
        self._var: float = 1.0
        self._count: int = 0

    def update(self, x: Tensor) -> None:
        """Update running statistics with a batch of values.

        Uses Welford-style online update blended with EMA for stability.

        Args:
            x: Tensor of any shape — all elements are used.
        """
        vals = x.detach().float().reshape(-1)
        batch_mean = vals.mean().item()
        batch_var = vals.var(unbiased=False).item() if vals.numel() > 1 else 0.0

        if self._count == 0:
            self._mean = batch_mean
            self._var = max(batch_var, 1e-8)
        else:
            self._mean = (1.0 - self.alpha) * self._mean + self.alpha * batch_mean
            self._var = (1.0 - self.alpha) * self._var + self.alpha * max(batch_var, 1e-8)

        self._count += vals.numel()

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize x using running statistics.

        Returns:
            (x - mean) / (std + 1e-8), same shape as x.
        """
        return (x - self._mean) / (self._var ** 0.5 + 1e-8)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._var ** 0.5


def clip_rewards(rewards: Tensor, clip: float) -> Tensor:
    """Clip rewards to [-clip, clip].

    Args:
        rewards: Tensor of any shape.
        clip: Clipping bound (must be positive).

    Returns:
        Clipped tensor, same shape.
    """
    return rewards.clamp(-clip, clip)


def kl_reward_penalty(
    log_probs_policy: Tensor,
    log_probs_ref: Tensor,
    coeff: float,
) -> Tensor:
    """Compute per-sequence KL penalty to subtract from reward.

    KL penalty = coeff * mean(log_pi - log_pi_ref, dim=-1)

    Args:
        log_probs_policy: (B, T) log probabilities under current policy.
        log_probs_ref:    (B, T) log probabilities under reference policy.
        coeff:            Penalty coefficient (kl_penalty_coeff).

    Returns:
        (B,) penalty values — caller subtracts these from reward.
    """
    return coeff * (log_probs_policy - log_probs_ref).mean(dim=-1)


def length_penalty(seq_lengths: Tensor, max_len: int, coeff: float) -> Tensor:
    """Compute per-sequence length penalty.

    penalty = coeff * (seq_lengths / max_len)
    With negative coeff this penalizes longer sequences.

    Args:
        seq_lengths: (B,) integer tensor of sequence lengths.
        max_len:     Maximum possible sequence length for normalization.
        coeff:       Length penalty coefficient (length_penalty_coeff).

    Returns:
        (B,) float penalties.
    """
    return coeff * (seq_lengths.float() / max_len)


def diversity_bonus(generated_ids_batch: list[Tensor], coeff: float) -> Tensor:
    """Compute pairwise Jaccard diversity bonus for a batch of completions.

    For each sequence, computes its mean Jaccard similarity to all other
    sequences in the batch; bonus = coeff * (1 - mean_similarity).

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        generated_ids_batch: List of B tensors, each 1-D token id sequence.
        coeff:               Diversity bonus coefficient.

    Returns:
        (B,) bonus values.
    """
    B = len(generated_ids_batch)
    device = generated_ids_batch[0].device if B > 0 else torch.device("cpu")

    if B == 0:
        return torch.zeros(0, device=device)

    # Convert each sequence to a set of unique token ids
    token_sets: list[set[int]] = [set(ids.tolist()) for ids in generated_ids_batch]

    bonuses = torch.zeros(B, device=device, dtype=torch.float32)

    if B == 1:
        # Single sequence — no pairwise comparison possible; bonus = 0
        return bonuses

    for i in range(B):
        sim_sum = 0.0
        for j in range(B):
            if i == j:
                continue
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            jaccard = intersection / union if union > 0 else 0.0
            sim_sum += jaccard
        mean_sim = sim_sum / (B - 1)
        bonuses[i] = coeff * (1.0 - mean_sim)

    return bonuses


class CompositeRewardFunction:
    """Combines base reward with KL penalty, length penalty, and diversity bonus.

    Pipeline:
        1. Clip base rewards to [-reward_clip, reward_clip].
        2. Normalize using running statistics (if normalize_rewards).
        3. Subtract KL penalty.
        4. Add length penalty (negative coeff → subtract).
        5. Add diversity bonus.
    """

    def __init__(self, config: RewardShapingConfig) -> None:
        self.config = config
        self._stats = RunningMeanStd(alpha=config.running_stats_alpha)

    def __call__(
        self,
        base_rewards: Tensor,
        log_probs_policy: Tensor,
        log_probs_ref: Tensor,
        generated_ids: list[Tensor],
    ) -> Tensor:
        """Shape rewards according to the composite pipeline.

        Args:
            base_rewards:     (B,) raw reward scores.
            log_probs_policy: (B, T) log probs under current policy.
            log_probs_ref:    (B, T) log probs under reference policy.
            generated_ids:    List of B 1-D token id tensors.

        Returns:
            (B,) shaped reward tensor.
        """
        cfg = self.config
        B = base_rewards.shape[0]

        # 1. Clip
        rewards = clip_rewards(base_rewards, cfg.reward_clip)

        # 2. Normalize using running stats
        if cfg.normalize_rewards:
            rewards = self._stats.normalize(rewards)

        # 3. Subtract KL penalty
        kl_pen = kl_reward_penalty(log_probs_policy, log_probs_ref, cfg.kl_penalty_coeff)
        rewards = rewards - kl_pen

        # 4. Add length penalty
        if cfg.length_penalty_coeff != 0.0:
            seq_lengths = torch.tensor(
                [ids.shape[0] for ids in generated_ids],
                dtype=torch.long,
                device=base_rewards.device,
            )
            max_len = log_probs_policy.shape[-1]
            len_pen = length_penalty(seq_lengths, max_len, cfg.length_penalty_coeff)
            rewards = rewards + len_pen

        # 5. Add diversity bonus
        if cfg.diversity_bonus_coeff != 0.0:
            div_bon = diversity_bonus(generated_ids, cfg.diversity_bonus_coeff)
            rewards = rewards + div_bon.to(rewards.device)

        return rewards

    def update_stats(self, rewards: Tensor) -> None:
        """Update running statistics with a batch of rewards.

        Args:
            rewards: (B,) or any shaped tensor of reward values.
        """
        self._stats.update(rewards)
