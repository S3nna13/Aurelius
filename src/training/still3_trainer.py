"""STILL-3 Trainer — Slow Thinking with LLMs (2025).

Implements two core techniques from the STILL-3 paper for reasoning efficiency:

1. **Minimum-std filtering** — discard rollout groups where all rewards are
   similar (easy tasks the model always gets right, or hard tasks it always
   gets wrong), keeping only groups where the model can learn something.

2. **Entropy bonus** — add a token-level entropy term to the RL objective to
   encourage diverse thinking chains and prevent premature collapse.

Public API
----------
STILL3Config        -- hyperparameters dataclass
STILL3Trainer       -- trainer class with all algorithm components

The trainer is registered in TRAINING_REGISTRY["still3"].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# STILL3Config
# ---------------------------------------------------------------------------


@dataclass
class STILL3Config:
    """Hyperparameters for the STILL-3 trainer.

    Attributes:
        min_std_threshold: discard rollout groups whose reward std falls below
            this value (both trivially-easy and trivially-hard tasks).
        entropy_coeff: weight for the entropy bonus term in the total loss.
            Higher values push the policy toward more diverse responses.
        gamma: discount factor (reserved for multi-step returns; currently
            applied in the reward pipeline but defaults to 1.0).
        group_size: expected number of rollouts per question (GRPO-style).
        normalize_rewards: if True, Z-score normalise rewards within each
            group before using them as advantages.
    """

    min_std_threshold: float = 0.05
    entropy_coeff: float = 0.01
    gamma: float = 1.0
    group_size: int = 8
    normalize_rewards: bool = True


# ---------------------------------------------------------------------------
# STILL3Trainer
# ---------------------------------------------------------------------------


class STILL3Trainer:
    """STILL-3 training algorithm.

    Combines minimum-std rollout filtering with an entropy bonus to improve
    reasoning efficiency in chain-of-thought RL training.

    Args:
        config: STILL3Config instance (uses defaults if None).
    """

    def __init__(self, config: STILL3Config | None = None) -> None:
        self.config = config if config is not None else STILL3Config()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_std(self, rewards_per_group: list[list[float]]) -> list[list[float]]:
        """Discard groups whose reward std is below ``min_std_threshold``.

        Groups where all rewards are identical (or nearly so) provide no
        useful learning signal — the model already knows the answer or has
        no chance of getting it right — and are therefore removed.

        Args:
            rewards_per_group: list of groups, each group being a list of
                per-rollout float rewards.

        Returns:
            Filtered list of groups; only groups with
            ``std(rewards) >= min_std_threshold`` are kept.
        """
        threshold = self.config.min_std_threshold
        filtered: list[list[float]] = []
        for group in rewards_per_group:
            if len(group) == 0:
                continue
            t = torch.tensor(group, dtype=torch.float32)
            std = t.std(unbiased=False).item()
            if std >= threshold:
                filtered.append(group)
        return filtered

    # ------------------------------------------------------------------
    # Entropy bonus
    # ------------------------------------------------------------------

    def compute_entropy_bonus(self, logits: Tensor) -> Tensor:
        """Compute mean token-level entropy of the policy distribution.

        High entropy signals a diverse distribution over the vocabulary;
        adding this as a bonus discourages the policy from collapsing to
        a single deterministic thinking chain.

        Args:
            logits: raw (un-normalised) logits of shape ``[B, T, V]``.

        Returns:
            Scalar tensor — the mean per-token entropy averaged over both
            the batch dimension B and the time dimension T.
        """
        # log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        probs = log_probs.exp()  # [B, T, V]
        # entropy per token: -sum_v p(v) log p(v)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
        return token_entropy.mean()  # scalar

    # ------------------------------------------------------------------
    # Reward normalisation
    # ------------------------------------------------------------------

    def normalize_group_rewards(self, rewards: list[float]) -> list[float]:
        """Z-score normalise rewards within a single group.

        Args:
            rewards: list of float rewards for one rollout group.

        Returns:
            Normalised rewards ``(r - mean) / (std + 1e-8)``.  If all
            rewards are identical the function returns a list of zeros.
        """
        t = torch.tensor(rewards, dtype=torch.float32)
        mean = t.mean()
        std = t.std(unbiased=False)
        if std.item() < 1e-8:
            return [0.0] * len(rewards)
        normalised = (t - mean) / (std + 1e-8)
        return normalised.tolist()

    # ------------------------------------------------------------------
    # Policy loss
    # ------------------------------------------------------------------

    def compute_policy_loss(
        self,
        log_probs: Tensor,
        rewards: Tensor,
        old_log_probs: Tensor | None = None,
    ) -> Tensor:
        """REINFORCE-style policy gradient loss with optional PPO clipping.

        Args:
            log_probs: shape ``[B]`` — log-probability of the sampled
                sequence under the *current* policy.
            rewards: shape ``[B]`` — advantages / normalised rewards.
            old_log_probs: if provided, apply PPO-style importance-weight
                clipping with ``eps=0.2``.  Shape must match ``log_probs``.

        Returns:
            Scalar policy-gradient loss.  Positive reward increases the
            log-prob of the sampled action (i.e. loss is driven negative).
        """
        if old_log_probs is None:
            # Pure REINFORCE
            loss = -(rewards * log_probs).mean()
        else:
            # PPO clipped surrogate
            ratio = (log_probs - old_log_probs.detach()).exp()
            clip_eps = 0.2
            clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
            surr1 = ratio * rewards
            surr2 = clipped_ratio * rewards
            loss = -torch.min(surr1, surr2).mean()
        return loss

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def total_loss(
        self,
        logits: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        old_log_probs: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute combined STILL-3 loss: policy gradient minus entropy bonus.

        ``total = policy_loss - entropy_coeff * entropy_bonus``

        The entropy bonus is *subtracted* because we want to *maximise*
        entropy (more diverse thinking), while loss is *minimised* by the
        optimiser.

        Args:
            logits: ``[B, T, V]`` raw logits for entropy computation.
            log_probs: ``[B]`` sequence log-probs under current policy.
            rewards: ``[B]`` advantages.
            old_log_probs: optional ``[B]`` for PPO clipping.

        Returns:
            Tuple of:
            - ``total_loss``: scalar Tensor (differentiable).
            - ``metrics``: dict with keys ``policy_loss``, ``entropy_bonus``,
              ``total`` (float values).
        """
        policy_loss = self.compute_policy_loss(log_probs, rewards, old_log_probs)
        entropy_bonus = self.compute_entropy_bonus(logits)

        total = policy_loss - self.config.entropy_coeff * entropy_bonus

        metrics: dict[str, float] = {
            "policy_loss": policy_loss.item(),
            "entropy_bonus": entropy_bonus.item(),
            "total": total.item(),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Full pipeline helper
    # ------------------------------------------------------------------

    def filter_and_prepare(self, groups: list[dict]) -> list[dict]:
        """Apply std filtering and optional reward normalisation to groups.

        Each group dict must contain at minimum a ``"rewards"`` key (list of
        floats).  All other keys (e.g. ``"log_probs"``, ``"logits"``) are
        passed through unchanged.

        Args:
            groups: list of dicts, each representing one rollout group.

        Returns:
            Filtered and (optionally) reward-normalised groups.  Groups
            whose reward std falls below ``min_std_threshold`` are removed.
        """
        threshold = self.config.min_std_threshold

        kept_groups: list[dict] = []
        for g in groups:
            rewards = g["rewards"]
            if len(rewards) == 0:
                continue
            t = torch.tensor(rewards, dtype=torch.float32)
            std = t.std(unbiased=False).item()
            if std >= threshold:
                kept_groups.append(g)

        # Normalise rewards within each kept group
        if self.config.normalize_rewards:
            result: list[dict] = []
            for g in kept_groups:
                new_g = dict(g)  # shallow copy to avoid mutating caller's dict
                new_g["rewards"] = self.normalize_group_rewards(g["rewards"])
                result.append(new_g)
            return result

        return kept_groups
