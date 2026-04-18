"""Asynchronous RL trainer with token-level double-sided importance sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AsyncRLConfig:
    """Configuration for the asynchronous RL trainer."""

    eps_low: float = 0.2
    eps_high: float = 0.2
    max_staleness: int = 4
    min_trajectory_len: int = 1
    group_size: int = 4
    gamma: float = 1.0


@dataclass
class Trajectory:
    """A single rollout trajectory with token-level data."""

    token_ids: torch.Tensor
    log_probs_rollout: torch.Tensor
    rewards: torch.Tensor
    rollout_version: int
    current_version: int = 0


class DoubleSidedIS:
    """Token-level double-sided importance sampling with clipping."""

    def __init__(self, eps_low: float, eps_high: float) -> None:
        self.eps_low = eps_low
        self.eps_high = eps_high

    def importance_ratio(
        self, log_pi_theta: torch.Tensor, log_pi_rollout: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token importance sampling ratio.

        Args:
            log_pi_theta:   per-token log probs under current policy.
            log_pi_rollout: per-token log probs under rollout policy.

        Returns:
            Per-token ratio tensor of same shape.
        """
        return torch.exp(log_pi_theta - log_pi_rollout)

    def clip_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        """Clamp importance ratio to [1 - eps_low, 1 + eps_high].

        Args:
            ratio: per-token IS ratios.

        Returns:
            Clamped tensor of same shape.
        """
        return ratio.clamp(1.0 - self.eps_low, 1.0 + self.eps_high)

    def masked_loss(
        self,
        log_pi_theta: torch.Tensor,
        log_pi_rollout: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute double-sided IS loss with out-of-bound token masking.

        Tokens whose importance ratio falls outside [1 - eps_low, 1 + eps_high]
        are masked out (zero gradient contribution).  The loss is the mean of
        ratio * advantages over the unmasked tokens.

        Args:
            log_pi_theta:   per-token log probs under current policy, shape (T,).
            log_pi_rollout: per-token log probs under rollout policy, shape (T,).
            advantages:     per-token advantages, shape (T,).

        Returns:
            Scalar loss tensor (negative mean, suitable for minimization with
            gradient ascent on the advantage signal).
        """
        ratio = self.importance_ratio(log_pi_theta, log_pi_rollout)

        lo = 1.0 - self.eps_low
        hi = 1.0 + self.eps_high
        in_range = (ratio >= lo) & (ratio <= hi)

        if not in_range.any():
            return torch.tensor(0.0, dtype=log_pi_theta.dtype, requires_grad=True)

        masked_ratio = ratio * in_range.float()
        masked_adv = advantages * in_range.float()

        n_unmasked = in_range.float().sum().clamp(min=1.0)
        loss = -(masked_ratio * masked_adv).sum() / n_unmasked
        return loss


class AsyncRLTrainer:
    """Asynchronous RL trainer decoupling trajectory generation from updates."""

    def __init__(self, model: nn.Module, config: AsyncRLConfig) -> None:
        self.model = model
        self.config = config
        self._version: int = 0

    def _is_stale(self, trajectory: Trajectory) -> bool:
        """Return True if the trajectory is too old to use safely.

        Args:
            trajectory: trajectory to check.

        Returns:
            True when current_version - rollout_version > max_staleness.
        """
        return (
            trajectory.current_version - trajectory.rollout_version
            > self.config.max_staleness
        )

    def _filter_trajectories(
        self, trajectories: List[Trajectory]
    ) -> List[Trajectory]:
        """Remove stale and too-short trajectories; stamp current_version.

        Args:
            trajectories: raw list of trajectories from the rollout buffer.

        Returns:
            Filtered list with current_version updated to self._version.
        """
        filtered: List[Trajectory] = []
        for traj in trajectories:
            traj.current_version = self._version
            if len(traj.token_ids) < self.config.min_trajectory_len:
                continue
            if self._is_stale(traj):
                continue
            filtered.append(traj)
        return filtered

    def _group_reward_baseline(
        self, trajectories: List[Trajectory]
    ) -> List[torch.Tensor]:
        """Subtract group mean reward to produce per-trajectory advantages.

        Trajectories are split into consecutive groups of size config.group_size.
        Any trailing trajectories that do not fill a complete group are treated
        as a partial group.

        Args:
            trajectories: filtered list of trajectories.

        Returns:
            List of scalar advantage tensors aligned with trajectories.
        """
        n = len(trajectories)
        advantages: List[torch.Tensor] = [torch.tensor(0.0)] * n

        g = self.config.group_size
        for start in range(0, n, g):
            group = trajectories[start : start + g]
            group_rewards = torch.stack(
                [
                    t.rewards.float().mean() if t.rewards.numel() > 1 else t.rewards.float().squeeze()
                    for t in group
                ]
            )
            group_mean = group_rewards.mean()
            for i, traj in enumerate(group):
                traj_reward = (
                    traj.rewards.float().mean()
                    if traj.rewards.numel() > 1
                    else traj.rewards.float().squeeze()
                )
                advantages[start + i] = traj_reward - group_mean
        return advantages

    def compute_log_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-token log probs for a 1-D sequence using the current model.

        Uses a cross-entropy-style computation: for position t the log prob of
        token_ids[t] is derived from logits at position t-1.  The returned
        tensor has length T-1 (one log prob per predicted token).

        Args:
            token_ids: 1-D LongTensor of length T (T >= 2).

        Returns:
            1-D float tensor of per-token log probs, shape (T-1,).
        """
        if token_ids.dim() == 1:
            input_ids = token_ids.unsqueeze(0)
        else:
            input_ids = token_ids

        _, logits, _ = self.model(input_ids)
        # logits: (1, T, vocab)
        # shift: predict token[1..T] from position [0..T-1]
        shift_logits = logits[0, :-1, :]      # (T-1, vocab)
        shift_targets = input_ids[0, 1:]      # (T-1,)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(shift_targets.size(0), device=shift_targets.device),
            shift_targets,
        ]  # (T-1,)
        return token_log_probs

    def train_step(
        self,
        trajectories: List[Trajectory],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Perform one asynchronous RL update step.

        Steps: filter -> compute advantages -> compute current log probs ->
        double-sided IS loss -> backward -> optimizer step.

        Args:
            trajectories: list of Trajectory objects from the rollout buffer.
            optimizer:    optimizer attached to self.model.

        Returns:
            Dict with keys 'loss', 'n_trajectories', 'mean_advantage'.
        """
        self._version += 1

        filtered = self._filter_trajectories(trajectories)

        if not filtered:
            return {"loss": 0.0, "n_trajectories": 0, "mean_advantage": 0.0}

        advantages = self._group_reward_baseline(filtered)

        is_fn = DoubleSidedIS(self.config.eps_low, self.config.eps_high)

        self.model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0)
        n_valid = 0

        for traj, adv in zip(filtered, advantages):
            token_ids = traj.token_ids
            if token_ids.numel() < 2:
                continue

            log_probs_current = self.compute_log_probs(token_ids)

            # Align rollout log probs with current (both length T-1)
            log_probs_rollout = traj.log_probs_rollout
            T = log_probs_current.size(0)
            if log_probs_rollout.size(0) != T:
                log_probs_rollout = log_probs_rollout[:T]

            adv_scalar = adv.detach()
            per_token_adv = adv_scalar.expand(T)

            loss = is_fn.masked_loss(
                log_probs_current,
                log_probs_rollout.detach(),
                per_token_adv,
            )
            total_loss = total_loss + loss
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
            total_loss.backward()
            optimizer.step()

        mean_adv = float(
            torch.stack([a.detach() for a in advantages]).mean().item()
        )

        return {
            "loss": total_loss.item(),
            "n_trajectories": len(filtered),
            "mean_advantage": mean_adv,
        }
