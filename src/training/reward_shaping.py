"""Reward shaping, normalization, and transformation utilities for RL-based LLM training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RewardShapingConfig:
    normalize_method: str = "zscore"  # "zscore" | "minmax" | "rank" | "none"
    clip_range: float = 5.0  # clip rewards to ±clip_range after normalization
    gamma: float = 1.0  # discount factor for multi-step returns
    use_potential_shaping: bool = False  # potential-based shaping
    ewma_alpha: float = 0.01  # for running stats (EWMARewardNormalizer)
    penalty_coeff: float = 0.1  # for length/repetition penalties


def normalize_rewards_zscore(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Z-score normalize rewards: (r - mean) / (std + eps)."""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def normalize_rewards_minmax(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Min-max normalize rewards to [0, 1]: (r - min) / (max - min + eps)."""
    r_min = rewards.min()
    r_max = rewards.max()
    return (rewards - r_min) / (r_max - r_min + eps)


def normalize_rewards_rank(rewards: Tensor) -> Tensor:
    """Rank-based normalization: replace each value with its rank / (N-1), mapped to [-1, 1]."""
    flat = rewards.flatten()
    N = flat.shape[0]
    if N == 1:
        return torch.zeros_like(rewards)
    # argsort of argsort gives rank for each element
    sorted_indices = torch.argsort(flat)
    ranks = torch.zeros_like(flat, dtype=torch.long)
    ranks[sorted_indices] = torch.arange(N, device=flat.device)
    # normalize ranks to [-1, 1]
    normalized = ranks.float() / (N - 1) * 2.0 - 1.0
    return normalized.reshape(rewards.shape)


def normalize_rewards(rewards: Tensor, method: str = "zscore", eps: float = 1e-8) -> Tensor:
    """Dispatch to appropriate normalization based on method string."""
    if method == "zscore":
        return normalize_rewards_zscore(rewards, eps=eps)
    elif method == "minmax":
        return normalize_rewards_minmax(rewards, eps=eps)
    elif method == "rank":
        return normalize_rewards_rank(rewards)
    elif method == "none":
        return rewards
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")


def compute_discounted_returns(rewards: Tensor, gamma: float = 1.0) -> Tensor:
    """Compute discounted return at each timestep G_t = r_t + gamma*r_{t+1} + ...

    Args:
        rewards: (B, T) per-step rewards
        gamma: discount factor

    Returns:
        (B, T) discounted returns
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    G = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        G = rewards[:, t] + gamma * G
        returns[:, t] = G
    return returns


def apply_potential_shaping(rewards: Tensor, potentials: Tensor, gamma: float = 1.0) -> Tensor:
    """Potential-based reward shaping: r'_t = r_t + gamma * phi_{t+1} - phi_t.

    Args:
        rewards: (B, T) base rewards
        potentials: (B, T+1) potential values including terminal potential
        gamma: discount factor

    Returns:
        (B, T) shaped rewards
    """
    phi_t = potentials[:, :-1]  # (B, T)
    phi_next = potentials[:, 1:]  # (B, T)
    return rewards + gamma * phi_next - phi_t


def length_penalty(
    token_ids: Tensor,
    min_length: int = 10,
    max_length: int = 200,
    coeff: float = 0.1,
) -> Tensor:
    """Penalize sequences outside [min_length, max_length].

    Args:
        token_ids: (B, S) token ids, pad=0
        min_length: minimum acceptable length (non-pad tokens)
        max_length: maximum acceptable length
        coeff: penalty coefficient

    Returns:
        (B,) penalties (non-negative values to be subtracted)
    """
    # Count non-pad tokens per sequence
    lengths = (token_ids != 0).sum(dim=-1).float()  # (B,)
    penalties = torch.zeros_like(lengths)

    too_short = lengths < min_length
    too_long = lengths > max_length

    penalties[too_short] = coeff * (min_length - lengths[too_short])
    penalties[too_long] = coeff * (lengths[too_long] - max_length)

    return penalties


def repetition_penalty(token_ids: Tensor, window: int = 20, coeff: float = 0.1) -> Tensor:
    """Penalize repeated n-grams in generated sequences.

    For each sequence: count unique 2-grams in a sliding window,
    penalty = coeff * (1 - unique_frac).

    Args:
        token_ids: (B, S) token ids
        window: sliding window size
        coeff: penalty coefficient

    Returns:
        (B,) penalties
    """
    B, S = token_ids.shape
    penalties = torch.zeros(B, device=token_ids.device, dtype=torch.float)

    for i in range(B):
        seq = token_ids[i].tolist()
        # Remove padding (0s at end)
        seq = [t for t in seq if t != 0]

        if len(seq) < 2:
            continue

        total_unique_frac = 0.0
        num_windows = 0

        for start in range(0, max(1, len(seq) - 1), window):
            chunk = seq[start : start + window]
            if len(chunk) < 2:
                continue
            bigrams = [(chunk[j], chunk[j + 1]) for j in range(len(chunk) - 1)]
            if not bigrams:
                continue
            unique_frac = len(set(bigrams)) / len(bigrams)
            total_unique_frac += unique_frac
            num_windows += 1

        if num_windows > 0:
            avg_unique_frac = total_unique_frac / num_windows
            penalties[i] = coeff * (1.0 - avg_unique_frac)

    return penalties


class EWMARewardNormalizer:
    """Exponentially weighted moving average normalizer for online normalization."""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self._mean: float = 0.0
        self._var: float = 1.0

    def update(self, rewards: Tensor) -> None:
        """Update running mean and variance with current batch using EWMA."""
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item() if rewards.numel() > 1 else 0.0

        # EWMA update
        self._mean = (1 - self.alpha) * self._mean + self.alpha * batch_mean
        self._var = (1 - self.alpha) * self._var + self.alpha * batch_var

    def normalize(self, rewards: Tensor) -> Tensor:
        """Normalize rewards using running stats."""
        return (rewards - self._mean) / math.sqrt(self._var + 1e-8)

    def reset(self) -> None:
        """Reset running stats to initial values."""
        self._mean = 0.0
        self._var = 1.0


class RewardShaper:
    """Combines multiple reward components and normalizations."""

    def __init__(self, config: RewardShapingConfig) -> None:
        self.config = config

    def shape(
        self,
        base_rewards: Tensor,
        token_ids: Tensor | None = None,
        potentials: Tensor | None = None,
    ) -> Tensor:
        """Apply normalization, clipping, and optional penalties/shaping.

        Args:
            base_rewards: (B,) or (B, T) base reward values
            token_ids: optional (B, S) token ids for length/repetition penalties
            potentials: optional (B, T+1) for potential-based shaping

        Returns:
            Shaped rewards of same shape as base_rewards
        """
        cfg = self.config
        rewards = normalize_rewards(base_rewards, method=cfg.normalize_method)
        rewards = rewards.clamp(-cfg.clip_range, cfg.clip_range)

        if token_ids is not None:
            lp = length_penalty(token_ids, coeff=cfg.penalty_coeff)  # (B,)
            rp = repetition_penalty(token_ids, coeff=cfg.penalty_coeff)  # (B,)
            penalties = lp + rp  # (B,)

            # Subtract penalties from rewards (broadcast if needed)
            if rewards.dim() == 2:
                rewards = rewards - penalties.unsqueeze(1)
            else:
                rewards = rewards - penalties

        if potentials is not None and cfg.use_potential_shaping:
            if rewards.dim() == 1:
                # Expand to (B, 1) for potential shaping, then squeeze
                rewards = rewards.unsqueeze(1)
                rewards = apply_potential_shaping(rewards, potentials, gamma=cfg.gamma)
                rewards = rewards.squeeze(1)
            else:
                rewards = apply_potential_shaping(rewards, potentials, gamma=cfg.gamma)

        return rewards

    def compute_shaped_returns(
        self,
        base_rewards: Tensor,
        gamma: float | None = None,
    ) -> Tensor:
        """Normalize rewards then compute discounted returns.

        Args:
            base_rewards: (B, T) per-step rewards
            gamma: discount factor; uses config.gamma if None

        Returns:
            (B, T) shaped discounted returns
        """
        if gamma is None:
            gamma = self.config.gamma

        rewards = normalize_rewards(base_rewards, method=self.config.normalize_method)
        rewards = rewards.clamp(-self.config.clip_range, self.config.clip_range)
        return compute_discounted_returns(rewards, gamma=gamma)
