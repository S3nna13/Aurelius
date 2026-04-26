"""GRPO: Group Relative Policy Optimization (arXiv:2402.03300, DeepSeek-R1).

Samples N responses per prompt, computes scalar rewards, normalizes within
the group to get advantages, then applies a clipped policy-gradient loss.
No reference model required.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# New canonical API (required by task spec)
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    n_samples: int = 8  # N responses per prompt
    beta: float = 0.01  # KL penalty coefficient
    clip_ratio: float = 0.2  # PPO clip epsilon
    kl_coef: float = 0.1  # Additional KL regularization coefficient
    max_new_tokens: int = 64  # Maximum tokens to generate per completion
    temperature: float = 1.0  # Sampling temperature


def sample_completions(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> list[torch.Tensor]:
    """Generate n_samples completions for a given prompt via temperature sampling.

    Args:
        model: The policy model with a generate() method.
        input_ids: (1, prompt_len) — prompt token ids.
        n_samples: Number of completions to generate.
        max_new_tokens: Maximum new tokens per completion.
        temperature: Sampling temperature.

    Returns:
        List of n_samples token tensors, each of shape (1, prompt_len + gen_len).
    """
    completions = []
    for _ in range(n_samples):
        with torch.no_grad():
            full_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        completions.append(full_ids)
    return completions


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Normalize rewards within a group to produce advantages.

    Args:
        rewards: (N,) — scalar reward for each rollout in the group.

    Returns:
        (N,) — normalized advantages: (r - mean) / (std + 1e-8).
        Returns zeros when all rewards are identical (std == 0).
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + 1e-8)


def grpo_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """Clipped policy gradient loss (GRPO/PPO style).

    Args:
        log_probs_new: (N,) — log probs under the current policy.
        log_probs_old: (N,) — log probs under the old/reference policy (detached).
        advantages: (N,) — group-normalized advantages.
        clip_ratio: PPO clipping epsilon (ε). Ratio is clipped to [1-ε, 1+ε].

    Returns:
        Scalar loss (negative clipped objective, to minimize).
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_start: int | None = None,
) -> torch.Tensor:
    """Compute sum of per-token log probs for a sequence.

    When response_start is provided, sums only over the response portion.
    When response_start is None, sums over all tokens (excluding the first,
    since the model predicts token[t+1] from token[t]).

    Args:
        model: The policy model.
        input_ids: (1, seq_len) — full token sequence.
        response_start: Optional index where the response begins. If None,
            log probs are summed over all positions (tokens 1 onward).

    Returns:
        Scalar — sum of log probs (always <= 0 for valid probability distributions).
    """
    _, logits, _ = model(input_ids)
    # Shift: logits[:, t] predicts input_ids[:, t+1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq_len-1, vocab)
    targets = input_ids[:, 1:]  # (1, seq_len-1)
    token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    if response_start is not None:
        # Sum only response tokens (offset by 1 for the shift)
        token_lp = token_lp[:, max(0, response_start - 1) :]

    return token_lp.sum(dim=-1).squeeze(0)  # scalar


class GRPOTrainer:
    """Train a model using GRPO on prompts scored by a reward function.

    Args:
        model: The policy model (AureliusTransformer or compatible nn.Module).
        ref_model: Reference model for KL regularization (may be None).
        config: GRPOConfig instance.
        optimizer: A pre-built optimizer for the policy model parameters.
        reward_fn: Callable mapping (completion_ids: torch.Tensor) -> float.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer,
        reward_fn: Callable[[torch.Tensor], float],
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer
        self.reward_fn = reward_fn

    def train_step(self, prompt_ids: torch.Tensor) -> dict:
        """One GRPO training step: sample, score, advantage, update.

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.

        Returns:
            dict with keys: "loss" (float), "mean_reward" (float),
            "advantage_std" (float).
        """
        cfg = self.config

        # --- Rollout: generate n_samples completions ---------------------------
        completions = sample_completions(
            self.model, prompt_ids, cfg.n_samples, cfg.max_new_tokens, cfg.temperature
        )

        # --- Score completions via reward_fn -----------------------------------
        rewards = torch.tensor([float(self.reward_fn(c)) for c in completions], dtype=torch.float32)

        # --- Compute group-relative advantages ---------------------------------
        advantages = group_relative_advantages(rewards)

        # --- Collect old log probs (detached, under rollout policy) ------------
        old_log_probs_list = []
        with torch.no_grad():
            for c in completions:
                lp = compute_sequence_log_probs(self.model, c, response_start=prompt_ids.shape[1])
                old_log_probs_list.append(lp.detach())
        old_log_probs = torch.stack(old_log_probs_list)

        # --- Recompute log probs under current (training) policy ---------------
        self.model.train()
        new_log_probs_list = []
        for c in completions:
            lp = compute_sequence_log_probs(self.model, c, response_start=prompt_ids.shape[1])
            new_log_probs_list.append(lp)
        new_log_probs = torch.stack(new_log_probs_list)

        # --- Policy gradient loss ----------------------------------------------
        loss = grpo_policy_loss(new_log_probs, old_log_probs, advantages, cfg.clip_ratio)

        # --- Optional KL regularization ----------------------------------------
        if cfg.kl_coef > 0:
            kl_penalty = (new_log_probs - old_log_probs).mean()
            loss = loss + cfg.kl_coef * kl_penalty

        # --- Optimizer step ----------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "advantage_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
        }


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within a group to get advantages (legacy name).

    Args:
        rewards: (N,) — scalar reward for each rollout.

    Returns:
        (N,) — group-normalized advantages: (r - mean) / (std + eps).
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std()
    return (rewards - rewards.mean()) / (std + eps)


def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute clipped GRPO/PPO policy gradient loss (legacy name).

    Args:
        log_probs_new: (N,) — log probs under current policy.
        log_probs_old: (N,) — log probs under old policy (detached).
        advantages: (N,) — group-normalized advantages.
        clip_eps: PPO clipping epsilon.

    Returns:
        Scalar loss (to minimize).
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss
