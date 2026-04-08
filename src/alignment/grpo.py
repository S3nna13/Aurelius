"""GRPO: Group Relative Policy Optimization (arXiv:2402.03300, DeepSeek-R1).

Samples N responses per prompt, computes scalar rewards, normalizes within
the group to get advantages, then applies a clipped policy-gradient loss.
No reference model required.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GRPOConfig:
    num_rollouts: int = 8          # N responses per prompt
    clip_eps: float = 0.2          # PPO clip epsilon
    kl_coef: float = 0.01          # KL penalty coefficient (optional regularization)
    advantage_eps: float = 1e-8    # denominator stability
    learning_rate: float = 1e-6
    num_steps: int = 100           # training iterations
    max_new_tokens: int = 256
    temperature: float = 0.8
    batch_size: int = 4            # prompts per batch
    seed: int = 42


def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Compute per-token log probs for the response portion of input_ids.

    Args:
        model: The policy model.
        input_ids: (1, total_len) — prompt + response tokens concatenated.
        response_start: Index where the response begins in input_ids.

    Returns:
        Scalar — sum of log probs over response tokens.
    """
    _, logits, _ = model(input_ids)
    # Shift: logits[:, :-1] predicts input_ids[:, 1:]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]  # (1, total_len - 1)
    token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, total_len-1)
    # Sum only over response tokens (offset by 1 due to shift)
    response_lp = token_lp[:, response_start - 1:].sum(dim=-1)  # (1,)
    return response_lp.squeeze(0)  # scalar


def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute clipped GRPO/PPO policy gradient loss.

    Args:
        log_probs_new: (N,) — log probs of responses under current policy.
        log_probs_old: (N,) — log probs under old policy (detached, from rollout).
        advantages: (N,) — group-normalized advantages.
        clip_eps: PPO clipping epsilon.

    Returns:
        Scalar loss (to minimize).
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within a group to get advantages.

    Args:
        rewards: (N,) — scalar reward for each rollout.

    Returns:
        (N,) — group-normalized advantages: (r - mean) / (std + eps).
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    std = rewards.std()
    return (rewards - rewards.mean()) / (std + eps)


class GRPOTrainer:
    """Train a model using GRPO on a set of prompts with a reward function.

    Args:
        model: The policy model (AureliusTransformer or any nn.Module).
        reward_fn: Callable mapping (prompt: str, response: str) -> float.
        cfg: GRPO configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[str, str], float],
        cfg: GRPOConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.cfg = cfg or GRPOConfig()
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
        )

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        prompt_text: str,
        tokenizer,
    ) -> tuple[list[torch.Tensor], list[float], list[torch.Tensor]]:
        """Generate N responses, score them, and collect old log probs.

        Returns:
            response_id_list: list of N response token tensors (each variable length)
            rewards: list of N scalar rewards
            old_log_probs: list of N log prob scalars (detached)
        """
        self.model.eval()
        response_id_list = []
        rewards = []
        old_log_probs = []

        with torch.no_grad():
            for _ in range(self.cfg.num_rollouts):
                full_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                )
                response_ids = full_ids[:, prompt_ids.shape[1]:]
                response_text = tokenizer.decode(response_ids[0].tolist())
                reward = float(self.reward_fn(prompt_text, response_text))

                lp = compute_sequence_log_probs(self.model, full_ids, prompt_ids.shape[1])

                response_id_list.append(response_ids)
                rewards.append(reward)
                old_log_probs.append(lp.detach())

        return response_id_list, rewards, old_log_probs

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_text: str,
        tokenizer,
    ) -> dict[str, float]:
        """One GRPO training step: rollout + advantage + loss + update."""
        response_ids_list, rewards, old_log_probs_list = self.rollout(
            prompt_ids, prompt_text, tokenizer
        )

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = compute_advantages(rewards_t, eps=self.cfg.advantage_eps)
        old_log_probs_t = torch.stack(old_log_probs_list)

        # Recompute log probs under current policy
        self.model.train()
        new_log_probs = []
        for response_ids in response_ids_list:
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            lp = compute_sequence_log_probs(self.model, full_ids, prompt_ids.shape[1])
            new_log_probs.append(lp)
        new_log_probs_t = torch.stack(new_log_probs)

        loss = grpo_loss(new_log_probs_t, old_log_probs_t, advantages, self.cfg.clip_eps)

        # Optional KL regularization: penalizes deviation from rollout policy.
        # This also ensures a non-zero gradient when all advantages are identical.
        if self.cfg.kl_coef > 0:
            kl = (new_log_probs_t - old_log_probs_t).mean()
            loss = loss + self.cfg.kl_coef * kl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards_t.mean().item(),
            "reward_std": rewards_t.std().item(),
        }
