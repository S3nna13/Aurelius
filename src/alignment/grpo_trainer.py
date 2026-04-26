"""GRPO Trainer: Group Relative Policy Optimization with dataclass-based sample tracking.

Implements a dataclass-oriented GRPO pipeline where each sample is tracked as a
GroupSample object with its prompt, response, log-prob, reward, and advantage.
This complements grpo.py (string/tokenizer-based), grpo_advanced.py (per-token
tensor-based), and grpo_v2.py (Dr. GRPO + clip-higher variants).

Key difference: uses GroupSample dataclass for structured rollout storage and
supports both policy + KL + entropy terms in the loss, plus an evaluate() method.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    n_group: int = 8
    clip_ratio: float = 0.2
    kl_coeff: float = 0.04
    max_new_tokens: int = 128
    temperature: float = 1.0
    normalize_rewards: bool = True
    entropy_bonus: float = 0.01


# ---------------------------------------------------------------------------
# GroupSample dataclass
# ---------------------------------------------------------------------------


@dataclass
class GroupSample:
    prompt_ids: list
    response_ids: list
    log_prob: float
    reward: float
    advantage: float = 0.0


# ---------------------------------------------------------------------------
# sample_group
# ---------------------------------------------------------------------------


def sample_group(
    model: nn.Module,
    prompt_ids: list,
    n_group: int,
    max_new_tokens: int,
    temperature: float,
) -> list:
    """Sample n_group responses for prompt_ids using temperature sampling.

    Returns list of GroupSample with reward=0.0 and advantage=0.0.
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for _ in range(n_group):
            cur_ids = torch.tensor([prompt_ids], dtype=torch.long)
            response_tokens = []
            total_log_prob = 0.0

            for _ in range(max_new_tokens):
                _, logits, _ = model(cur_ids)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token.item())
                log_prob_t = float(F.log_softmax(next_logits, dim=-1)[0, token_id].item())
                response_tokens.append(token_id)
                total_log_prob += log_prob_t
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

            samples.append(
                GroupSample(
                    prompt_ids=list(prompt_ids),
                    response_ids=response_tokens,
                    log_prob=total_log_prob,
                    reward=0.0,
                    advantage=0.0,
                )
            )

    return samples


# ---------------------------------------------------------------------------
# compute_group_advantages
# ---------------------------------------------------------------------------


def compute_group_advantages(samples: list, normalize: bool = True) -> list:
    """Compute group-relative advantages.

    advantage_i = reward_i - mean(rewards)
    If normalize=True: divide by std(rewards) + 1e-8.
    """
    rewards = [s.reward for s in samples]
    n = len(rewards)

    if n == 0:
        return []

    mean_r = sum(rewards) / n

    if n == 1:
        return [
            GroupSample(
                prompt_ids=s.prompt_ids,
                response_ids=s.response_ids,
                log_prob=s.log_prob,
                reward=s.reward,
                advantage=0.0,
            )
            for s in samples
        ]

    raw_advantages = [r - mean_r for r in rewards]

    if normalize:
        variance = sum((r - mean_r) ** 2 for r in rewards) / n
        std_r = math.sqrt(variance) + 1e-8
        advantages = [a / std_r for a in raw_advantages]
    else:
        advantages = raw_advantages

    return [
        GroupSample(
            prompt_ids=s.prompt_ids,
            response_ids=s.response_ids,
            log_prob=s.log_prob,
            reward=s.reward,
            advantage=adv,
        )
        for s, adv in zip(samples, advantages)
    ]


# ---------------------------------------------------------------------------
# Helper: compute sequence log-prob
# ---------------------------------------------------------------------------


def _compute_response_log_prob(model: nn.Module, sample: GroupSample) -> Tensor:
    """Compute sum of log-probs over response tokens under model."""
    all_ids = sample.prompt_ids + sample.response_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long)
    prompt_len = len(sample.prompt_ids)

    _, logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    response_lp = token_lp[:, prompt_len - 1 :].sum(dim=-1)
    return response_lp.squeeze(0)


# ---------------------------------------------------------------------------
# compute_grpo_loss
# ---------------------------------------------------------------------------


def compute_grpo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    samples: list,
    config: GRPOConfig,
) -> tuple:
    """Compute GRPO objective: policy loss + KL penalty - entropy bonus.

    Returns (total_loss, metrics_dict) where metrics_dict has keys:
    "policy_loss", "kl", "entropy", "clip_fraction".
    """
    # Reference log probs (no grad)
    ref_log_probs_list = []
    with torch.no_grad():
        ref_model.eval()
        for s in samples:
            ref_lp = _compute_response_log_prob(ref_model, s)
            ref_log_probs_list.append(ref_lp.item())

    # Current policy log probs (with grad)
    model.train()
    current_log_probs = []
    for s in samples:
        cur_lp = _compute_response_log_prob(model, s)
        current_log_probs.append(cur_lp)

    cur_lp_tensor = torch.stack(current_log_probs)
    old_lp_tensor = torch.tensor([s.log_prob for s in samples], dtype=torch.float32)
    ref_lp_tensor = torch.tensor(ref_log_probs_list, dtype=torch.float32)
    adv_tensor = torch.tensor([s.advantage for s in samples], dtype=torch.float32)

    # Policy gradient with clipping
    ratio = torch.exp(cur_lp_tensor - old_lp_tensor)
    lo = 1.0 - config.clip_ratio
    hi = 1.0 + config.clip_ratio
    clipped_ratio = torch.clamp(ratio, lo, hi)
    policy_loss = -torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor).mean()

    # KL penalty
    kl_penalty = (cur_lp_tensor - ref_lp_tensor).mean() * config.kl_coeff

    # Entropy bonus
    entropy = -cur_lp_tensor.mean() * config.entropy_bonus

    total_loss = policy_loss + kl_penalty - entropy

    # Clip fraction
    with torch.no_grad():
        clipped_mask = (ratio < lo) | (ratio > hi)
        clip_fraction = float(clipped_mask.float().mean().item())

    metrics = {
        "policy_loss": float(policy_loss.detach().item()),
        "kl": float(kl_penalty.detach().item()),
        "entropy": float(entropy.detach().item()),
        "clip_fraction": clip_fraction,
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Train a model with GRPO using GroupSample-based rollout tracking.

    Uses GRPOConfig with n_group, clip_ratio, kl_coeff, entropy_bonus fields.
    Complements existing trainers by adding structured sample storage and
    an evaluate() method for reward assessment without gradient updates.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable,
        config: GRPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

    def train_step(self, prompt_ids: list) -> dict:
        """Run one full GRPO training step.

        Returns metrics dict with: loss, mean_reward, mean_advantage,
        policy_loss, kl, entropy, clip_fraction.
        """
        cfg = self.config

        # 1. Sample group
        samples = sample_group(
            self.model,
            prompt_ids,
            n_group=cfg.n_group,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )

        # 2. Score with reward_fn
        scored = []
        for s in samples:
            scored.append(
                GroupSample(
                    prompt_ids=s.prompt_ids,
                    response_ids=s.response_ids,
                    log_prob=s.log_prob,
                    reward=float(self.reward_fn(s.response_ids)),
                    advantage=s.advantage,
                )
            )
        samples = scored

        # 3. Compute advantages
        samples = compute_group_advantages(samples, normalize=cfg.normalize_rewards)

        # 4. Compute GRPO loss
        loss, loss_metrics = compute_grpo_loss(self.model, self.ref_model, samples, cfg)

        # 5. Backward + step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        rewards = [s.reward for s in samples]
        advantages = [s.advantage for s in samples]
        n = len(rewards)

        metrics = {
            "loss": loss.item(),
            "mean_reward": sum(rewards) / n,
            "mean_advantage": sum(advantages) / n,
        }
        metrics.update(loss_metrics)
        return metrics

    def evaluate(self, prompts: list, n_eval=None) -> dict:
        """Evaluate mean reward over prompts without gradient updates.

        Returns dict with "mean_reward" key.
        """
        cfg = self.config
        n_samples = n_eval if n_eval is not None else cfg.n_group
        all_rewards = []

        self.model.eval()
        with torch.no_grad():
            for p_ids in prompts:
                group = sample_group(
                    self.model,
                    p_ids,
                    n_group=n_samples,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                )
                for s in group:
                    r = float(self.reward_fn(s.response_ids))
                    all_rewards.append(r)

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        return {"mean_reward": mean_reward}
