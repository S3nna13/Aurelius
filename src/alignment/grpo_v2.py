"""GRPO v2: Enhanced Group Relative Policy Optimization.

Improvements over basic GRPO (grpo.py):
- Dr. GRPO (DeepSeek 2025): unbiased std estimation with Bessel correction
- Clip-higher variant: asymmetric PPO clipping (clip_high > clip_low)
- Reference-free mode: no KL penalty, no ref model required
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOv2Config:
    group_size: int = 8  # G: completions per question
    clip_eps_low: float = 0.2  # lower clip bound: ratio >= 1 - clip_eps_low
    clip_eps_high: float = 0.2  # upper clip bound: ratio <= 1 + clip_eps_high
    beta: float = 0.04  # KL penalty weight
    use_dr_grpo: bool = True  # Dr. GRPO: unbiased std (ddof=1)
    use_reference_free: bool = False  # skip KL penalty entirely
    min_group_for_std: int = 2  # minimum G for std normalization
    advantage_eps: float = 1e-8  # numerical stability in denominator


def compute_grpo_advantages(
    rewards: torch.Tensor,  # (G,) rewards for one question's completions
    config: GRPOv2Config,
) -> torch.Tensor:
    """Compute normalized advantages for one group.

    Standard GRPO:  A_i = (r_i - mean) / (std(ddof=0) + eps)
    Dr. GRPO:       A_i = (r_i - mean) / (std(ddof=1) + eps)

    Edge cases:
    - G == 1: return zeros (no contrast possible)
    - G < min_group_for_std: return centered rewards only (no std division)
    - all rewards identical: std == 0, return zeros after centering

    Returns: (G,) advantages
    """
    G = rewards.numel()

    if G == 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    centered = rewards - mean

    if G < config.min_group_for_std:
        return centered

    if config.use_dr_grpo:
        std = rewards.std(correction=1)
    else:
        std = rewards.std(correction=0)

    return centered / (std + config.advantage_eps)


def grpo_loss(
    log_probs: torch.Tensor,  # (G,) policy log probs per completion
    ref_log_probs: torch.Tensor | None,  # (G,) reference log probs (None = reference-free)
    rewards: torch.Tensor,  # (G,) rewards
    config: GRPOv2Config,
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO loss for one group.

    Steps:
    1. Compute normalized advantages.
    2. Compute PPO-clip policy loss (on-policy: ratio = 1).
    3. Optionally add KL penalty.
    4. Return (total_loss, metrics_dict).

    Metrics keys: 'policy_loss', 'kl_loss', 'mean_reward', 'mean_advantage', 'reward_std'
    """
    advantages = compute_grpo_advantages(rewards, config)

    # On-policy single-step: ratio = exp(log_probs - log_probs.detach()) = 1
    # Using this form keeps the gradient w.r.t. log_probs flowing through.
    ratio = torch.exp(log_probs - log_probs.detach())

    clip_low = 1.0 - config.clip_eps_low
    clip_high = 1.0 + config.clip_eps_high

    clipped = ratio.clamp(clip_low, clip_high)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

    if config.use_reference_free or ref_log_probs is None:
        kl_loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
    else:
        kl_loss = config.beta * (log_probs - ref_log_probs).mean()

    total_loss = policy_loss + kl_loss

    n = rewards.numel()
    corr = 1 if n > 1 else 0
    metrics = {
        "policy_loss": policy_loss.detach().item(),
        "kl_loss": kl_loss.detach().item(),
        "mean_reward": rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "reward_std": rewards.std(correction=corr).item(),
    }

    return total_loss, metrics


def clip_higher_ratio(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_low: float,
    clip_high: float,
) -> torch.Tensor:
    """Asymmetric PPO clipping element-wise surrogate.

    Standard PPO uses clip_low = 1 - eps, clip_high = 1 + eps (symmetric).
    Clip-higher sets clip_high > clip_low, allowing larger positive updates
    when advantage is high.

    Returns: element-wise min(ratio * adv, clipped_ratio * adv)
    """
    clipped = ratio.clamp(clip_low, clip_high)
    return torch.min(ratio * advantage, clipped * advantage)


class GRPOv2Trainer:
    """Enhanced GRPO trainer with Dr. GRPO corrections and clip-higher support.

    Args:
        model: AureliusTransformer (policy model).
        ref_model: AureliusTransformer or None (reference; None = reference-free).
        optimizer: torch.optim.Optimizer
        reward_fn: callable(completion: str) -> float
        tokenizer_encode: callable(text: str) -> list[int]
        tokenizer_decode: callable(ids: list[int]) -> str
        config: GRPOv2Config
        max_seq_len: int
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        reward_fn: Callable[[str], float],
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: GRPOv2Config | None = None,
        max_seq_len: int = 64,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config or GRPOv2Config()
        self.max_seq_len = max_seq_len

    def generate_completions(
        self,
        prompt_ids: list[int],
        n: int,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Generate n completions from prompt_ids with temperature sampling.

        Returns a list of n token-id lists (completion tokens only).
        """
        self.model.eval()
        completions: list[list[int]] = []
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        max_new = max(1, self.max_seq_len - len(prompt_ids))

        with torch.no_grad():
            for _ in range(n):
                full_ids = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=max_new,
                    temperature=temperature,
                )
                completion_ids = full_ids[0, len(prompt_ids) :].tolist()
                completions.append(completion_ids)

        return completions

    def score_completions(
        self,
        prompt: str,
        completions: list[str],
    ) -> list[float]:
        """Apply reward_fn to each completion string. Returns list of floats."""
        return [float(self.reward_fn(c)) for c in completions]

    def _compute_log_probs(
        self,
        model: nn.Module,
        prompt_ids: list[int],
        completion_ids: list[int],
    ) -> torch.Tensor:
        """Compute sum of log probs over completion tokens given prompt context."""
        all_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([all_ids], dtype=torch.long)

        _, logits, _ = model(input_tensor)
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, L-1, V)
        targets = input_tensor[:, 1:]  # (1, L-1)
        token_lp = lp.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

        completion_start = len(prompt_ids) - 1
        response_lp = token_lp[:, completion_start:].sum(dim=-1)  # (1,)
        return response_lp.squeeze(0)  # scalar

    def train_step(self, prompts: list[str]) -> dict:
        """Run one GRPO training step over a batch of prompts.

        Returns: {'loss': float, 'mean_reward': float, 'reward_std': float}
        """
        G = self.config.group_size
        all_losses: list[torch.Tensor] = []
        all_rewards: list[float] = []

        for prompt in prompts:
            prompt_ids = self.tokenizer_encode(prompt)

            completion_id_lists = self.generate_completions(prompt_ids, n=G)
            completion_strings = [self.tokenizer_decode(ids) for ids in completion_id_lists]

            rewards_list = self.score_completions(prompt, completion_strings)
            rewards = torch.tensor(rewards_list, dtype=torch.float32)
            all_rewards.extend(rewards_list)

            self.model.train()
            log_probs_list: list[torch.Tensor] = []
            for cids in completion_id_lists:
                lp = self._compute_log_probs(self.model, prompt_ids, cids)
                log_probs_list.append(lp)
            log_probs = torch.stack(log_probs_list)

            ref_log_probs: torch.Tensor | None = None
            if not self.config.use_reference_free and self.ref_model is not None:
                with torch.no_grad():
                    ref_lps: list[torch.Tensor] = []
                    for cids in completion_id_lists:
                        lp = self._compute_log_probs(self.ref_model, prompt_ids, cids)
                        ref_lps.append(lp)
                    ref_log_probs = torch.stack(ref_lps)

            loss, _ = grpo_loss(log_probs, ref_log_probs, rewards, self.config)
            all_losses.append(loss)

        total_loss = torch.stack(all_losses).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        all_rewards_t = torch.tensor(all_rewards, dtype=torch.float32)
        n = all_rewards_t.numel()
        corr = 1 if n > 1 else 0
        return {
            "loss": total_loss.item(),
            "mean_reward": all_rewards_t.mean().item(),
            "reward_std": all_rewards_t.std(correction=corr).item(),
        }
