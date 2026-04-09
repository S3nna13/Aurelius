"""RLVR: Reinforcement Learning with Verifiable Rewards for math/code correctness."""
from __future__ import annotations

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

Tensor = torch.Tensor


@dataclass
class RLVRConfig:
    n_samples: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    kl_coeff: float = 0.04
    clip_ratio: float = 0.2
    normalize_rewards: bool = True


class VerifiableReward:
    """Base class for verifiable reward functions.

    Returns reward in {0.0, 0.5, 1.0}: 0=wrong, 0.5=partial, 1=correct.
    """

    def __call__(self, prompt: str, completion: str, ground_truth: str) -> float:
        raise NotImplementedError


class MathReward(VerifiableReward):
    """Extract final numeric answer from completion and compare to ground_truth."""

    def __call__(self, prompt: str, completion: str, ground_truth: str) -> float:
        numbers = re.findall(r'-?\d+\.?\d*', completion)
        if not numbers:
            return 0.0
        try:
            predicted = float(numbers[-1])
            truth = float(ground_truth)
        except ValueError:
            return 0.0

        if abs(predicted - truth) < 1e-6:
            return 1.0
        if truth != 0.0 and abs(predicted - truth) / abs(truth) <= 0.10:
            return 0.5
        if truth == 0.0 and abs(predicted - truth) <= 0.10:
            return 0.5
        return 0.0


class FormatReward(VerifiableReward):
    """Reward based on format compliance (LaTeX boxed answer + reasoning length)."""

    def __call__(self, prompt: str, completion: str, ground_truth: str) -> float:
        score = 0.0
        if r'\boxed{' in completion:
            score += 0.5
        if len(completion) > 50:
            score += 0.5
        return min(score, 1.0)


class CompositeReward(VerifiableReward):
    """Weighted combination of multiple reward functions."""

    def __init__(self, rewards: list[tuple[VerifiableReward, float]]) -> None:
        self.rewards = rewards

    def __call__(self, prompt: str, completion: str, ground_truth: str) -> float:
        total_weight = sum(w for _, w in self.rewards)
        if total_weight == 0.0:
            return 0.0
        weighted_sum = sum(fn(prompt, completion, ground_truth) * w for fn, w in self.rewards)
        return weighted_sum / total_weight


def sample_completions(
    model: nn.Module,
    input_ids: Tensor,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Sample n_samples completions with temperature.

    Returns:
        completions: (n_samples, max_new_tokens) sampled token ids.
        log_probs:   (n_samples, max_new_tokens) per-token log probs.
    """
    model.eval()
    all_ids: list[Tensor] = []
    all_log_probs: list[Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            cur_ids = input_ids.clone()
            step_ids: list[int] = []
            step_lps: list[float] = []

            for _ in range(max_new_tokens):
                _, logits, _ = model(cur_ids)
                next_logits = logits[:, -1, :]  # (1, V)
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                log_probs_dist = F.log_softmax(next_logits, dim=-1)
                probs = torch.exp(log_probs_dist)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                token_id = int(next_token.item())
                token_lp = float(log_probs_dist[0, token_id].item())
                step_ids.append(token_id)
                step_lps.append(token_lp)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

            all_ids.append(torch.tensor(step_ids, dtype=torch.long))
            all_log_probs.append(torch.tensor(step_lps, dtype=torch.float32))

    completions = torch.stack(all_ids, dim=0)        # (n_samples, max_new_tokens)
    log_probs = torch.stack(all_log_probs, dim=0)    # (n_samples, max_new_tokens)
    return completions, log_probs


def compute_rlvr_loss(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    rewards: Tensor,
    config: RLVRConfig,
) -> tuple[Tensor, dict]:
    """GRPO-style loss with PPO clipping and KL regularization.

    Args:
        log_probs:     (n_samples, T) log probs under current policy.
        ref_log_probs: (n_samples, T) log probs under reference policy.
        rewards:       (n_samples,) scalar reward for each completion.
        config:        RLVRConfig.

    Returns:
        (loss, metrics_dict)
    """
    if config.normalize_rewards:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards

    # Sum log probs over token dimension
    lp_sum = log_probs.sum(dim=-1)
    ref_lp_sum = ref_log_probs.sum(dim=-1)

    ratio = torch.exp(lp_sum - ref_lp_sum)  # (n_samples,)
    clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)

    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl_loss = (log_probs - ref_log_probs).mean()
    total = policy_loss + config.kl_coeff * kl_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "mean_reward": rewards.mean().item(),
    }
    return total, metrics


class RLVRTrainer:
    """Train a policy model with Reinforcement Learning from Verifiable Rewards."""

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_fn: VerifiableReward,
        config: RLVRConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

    def train_step(
        self,
        prompt_ids: Tensor,
        prompt_text: str,
        ground_truth: str,
    ) -> dict:
        """Run one RLVR training step.

        Returns:
            dict with keys: loss, mean_reward, n_samples.
        """
        cfg = self.config

        # 1. Sample completions from policy (no grad)
        completion_ids, _ = sample_completions(
            self.policy_model,
            prompt_ids,
            n_samples=cfg.n_samples,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )  # (n_samples, T)

        # 2. Decode completions to text using byte-level decode
        completions_text: list[str] = []
        for i in range(cfg.n_samples):
            ids = completion_ids[i].tolist()
            text = bytes(tok % 256 for tok in ids).decode(errors='replace')
            completions_text.append(text)

        # 3. Score with reward_fn
        reward_list = [
            float(self.reward_fn(prompt_text, comp, ground_truth))
            for comp in completions_text
        ]
        rewards = torch.tensor(reward_list, dtype=torch.float32)

        # 4. Compute ref log probs (no grad)
        with torch.no_grad():
            self.ref_model.eval()
            _, ref_log_probs = sample_completions(
                self.ref_model,
                prompt_ids,
                n_samples=cfg.n_samples,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )

        # 5. Recompute policy log probs with gradient
        self.policy_model.train()
        policy_log_probs = self._recompute_log_probs(prompt_ids, completion_ids)

        loss, metrics = compute_rlvr_loss(policy_log_probs, ref_log_probs, rewards, cfg)

        # 6. Backward + step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": metrics["mean_reward"],
            "n_samples": cfg.n_samples,
        }

    def _recompute_log_probs(
        self,
        prompt_ids: Tensor,
        completion_ids: Tensor,
    ) -> Tensor:
        """Recompute per-token log probs for completions with gradient."""
        n_samples, T = completion_ids.shape
        prompt_len = prompt_ids.shape[1]

        prompt_expanded = prompt_ids.expand(n_samples, -1)
        full_ids = torch.cat([prompt_expanded, completion_ids], dim=1)

        _, logits, _ = self.policy_model(full_ids)

        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)

        completion_lp = log_probs_all[:, prompt_len - 1: prompt_len - 1 + T, :]

        token_lp = completion_lp.gather(
            2, completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_lp
