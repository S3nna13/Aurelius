"""GRPO Advanced: DeepSeekMath-style group relative policy optimization with per-token advantages."""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class GRPOAdvancedConfig:
    n_group: int = 8  # group size (samples per prompt)
    epsilon: float = 0.2  # PPO clip ratio
    beta: float = 0.04  # KL penalty coefficient
    max_new_tokens: int = 32
    temperature: float = 1.0
    normalize_advantages: bool = True


def sample_group(
    model: nn.Module,
    input_ids: Tensor,
    n_group: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Sample n_group completions from model for a single prompt.

    Args:
        model: Policy model (AureliusTransformer).
        input_ids: (1, prompt_len) prompt token ids.
        n_group: Number of completions to sample.
        max_new_tokens: Number of tokens to generate per completion.
        temperature: Sampling temperature.

    Returns:
        group_ids: (n_group, max_new_tokens) sampled token ids.
        group_log_probs: (n_group, max_new_tokens) log probs of each token under current policy.
    """
    model.eval()
    all_ids: list[Tensor] = []
    all_log_probs: list[Tensor] = []

    with torch.no_grad():
        for _ in range(n_group):
            cur_ids = input_ids.clone()
            step_ids: list[int] = []
            step_lps: list[float] = []

            for _ in range(max_new_tokens):
                _, logits, _ = model(cur_ids)
                next_logits = logits[:, -1, :]  # (1, V)
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                log_probs_dist = F.log_softmax(next_logits, dim=-1)  # (1, V)
                probs = torch.exp(log_probs_dist)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                token_id = int(next_token.item())
                token_lp = float(log_probs_dist[0, token_id].item())
                step_ids.append(token_id)
                step_lps.append(token_lp)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

            all_ids.append(torch.tensor(step_ids, dtype=torch.long))
            all_log_probs.append(torch.tensor(step_lps, dtype=torch.float32))

    group_ids = torch.stack(all_ids, dim=0)  # (n_group, max_new_tokens)
    group_log_probs = torch.stack(all_log_probs, dim=0)  # (n_group, max_new_tokens)
    return group_ids, group_log_probs


def compute_group_advantages(rewards: Tensor, normalize: bool = True) -> Tensor:
    """Compute group-relative advantages.

    Args:
        rewards: (n_group,) scalar reward for each completion.
        normalize: If True, divide by std + 1e-8.

    Returns:
        advantages: (n_group,) group-relative advantages.
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean()
    advantages = rewards - mean

    if normalize:
        std = rewards.std()
        advantages = advantages / (std + 1e-8)

    return advantages


def per_token_advantage(advantages: Tensor, seq_len: int) -> Tensor:
    """Expand scalar per-sequence advantages to per-token level.

    Args:
        advantages: (n_group,) one advantage per completion.
        seq_len: Token sequence length T.

    Returns:
        (n_group, seq_len) — each token gets the same advantage as its sequence.
    """
    return advantages.unsqueeze(1).expand(-1, seq_len)


def grpo_clipped_loss(
    log_probs_policy: Tensor,
    log_probs_ref: Tensor,
    advantages: Tensor,
    epsilon: float,
    beta: float,
) -> Tensor:
    """Compute the clipped GRPO loss with KL regularization.

    Args:
        log_probs_policy: (n_group, T) log probs under current policy.
        log_probs_ref: (n_group, T) log probs under reference policy.
        advantages: (n_group, T) per-token advantages.
        epsilon: PPO clip ratio.
        beta: KL penalty coefficient.

    Returns:
        Scalar loss.
    """
    ratio = torch.exp(log_probs_policy - log_probs_ref)
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl_penalty = (log_probs_policy - log_probs_ref).mean()
    return policy_loss + beta * kl_penalty


class GRPOAdvancedTrainer:
    """Trainer implementing DeepSeekMath-style GRPO with per-token advantages and reference KL.

    Args:
        policy_model: The policy model being trained.
        ref_model: The frozen reference model for KL regularization.
        reward_fn: Callable mapping (completion_ids: Tensor) -> float.
        config: GRPOAdvancedConfig.
        optimizer: torch.optim.Optimizer for policy_model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[Tensor], float],
        config: GRPOAdvancedConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

    def _compute_token_log_probs(
        self,
        model: nn.Module,
        input_ids: Tensor,
        completion_ids: Tensor,
    ) -> Tensor:
        """Compute per-token log probs for completion tokens.

        Args:
            model: Model to use.
            input_ids: (1, prompt_len) prompt.
            completion_ids: (n_group, T) completion token ids.

        Returns:
            (n_group, T) log probs.
        """
        n_group, T = completion_ids.shape
        prompt_len = input_ids.shape[1]

        # Expand prompt to n_group
        prompt_expanded = input_ids.expand(n_group, -1)  # (n_group, prompt_len)
        full_ids = torch.cat([prompt_expanded, completion_ids], dim=1)  # (n_group, prompt_len+T)

        _, logits, _ = model(full_ids)  # (n_group, prompt_len+T, V)

        # log probs for next-token prediction (shifted)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (n_group, prompt_len+T-1, V)

        # completion tokens correspond to positions prompt_len-1 .. prompt_len+T-2 in log_probs
        completion_log_probs = log_probs[
            :, prompt_len - 1 : prompt_len - 1 + T, :
        ]  # (n_group, T, V)

        # Gather the log probs for the actual tokens
        token_lp = completion_log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(
            -1
        )  # (n_group, T)

        return token_lp

    def train_step(self, input_ids: Tensor) -> dict:
        """Run one GRPO advanced training step.

        Args:
            input_ids: (1, prompt_len) prompt token ids.

        Returns:
            dict with keys: "loss", "mean_reward", "reward_std", "mean_advantage".
        """
        cfg = self.config

        # 1. Sample n_group completions
        group_ids, _ = sample_group(
            self.policy_model,
            input_ids,
            n_group=cfg.n_group,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )  # (n_group, T)

        # 2. Compute rewards
        rewards_list = [float(self.reward_fn(group_ids[g])) for g in range(cfg.n_group)]
        rewards = torch.tensor(rewards_list, dtype=torch.float32)

        # 3. Compute group advantages
        adv_scalar = compute_group_advantages(rewards, normalize=cfg.normalize_advantages)

        # 4. Expand to per-token advantages
        T = group_ids.shape[1]
        adv_token = per_token_advantage(adv_scalar, T)  # (n_group, T)

        # 5. Re-compute log probs from current policy (with gradient)
        self.policy_model.train()
        log_probs_policy = self._compute_token_log_probs(
            self.policy_model, input_ids, group_ids
        )  # (n_group, T)

        # 6. Compute ref log probs (no grad)
        with torch.no_grad():
            self.ref_model.eval()
            log_probs_ref = self._compute_token_log_probs(
                self.ref_model, input_ids, group_ids
            )  # (n_group, T)

        # 7. Compute clipped loss
        loss = grpo_clipped_loss(
            log_probs_policy,
            log_probs_ref,
            adv_token,
            epsilon=cfg.epsilon,
            beta=cfg.beta,
        )

        # 8. Backward + step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        n = rewards.numel()
        corr = 1 if n > 1 else 0
        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std(correction=corr).item(),
            "mean_advantage": adv_scalar.mean().item(),
        }
