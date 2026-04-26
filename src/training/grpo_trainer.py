"""GRPO (Group Relative Policy Optimization) trainer.

GRPO eliminates the critic/value network by using group-relative rewards to
estimate advantages:
- GroupRewardNormalizer  -- normalize rewards within a group of G samples
- GRPOLoss              -- clipped policy-gradient loss with KL penalty
- GroupSampler          -- generate G responses per prompt autoregressively
- GRPOTrainer           -- full train step: sample -> reward -> normalize -> update
- RewardFunction        -- abstract base + LengthReward + UniqueTokenReward
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# GroupRewardNormalizer
# ---------------------------------------------------------------------------


class GroupRewardNormalizer:
    """Normalize rewards within a group to produce advantages.

    Args:
        eps: small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Return group-relative advantages.

        Args:
            rewards: shape (G,) for a single prompt, or (B, G) for a batch.

        Returns:
            advantages of the same shape. If all rewards are identical within
            a group the advantages are zero (no gradient signal).
        """
        if rewards.dim() == 1:
            std = rewards.std(unbiased=False)
            if std.item() < self.eps:
                return torch.zeros_like(rewards)
            mean = rewards.mean()
            return (rewards - mean) / (std + self.eps)

        # Batch -- shape (B, G)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        mask = (std < self.eps).expand_as(rewards)
        advantages = (rewards - mean) / (std + self.eps)
        advantages = advantages.masked_fill(mask, 0.0)
        return advantages

    def clip_advantages(self, advantages: torch.Tensor, clip_range: float = 10.0) -> torch.Tensor:
        """Clamp advantages to [-clip_range, clip_range]."""
        return advantages.clamp(-clip_range, clip_range)


# ---------------------------------------------------------------------------
# GRPOLoss
# ---------------------------------------------------------------------------


class GRPOLoss(nn.Module):
    """Core GRPO policy-gradient loss.

    Args:
        clip_eps:  PPO clip epsilon.
        kl_coeff:  coefficient for the KL-penalty term.
    """

    def __init__(self, clip_eps: float = 0.2, kl_coeff: float = 0.01) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff

    def forward(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute GRPO loss.

        Args:
            policy_logps: (B, G) log-probs under policy.
            ref_logps:    (B, G) log-probs under reference model.
            advantages:   (B, G) group-relative advantages.

        Returns:
            (total_loss, kl_penalty, clip_fraction)
        """
        log_ratio = policy_logps - ref_logps.detach()
        ratio = log_ratio.exp()

        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        kl_penalty = self.kl_coeff * torch.mean(policy_logps - ref_logps.detach())

        total_loss = policy_loss + kl_penalty

        clip_fraction = torch.mean(((ratio - clipped_ratio).abs() > 1e-6).float())

        return total_loss, kl_penalty.detach(), clip_fraction.detach()


# ---------------------------------------------------------------------------
# GroupSampler
# ---------------------------------------------------------------------------


class GroupSampler:
    """Generate G independent responses per prompt via autoregressive sampling.

    Args:
        model:        language model returning (B, T, V) logits.
        group_size:   number of responses G to sample per prompt.
        temperature:  sampling temperature.
    """

    def __init__(
        self,
        model: nn.Module,
        group_size: int = 4,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.group_size = group_size
        self.temperature = temperature

    @torch.no_grad()
    def sample_group(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Sample group_size responses from the model.

        Args:
            input_ids:      (1, T) or (T,) prompt token ids.
            max_new_tokens: maximum new tokens per response.

        Returns:
            responses:  list of G tensors, each shape (T_new,).
            log_probs:  (G,) cumulative log-probs per response.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = input_ids.device

        responses: list[torch.Tensor] = []
        log_probs_list: list[torch.Tensor] = []

        for _ in range(self.group_size):
            cur_ids = input_ids.clone()
            new_toks: list[int] = []
            cumulative_logp = torch.tensor(0.0, device=device)

            for _step in range(max_new_tokens):
                logits = self.model(cur_ids)
                next_logits = logits[:, -1, :]
                if self.temperature != 1.0:
                    next_logits = next_logits / self.temperature
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                token_id = next_tok.item()

                log_p = F.log_softmax(next_logits, dim=-1)
                cumulative_logp = cumulative_logp + log_p[0, token_id]

                new_toks.append(token_id)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)

            responses.append(torch.tensor(new_toks, dtype=torch.long, device=device))
            log_probs_list.append(cumulative_logp)

        log_probs = torch.stack(log_probs_list)
        return responses, log_probs


# ---------------------------------------------------------------------------
# RewardFunction
# ---------------------------------------------------------------------------


class RewardFunction(abc.ABC):
    """Abstract base class for reward functions."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, response_ids: torch.Tensor) -> float:
        """Compute reward for a single response.

        Args:
            response_ids: 1-D tensor of token ids.

        Returns:
            scalar float reward.
        """
        ...


class LengthReward(RewardFunction):
    """reward = 1.0 - |len - target| / target, clamped to [0, 1]."""

    def __init__(self, target_length: int) -> None:
        super().__init__()
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        self.target_length = target_length

    def __call__(self, response_ids: torch.Tensor) -> float:
        length = response_ids.numel()
        raw = 1.0 - abs(length - self.target_length) / self.target_length
        return float(max(0.0, min(1.0, raw)))


class UniqueTokenReward(RewardFunction):
    """reward = unique_tokens / total_tokens (lexical diversity)."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, response_ids: torch.Tensor) -> float:
        total = response_ids.numel()
        if total == 0:
            return 0.0
        unique = float(response_ids.unique().numel())
        return unique / total


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Full GRPO training loop -- one step per call.

    Args:
        policy_model:   model being trained.
        ref_model:      frozen reference model (all params frozen on init).
        optimizer:      optimizer attached to policy_model parameters.
        group_size:     G -- number of responses sampled per prompt.
        max_new_tokens: maximum tokens generated per response.
        clip_eps:       PPO clip epsilon.
        kl_coeff:       KL penalty coefficient.
        temperature:    sampling temperature.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        group_size: int = 4,
        max_new_tokens: int = 20,
        clip_eps: float = 0.2,
        kl_coeff: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens

        # Freeze reference model completely
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        self.ref_model.eval()

        self.sampler = GroupSampler(policy_model, group_size, temperature)
        self.normalizer = GroupRewardNormalizer()
        self.loss_fn = GRPOLoss(clip_eps=clip_eps, kl_coeff=kl_coeff)

    def _sequence_log_prob(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of per-token log-probs for response_ids given input_ids.

        Args:
            model:        language model.
            input_ids:    (1, T_prompt).
            response_ids: (T_new,).

        Returns:
            scalar log-prob tensor.
        """
        resp = response_ids.unsqueeze(0)
        full = torch.cat([input_ids, resp], dim=1)

        logits = model(full)

        T_prompt = input_ids.size(1)
        T_new = response_ids.size(0)

        pred_logits = logits[:, T_prompt - 1 : T_prompt - 1 + T_new, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        targets = response_ids.unsqueeze(0).unsqueeze(-1)
        token_logps = log_probs.gather(2, targets).squeeze(-1)
        return token_logps.sum()

    def train_step(
        self,
        input_ids: torch.Tensor,
        reward_fn: Callable[[torch.Tensor], float],
    ) -> dict[str, float]:
        """Perform one GRPO training step.

        Args:
            input_ids: (1, T) or (T,) prompt token ids.
            reward_fn: callable mapping response_ids (1-D Tensor) to float.

        Returns:
            dict with keys: loss, kl_penalty, clip_fraction, mean_reward,
            reward_std.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # 1. Sample G responses
        responses, _sample_logps = self.sampler.sample_group(input_ids, self.max_new_tokens)

        # 2. Compute rewards
        rewards_list = [reward_fn(r) for r in responses]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=input_ids.device)

        # 3. Normalize rewards -> advantages, shape (G,) -> (1, G)
        advantages = self.normalizer.normalize(rewards)
        advantages = self.normalizer.clip_advantages(advantages)
        advantages = advantages.unsqueeze(0)

        # 4. Re-compute log-probs with gradient graph
        self.policy_model.train()

        policy_logps_list: list[torch.Tensor] = []
        ref_logps_list: list[torch.Tensor] = []

        for resp in responses:
            plp = self._sequence_log_prob(self.policy_model, input_ids, resp)
            policy_logps_list.append(plp)

            with torch.no_grad():
                rlp = self._sequence_log_prob(self.ref_model, input_ids, resp)
            ref_logps_list.append(rlp)

        policy_logps = torch.stack(policy_logps_list).unsqueeze(0)
        ref_logps = torch.stack(ref_logps_list).unsqueeze(0)

        # 5. Loss, backward, step
        self.optimizer.zero_grad()
        loss, kl_penalty, clip_fraction = self.loss_fn(policy_logps, ref_logps, advantages)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "kl_penalty": kl_penalty.item(),
            "clip_fraction": clip_fraction.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
        }
