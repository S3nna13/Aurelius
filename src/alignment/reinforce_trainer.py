"""REINFORCE and RLOO (Leave-One-Out) policy gradient for LLM fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReinforceConfig:
    """Hyperparameters for REINFORCE / RLOO policy gradient training."""

    n_samples: int = 4          # rollouts per prompt
    kl_coeff: float = 0.05      # KL penalty against reference model
    gamma: float = 1.0          # discount factor (1.0 = no discount for text gen)
    normalize_rewards: bool = True
    max_new_tokens: int = 32
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Core policy-gradient utilities
# ---------------------------------------------------------------------------

def compute_reinforce_loss(
    log_probs: Tensor,
    rewards: Tensor,
    baseline: Tensor | None = None,
) -> Tensor:
    """Compute the REINFORCE policy-gradient loss.

    Args:
        log_probs: (B, T) per-token log probabilities of generated tokens.
        rewards:   (B,)  scalar reward per sequence.
        baseline:  (B,)  optional variance-reduction baseline; defaults to 0.

    Returns:
        Scalar loss = -mean((rewards - baseline) * log_probs.sum(dim=-1)).
    """
    if baseline is None:
        baseline = torch.zeros_like(rewards)

    seq_log_probs = log_probs.sum(dim=-1)          # (B,)
    advantages = rewards - baseline                 # (B,)
    loss = -(advantages * seq_log_probs).mean()
    return loss


def compute_rloo_baseline(rewards: Tensor) -> Tensor:
    """Compute Leave-One-Out (RLOO) baseline for variance reduction.

    For each sample i in a group, the baseline is the mean reward of all
    *other* samples in the group.

    Args:
        rewards: (G,) rewards for a single group (G = n_samples).

    Returns:
        (G,) baseline tensor where baseline[i] = mean of rewards[j] for j != i.
        Returns zeros when G == 1.
    """
    G = rewards.shape[0]
    if G == 1:
        return torch.zeros_like(rewards)

    total = rewards.sum()
    # baseline[i] = (total - rewards[i]) / (G - 1)
    baseline = (total - rewards) / (G - 1)
    return baseline


def compute_kl_penalty(
    log_probs_policy: Tensor,
    log_probs_ref: Tensor,
) -> Tensor:
    """Compute per-token approximate KL divergence: policy - reference.

    This is the first-order (linear) KL approximation used in RLHF:
        KL(policy || ref) approx log_policy - log_ref

    Args:
        log_probs_policy: (B, T) log probs under the policy model.
        log_probs_ref:    (B, T) log probs under the reference model.

    Returns:
        (B, T) per-token KL penalty.  Sum over T and multiply by kl_coeff
        before subtracting from sequence rewards.
    """
    return log_probs_policy - log_probs_ref


# ---------------------------------------------------------------------------
# Rollout sampling
# ---------------------------------------------------------------------------

def sample_rollout(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Generate tokens autoregressively and collect log probs.

    Args:
        model:          AureliusTransformer (or compatible) -- forward returns
                        (loss, logits, past_key_values).
        input_ids:      (1, prompt_len) prompt token ids.
        max_new_tokens: Number of tokens to generate.
        temperature:    Sampling temperature; uses multinomial sampling.

    Returns:
        Tuple (generated_ids, log_probs):
            generated_ids: (1, max_new_tokens) generated token ids.
            log_probs:     (1, max_new_tokens) log probs of each generated token.
    """
    generated_ids_list: list[Tensor] = []
    log_probs_list: list[Tensor] = []

    past_key_values = None
    cur_ids = input_ids  # (1, S)

    for _ in range(max_new_tokens):
        _, logits, past_key_values = model(cur_ids, past_key_values=past_key_values)
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        if temperature != 1.0:
            next_logits = next_logits / temperature

        log_probs_step = F.log_softmax(next_logits, dim=-1)  # (1, vocab_size)

        # Sample next token
        probs = log_probs_step.exp()
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Gather log prob of the sampled token
        token_log_prob = log_probs_step.gather(1, next_token)  # (1, 1)

        generated_ids_list.append(next_token)
        log_probs_list.append(token_log_prob)

        # Next step: only the new token (KV cache handles context)
        cur_ids = next_token

    generated_ids = torch.cat(generated_ids_list, dim=1)  # (1, max_new_tokens)
    log_probs_out = torch.cat(log_probs_list, dim=1)      # (1, max_new_tokens)
    return generated_ids, log_probs_out


# ---------------------------------------------------------------------------
# ReinforceTrainer
# ---------------------------------------------------------------------------

class ReinforceTrainer:
    """Policy-gradient trainer using REINFORCE with RLOO baseline and KL penalty.

    Supports both REINFORCE (baseline=None or mean) and RLOO (Leave-One-Out)
    variance reduction, plus a per-token KL penalty against a frozen reference
    model.
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        reward_fn: Callable[[list[int]], float],
        config: ReinforceConfig,
        optimizer,
    ) -> None:
        self.policy = policy_model
        self.ref = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, prompt_ids: Tensor) -> dict[str, float]:
        """Perform one REINFORCE update step.

        Args:
            prompt_ids: (1, prompt_len) prompt tensor.

        Returns:
            Dict with keys "loss", "mean_reward", "mean_kl".
        """
        cfg = self.config
        self.policy.train()
        self.ref.train(False)

        all_log_probs: list[Tensor] = []
        all_rewards: list[float] = []
        all_kl: list[Tensor] = []

        for _ in range(cfg.n_samples):
            # Sample from policy
            gen_ids, log_probs = sample_rollout(
                self.policy, prompt_ids, cfg.max_new_tokens, cfg.temperature
            )
            # gen_ids:   (1, max_new_tokens)
            # log_probs: (1, max_new_tokens)

            # Reward from external function
            token_list = gen_ids[0].tolist()
            reward = self.reward_fn(token_list)
            all_rewards.append(reward)

            # KL penalty using reference model log probs
            with torch.no_grad():
                full_ids = torch.cat([prompt_ids, gen_ids], dim=1)  # (1, P+T)
                _, ref_logits, _ = self.ref(full_ids)
                P = prompt_ids.shape[1]
                T = gen_ids.shape[1]
                ref_log_probs_all = F.log_softmax(ref_logits[:, :-1, :], dim=-1)  # (1, P+T-1, V)
                ref_log_probs_gen = ref_log_probs_all[:, P - 1: P - 1 + T, :]     # (1, T, V)
                ref_log_probs_tok = ref_log_probs_gen.gather(
                    2, gen_ids.unsqueeze(-1)
                ).squeeze(-1)  # (1, T)

            kl = compute_kl_penalty(log_probs, ref_log_probs_tok)  # (1, T)
            all_kl.append(kl)

            # Adjust reward with KL penalty
            kl_penalty = kl.sum(dim=-1).squeeze(0)  # scalar
            adjusted_reward = reward - cfg.kl_coeff * kl_penalty.item()
            all_rewards[-1] = adjusted_reward

            all_log_probs.append(log_probs)  # (1, T)

        # Stack across samples: (n_samples, T) and (n_samples,)
        log_probs_batch = torch.cat(all_log_probs, dim=0)                        # (n_samples, T)
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32,
                                      device=prompt_ids.device)                  # (n_samples,)
        kl_batch = torch.cat(all_kl, dim=0)                                      # (n_samples, T)

        if cfg.normalize_rewards and cfg.n_samples > 1:
            std = rewards_tensor.std() + 1e-8
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / std

        # RLOO baseline
        baseline = compute_rloo_baseline(rewards_tensor)  # (n_samples,)

        # Compute loss
        loss = compute_reinforce_loss(log_probs_batch, rewards_tensor, baseline)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_kl = kl_batch.sum(dim=-1).mean().item()

        return {
            "loss": loss.item(),
            "mean_reward": rewards_tensor.mean().item(),
            "mean_kl": mean_kl,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, prompts: list[Tensor], n_eval: int) -> dict[str, float]:
        """Compute mean reward over a set of evaluation prompts.

        Args:
            prompts: List of (1, prompt_len) tensors.
            n_eval:  Number of prompts to evaluate (uses first n_eval).

        Returns:
            Dict with key "mean_reward".
        """
        self.policy.train(False)
        rewards: list[float] = []

        with torch.no_grad():
            for prompt in prompts[:n_eval]:
                gen_ids, _ = sample_rollout(
                    self.policy, prompt, self.config.max_new_tokens, self.config.temperature
                )
                reward = self.reward_fn(gen_ids[0].tolist())
                rewards.append(reward)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return {"mean_reward": mean_reward}
