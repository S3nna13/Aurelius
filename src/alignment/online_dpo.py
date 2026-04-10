"""Online DPO: dynamic preference pair generation and direct policy optimization.

Generates chosen/rejected preference pairs on-the-fly from the policy model
itself, scoring candidates with a callable reward function and computing DPO
loss directly -- no fixed offline dataset needed.

Algorithm:
  For each prompt:
    1. Sample n_samples completions from the current policy.
    2. Score each response with reward_fn.
    3. Select chosen = argmax reward, rejected = argmin reward.
    4. Compute DPO loss against the frozen reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineDPOConfig:
    """Configuration for Online DPO training."""

    beta: float = 0.1           # KL penalty coefficient
    n_samples: int = 2          # responses sampled per prompt
    temperature: float = 1.0    # sampling temperature
    max_new_tokens: int = 16    # maximum new tokens to generate per response
    label_smoothing: float = 0.0  # label smoothing for DPO loss


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_response(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Greedy/temperature sample from model.

    Args:
        model: Policy model. Forward signature:
               (loss, logits, past_key_values) = model(input_ids)
        prompt_ids: Shape (B, prompt_len) -- batched prompt token ids.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature. 1.0 = categorical; 0.0 = greedy.

    Returns:
        Tuple of:
          - response_ids: (B, max_new_tokens) -- sampled response token ids.
          - log_probs:    (B, max_new_tokens) -- per-token log probs under policy.
    """
    device = prompt_ids.device
    generated_ids: list[Tensor] = []
    generated_lps: list[Tensor] = []

    current = prompt_ids  # (B, seq)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            _, logits, _ = model(current)       # (B, seq, vocab)
            next_logits = logits[:, -1, :]      # (B, vocab)

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                token_lp = F.log_softmax(next_logits, dim=-1).gather(
                    1, next_token
                )  # (B, 1)
            else:
                scaled = next_logits / temperature
                log_p = F.log_softmax(scaled, dim=-1)   # (B, vocab)
                probs = log_p.exp()
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                token_lp = log_p.gather(1, next_token)  # (B, 1)

            generated_ids.append(next_token)    # each (B, 1)
            generated_lps.append(token_lp)      # each (B, 1)

            current = torch.cat([current, next_token], dim=1)  # (B, seq+1)

    response_ids = torch.cat(generated_ids, dim=1)  # (B, max_new_tokens)
    log_probs = torch.cat(generated_lps, dim=1)     # (B, max_new_tokens)

    return response_ids, log_probs


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

def compute_sequence_log_probs(
    model: nn.Module,
    prompt_ids: Tensor,
    response_ids: Tensor,
) -> Tensor:
    """Compute per-token log probs of response_ids given prompt_ids.

    Runs the model on [prompt_ids | response_ids] and extracts the log probs
    of the response tokens from the prompt-length positions onward.

    Args:
        model: Policy or reference model. Forward signature:
               (loss, logits, past_key_values) = model(input_ids)
        prompt_ids: Shape (B, prompt_len).
        response_ids: Shape (B, max_new_tokens).

    Returns:
        Shape (B, max_new_tokens) -- per-token log probs for each response token.
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (B, prompt_len + T)
    prompt_len = prompt_ids.shape[1]

    _, logits, _ = model(full_ids)  # (B, seq, vocab)

    # Shift: position i predicts token i+1
    log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, seq-1, vocab)

    # Response tokens start at index prompt_len in full_ids.
    # In the shifted prediction view, position (prompt_len - 1) predicts
    # full_ids[:, prompt_len], i.e. response_ids[:, 0].
    resp_start = prompt_len - 1
    T = response_ids.shape[1]
    log_probs_resp = log_probs_all[:, resp_start: resp_start + T, :]  # (B, T, vocab)

    # Gather actual token log probs
    token_lp = log_probs_resp.gather(
        2, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    return token_lp


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_chosen_log_probs: Tensor,
    policy_rejected_log_probs: Tensor,
    ref_chosen_log_probs: Tensor,
    ref_rejected_log_probs: Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict]:
    """Compute DPO loss.

    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    loss = -F.logsigmoid(logits)  (with optional label smoothing)

    Args:
        policy_chosen_log_probs:   (B,) summed log probs for chosen under policy.
        policy_rejected_log_probs: (B,) summed log probs for rejected under policy.
        ref_chosen_log_probs:      (B,) summed log probs for chosen under reference.
        ref_rejected_log_probs:    (B,) summed log probs for rejected under reference.
        beta: KL penalty coefficient.
        label_smoothing: Optional label smoothing in [0, 0.5).

    Returns:
        Tuple of (mean_loss, metrics) where metrics contains:
          "chosen_reward":   float
          "rejected_reward": float
          "reward_margin":   float
          "accuracy":        float
    """
    chosen_rewards = beta * (policy_chosen_log_probs - ref_chosen_log_probs)
    rejected_rewards = beta * (policy_rejected_log_probs - ref_rejected_log_probs)

    logits = chosen_rewards - rejected_rewards  # (B,)

    if label_smoothing > 0.0:
        loss = (
            -F.logsigmoid(logits) * (1.0 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(logits)

    mean_loss = loss.mean()

    chosen_reward_mean = chosen_rewards.mean().item()
    rejected_reward_mean = rejected_rewards.mean().item()
    reward_margin = chosen_reward_mean - rejected_reward_mean
    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    metrics = {
        "chosen_reward": chosen_reward_mean,
        "rejected_reward": rejected_reward_mean,
        "reward_margin": reward_margin,
        "accuracy": accuracy,
    }

    return mean_loss, metrics


# ---------------------------------------------------------------------------
# OnlineDPOTrainer
# ---------------------------------------------------------------------------

class OnlineDPOTrainer:
    """Online DPO: generate pairs on-the-fly, rank with reward_fn, apply DPO loss.

    Args:
        policy: Trainable policy (AureliusTransformer).
        ref_model: Frozen reference model.
        reward_fn: Callable mapping response_ids (1-D Tensor) -> float score.
        config: OnlineDPOConfig.
        optimizer: PyTorch optimizer bound to policy parameters.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[Tensor], float],
        config: OnlineDPOConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.config = config
        self.optimizer = optimizer

        # Ensure ref model is frozen
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def generate_preference_pair(
        self, prompt_ids: Tensor
    ) -> tuple[Tensor, Tensor, float, float]:
        """Sample n_samples responses, rank by reward_fn, return best/worst pair.

        Args:
            prompt_ids: Shape (1, prompt_len) -- single prompt (B=1).

        Returns:
            Tuple of:
              - chosen_ids:      (max_new_tokens,) -- response with highest reward.
              - rejected_ids:    (max_new_tokens,) -- response with lowest reward.
              - chosen_reward:   float.
              - rejected_reward: float.
        """
        self.policy.eval()
        responses: list[Tensor] = []
        rewards: list[float] = []

        with torch.no_grad():
            for _ in range(self.config.n_samples):
                resp_ids, _ = sample_response(
                    self.policy,
                    prompt_ids,
                    self.config.max_new_tokens,
                    self.config.temperature,
                )
                # resp_ids: (1, max_new_tokens) -> (max_new_tokens,)
                resp_1d = resp_ids[0]
                reward = float(self.reward_fn(resp_1d))
                responses.append(resp_1d)
                rewards.append(reward)

        best_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
        worst_idx = int(min(range(len(rewards)), key=lambda i: rewards[i]))

        return (
            responses[best_idx],
            responses[worst_idx],
            rewards[best_idx],
            rewards[worst_idx],
        )

    def train_step(self, prompt_ids: Tensor) -> dict:
        """Full Online DPO step.

        Steps:
          1. generate_preference_pair
          2. compute log probs from policy and ref
          3. dpo_loss
          4. backward + optimizer.step()

        Args:
            prompt_ids: Shape (1, prompt_len) -- single prompt (B=1).

        Returns:
            Dict with keys: loss, chosen_reward, rejected_reward, reward_margin, accuracy.
        """
        self.policy.train()
        self.ref_model.eval()

        # 1. Generate preference pair
        chosen_resp, rejected_resp, chosen_reward, rejected_reward = (
            self.generate_preference_pair(prompt_ids)
        )

        # chosen_resp / rejected_resp: (max_new_tokens,) -- unsqueeze for batch dim
        chosen_resp_b = chosen_resp.unsqueeze(0)      # (1, T)
        rejected_resp_b = rejected_resp.unsqueeze(0)  # (1, T)

        # 2. Policy log probs (with gradient)
        self.policy.train()
        pi_chosen_lp = compute_sequence_log_probs(
            self.policy, prompt_ids, chosen_resp_b
        ).sum(dim=-1)   # (1,)

        pi_rejected_lp = compute_sequence_log_probs(
            self.policy, prompt_ids, rejected_resp_b
        ).sum(dim=-1)   # (1,)

        # 3. Reference log probs (no gradient)
        with torch.no_grad():
            ref_chosen_lp = compute_sequence_log_probs(
                self.ref_model, prompt_ids, chosen_resp_b
            ).sum(dim=-1)   # (1,)

            ref_rejected_lp = compute_sequence_log_probs(
                self.ref_model, prompt_ids, rejected_resp_b
            ).sum(dim=-1)   # (1,)

        # 4. DPO loss
        loss, metrics = dpo_loss(
            pi_chosen_lp,
            pi_rejected_lp,
            ref_chosen_lp,
            ref_rejected_lp,
            beta=self.config.beta,
            label_smoothing=self.config.label_smoothing,
        )

        # 5. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": chosen_reward - rejected_reward,
            "accuracy": metrics["accuracy"],
        }
