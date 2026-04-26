"""Aurelius -- Iterative DPO (Online / Self-Play DPO) training.

Improves on standard DPO by iterating:
  1. Sample new responses from the current policy.
  2. Score them with a reward model.
  3. Create new preference pairs from the scored samples.
  4. Run another DPO update.

Each iteration the policy improves and generates harder training signals.

Pure PyTorch -- no HuggingFace dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IterativeDPOConfig:
    """Configuration for Iterative DPO training."""

    beta: float = 0.1  # KL penalty coefficient
    n_iterations: int = 3  # number of outer DPO iterations
    n_samples_per_prompt: int = 4  # responses sampled per prompt per iteration
    reward_threshold: float = 0.0  # minimum reward margin to form a pair
    max_new_tokens: int = 64  # max tokens generated per response
    update_ref_every_n_iters: int = 1  # how often to update the reference policy


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    """Per-iteration training result summary."""

    iteration: int
    mean_reward: float
    reward_margin: float
    n_pairs: int
    loss: float


# ---------------------------------------------------------------------------
# IterativeDPOTrainer
# ---------------------------------------------------------------------------


class IterativeDPOTrainer:
    """Iterative DPO trainer: repeatedly samples from the current policy, scores
    responses with a reward function, constructs preference pairs, and applies
    a DPO update.

    Args:
        policy:     Trainable policy model. Forward signature:
                    (loss, logits, past_key_values) = model(input_ids)
        ref_policy: Frozen reference model (same architecture as policy).
        reward_fn:  Callable mapping response_ids (1-D LongTensor) -> float.
        beta:       KL penalty coefficient (overridden by config.beta if config given).
        config:     IterativeDPOConfig. Created with defaults if not provided.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_fn: Callable[[torch.Tensor], float],
        beta: float = 0.1,
        config: IterativeDPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_fn = reward_fn

        if config is None:
            config = IterativeDPOConfig(beta=beta)
        self.config = config

        # Freeze reference policy
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

    # ------------------------------------------------------------------
    # Response sampling
    # ------------------------------------------------------------------

    def sample_responses(
        self,
        prompt_ids: torch.Tensor,
        n_samples: int,
        max_new_tokens: int = 64,
    ) -> list[torch.Tensor]:
        """Sample n_samples responses from the current policy.

        Args:
            prompt_ids:     Shape (1, prompt_len) -- single prompt (B=1).
            n_samples:      Number of independent responses to generate.
            max_new_tokens: Maximum new tokens per response.

        Returns:
            List of n_samples response tensors, each shape (max_new_tokens,).
        """
        self.policy.eval()
        responses: list[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(n_samples):
                current = prompt_ids.clone()  # (1, prompt_len)
                generated: list[torch.Tensor] = []

                for _ in range(max_new_tokens):
                    _, logits, _ = self.policy(current)  # (1, seq, vocab)
                    next_logits = logits[:, -1, :]  # (1, vocab)
                    log_p = F.log_softmax(next_logits, dim=-1)
                    probs = log_p.exp()
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                    generated.append(next_token[0])  # (1,)
                    current = torch.cat([current, next_token], dim=1)

                resp = torch.cat(generated, dim=0)  # (max_new_tokens,)
                responses.append(resp)

        self.policy.train()
        return responses

    # ------------------------------------------------------------------
    # Preference pair construction
    # ------------------------------------------------------------------

    def create_preference_pairs(
        self,
        prompt_ids: torch.Tensor,
        responses: list[torch.Tensor],
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Select the best (chosen) and worst (rejected) response by reward.

        Args:
            prompt_ids: Shape (1, prompt_len) -- single prompt.
            responses:  List of n response tensors, each (T,).
            rewards:    Shape (n,) -- scalar reward for each response.

        Returns:
            None if all rewards are identical (no learning signal).
            Otherwise (chosen_ids, rejected_ids) where each is (T,).
        """
        if rewards.max() == rewards.min():
            return None

        best_idx = int(rewards.argmax().item())
        worst_idx = int(rewards.argmin().item())

        chosen_ids = responses[best_idx]
        rejected_ids = responses[worst_idx]

        return chosen_ids, rejected_ids

    # ------------------------------------------------------------------
    # DPO loss
    # ------------------------------------------------------------------

    def _compute_sequence_log_probs(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute summed log prob of response tokens given prompt.

        Args:
            model:        Policy or reference model.
            prompt_ids:   Shape (1, prompt_len).
            response_ids: Shape (1, T).

        Returns:
            Scalar tensor -- sum of per-token log probs over the response.
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (1, prompt_len+T)
        prompt_len = prompt_ids.shape[1]

        _, logits, _ = model(full_ids)  # (1, seq, vocab)

        # Shift: position i predicts token i+1
        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq-1, vocab)

        # Response starts at prompt_len in full_ids; in shifted view that is prompt_len-1
        resp_start = prompt_len - 1
        T = response_ids.shape[1]
        log_probs_resp = log_probs_all[:, resp_start : resp_start + T, :]  # (1, T, vocab)

        token_lp = log_probs_resp.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # (1, T)

        return token_lp.sum(dim=-1).squeeze(0)  # scalar

    def compute_dpo_loss(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the DPO loss for a single preference pair.

        Args:
            prompt_ids:   Shape (1, prompt_len).
            chosen_ids:   Shape (T,) -- response with higher reward.
            rejected_ids: Shape (T,) -- response with lower reward.

        Returns:
            (loss, metrics) where metrics contains:
              'loss', 'chosen_reward', 'rejected_reward', 'reward_margin'.
        """
        self.policy.train()
        self.ref_policy.eval()

        chosen_b = chosen_ids.unsqueeze(0)  # (1, T)
        rejected_b = rejected_ids.unsqueeze(0)  # (1, T)

        # Policy log probs (with gradient)
        pi_chosen_lp = self._compute_sequence_log_probs(self.policy, prompt_ids, chosen_b)
        pi_rejected_lp = self._compute_sequence_log_probs(self.policy, prompt_ids, rejected_b)

        # Reference log probs (no gradient)
        with torch.no_grad():
            ref_chosen_lp = self._compute_sequence_log_probs(self.ref_policy, prompt_ids, chosen_b)
            ref_rejected_lp = self._compute_sequence_log_probs(
                self.ref_policy, prompt_ids, rejected_b
            )

        beta = self.config.beta
        chosen_reward = beta * (pi_chosen_lp - ref_chosen_lp)
        rejected_reward = beta * (pi_rejected_lp - ref_rejected_lp)

        logits = chosen_reward - rejected_reward
        loss = -F.logsigmoid(logits)

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_reward.detach().item(),
            "rejected_reward": rejected_reward.detach().item(),
            "reward_margin": (chosen_reward - rejected_reward).detach().item(),
        }

        return loss, metrics

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_iteration(self, prompts: list[torch.Tensor]) -> dict:
        """Run one DPO iteration over the provided prompts.

        For each prompt:
          1. Sample n_samples_per_prompt responses from current policy.
          2. Score each with reward_fn.
          3. Create best/worst preference pair.
          4. Compute DPO loss (accumulated).

        Args:
            prompts: List of prompt tensors, each shape (1, prompt_len).

        Returns:
            Dict with keys: mean_reward, reward_margin, n_pairs, loss.
        """
        all_rewards: list[float] = []
        all_margins: list[float] = []
        all_losses: list[torch.Tensor] = []
        n_pairs = 0

        cfg = self.config

        for prompt_ids in prompts:
            # Sample responses
            responses = self.sample_responses(
                prompt_ids,
                n_samples=cfg.n_samples_per_prompt,
                max_new_tokens=cfg.max_new_tokens,
            )

            # Score responses
            rewards_list = [float(self.reward_fn(r)) for r in responses]
            rewards = torch.tensor(rewards_list, dtype=torch.float32)
            all_rewards.extend(rewards_list)

            # Create preference pair
            pair = self.create_preference_pairs(prompt_ids, responses, rewards)
            if pair is None:
                continue

            chosen_ids, rejected_ids = pair

            # Compute DPO loss
            loss, metrics = self.compute_dpo_loss(prompt_ids, chosen_ids, rejected_ids)
            all_losses.append(loss)
            all_margins.append(metrics["reward_margin"])
            n_pairs += 1

        mean_reward = float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        mean_margin = float(sum(all_margins) / len(all_margins)) if all_margins else 0.0
        mean_loss = torch.stack(all_losses).mean().item() if all_losses else 0.0

        return {
            "mean_reward": mean_reward,
            "reward_margin": mean_margin,
            "n_pairs": n_pairs,
            "loss": mean_loss,
        }

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def run(self, prompts: list[torch.Tensor]) -> list[dict]:
        """Run n_iterations rounds of Iterative DPO.

        Updates the reference policy every update_ref_every_n_iters iterations.

        Args:
            prompts: List of prompt tensors, each shape (1, prompt_len).

        Returns:
            List of per-iteration metric dicts (length == n_iterations).
        """
        results: list[dict] = []

        for i in range(self.config.n_iterations):
            logger.info("Iterative DPO -- iteration %d/%d", i + 1, self.config.n_iterations)

            iter_metrics = self.run_iteration(prompts)
            iter_metrics["iteration"] = i
            results.append(iter_metrics)

            logger.info(
                "  mean_reward=%.4f  reward_margin=%.4f  n_pairs=%d  loss=%.4f",
                iter_metrics["mean_reward"],
                iter_metrics["reward_margin"],
                iter_metrics["n_pairs"],
                iter_metrics["loss"],
            )

            # Optionally refresh reference policy
            if (i + 1) % self.config.update_ref_every_n_iters == 0:
                self.update_ref_policy()
                logger.info("  Reference policy updated.")

        return results

    # ------------------------------------------------------------------
    # Reference policy update
    # ------------------------------------------------------------------

    def update_ref_policy(self) -> None:
        """Copy current policy weights to ref_policy (hard copy).

        After the copy the reference policy is re-frozen and set to eval mode.
        """
        with torch.no_grad():
            for ref_param, policy_param in zip(
                self.ref_policy.parameters(), self.policy.parameters()
            ):
                ref_param.data.copy_(policy_param.data)

        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()
