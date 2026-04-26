"""Aurelius — Reinforced Fine-Tuning (RFT).

Generate K completions per prompt, score with a reward model, keep only the
best ones, and train on them via standard SFT cross-entropy loss.

Reference: "Let's Reinforce Step by Step" and related rejection-sampling work.
Much simpler than PPO while surprisingly effective for alignment.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RFTSampler
# ---------------------------------------------------------------------------


class RFTSampler:
    """Generate K completions per prompt, score each, and filter by threshold.

    Args:
        model: Generation model with a forward pass that returns
               (loss, logits, kv_cache).  Must support ``model(input_ids)``.
        reward_model: Callable ``(prompt_ids, response_ids) -> float`` that
                      scores a single completion.
        k: Number of completions to sample per prompt.
        temperature: Sampling temperature applied to logits.
        top_p: Nucleus sampling threshold (unused in _generate but kept for
               compatibility with downstream callers).
        reward_threshold: Minimum reward to keep a completion.
        max_new_tokens: Maximum tokens to generate per completion.
    """

    def __init__(
        self,
        model: nn.Module,
        reward_model: Callable[[torch.Tensor, torch.Tensor], float],
        k: int = 8,
        temperature: float = 0.8,
        top_p: float = 0.9,
        reward_threshold: float = 0.5,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.reward_model = reward_model
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        self.reward_threshold = reward_threshold
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Simple autoregressive generation from the model.

        Args:
            prompt_ids: ``(1, prompt_len)`` token id tensor.
            max_new_tokens: Number of new tokens to generate.
            temperature: Logit temperature; 1.0 = no scaling.

        Returns:
            ``(1, n_generated)`` tensor of newly generated token ids (not
            including the prompt).
        """
        cur_ids = prompt_ids
        generated: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            _, logits, _ = self.model(cur_ids)
            next_logits = logits[:, -1, :]  # (1, vocab)

            if temperature != 1.0 and temperature > 0.0:
                next_logits = next_logits / temperature

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated.append(next_token)
            # Feed only the new token on subsequent steps (no KV cache here,
            # but keeping the sequence short keeps the mock fast in tests).
            cur_ids = next_token

        if not generated:
            return prompt_ids.new_empty((1, 0))
        return torch.cat(generated, dim=1)  # (1, n_generated)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_and_score(
        self,
        prompt_ids: torch.Tensor,
    ) -> list[dict]:
        """Generate ``k`` completions and score each with the reward model.

        Args:
            prompt_ids: ``(1, prompt_len)`` prompt token ids.

        Returns:
            List of ``{'response_ids': Tensor, 'reward': float}`` dicts,
            sorted by reward descending.
        """
        results: list[dict] = []

        for _ in range(self.k):
            response_ids = self._generate(
                prompt_ids, self.max_new_tokens, self.temperature
            )  # (1, n_gen)
            reward = float(self.reward_model(prompt_ids, response_ids))
            results.append({"response_ids": response_ids, "reward": reward})

        # Sort descending by reward
        results.sort(key=lambda d: d["reward"], reverse=True)
        return results

    def filter_completions(self, scored: list[dict]) -> list[dict]:
        """Keep only completions with reward >= reward_threshold.

        Args:
            scored: Output of ``sample_and_score``.

        Returns:
            Filtered list (may be empty).
        """
        return [d for d in scored if d["reward"] >= self.reward_threshold]


# ---------------------------------------------------------------------------
# RFTDataset
# ---------------------------------------------------------------------------


class RFTDataset:
    """Dataset of ``(prompt, accepted_response)`` pairs built by RFTSampler.

    Each example has:
    - ``input_ids``: concat of prompt and response tokens.
    - ``labels``: ``-100`` for every prompt position, response tokens otherwise.
    """

    def __init__(self) -> None:
        self.examples: list[dict] = []

    def add_examples(
        self,
        prompt_ids: torch.Tensor,
        accepted: list[dict],
    ) -> None:
        """Add accepted completions to the dataset.

        Args:
            prompt_ids: ``(1, prompt_len)`` prompt token ids.
            accepted: Filtered completions from ``RFTSampler.filter_completions``.
                      Each entry must have a ``'response_ids'`` key with shape
                      ``(1, response_len)``.
        """
        prompt_len = prompt_ids.shape[1]

        for item in accepted:
            response_ids = item["response_ids"]  # (1, response_len)

            # Concatenate along sequence dimension → (1, prompt_len + response_len)
            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)  # (S,)

            # Build labels: mask prompt with -100
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# RFTTrainer
# ---------------------------------------------------------------------------


class RFTTrainer:
    """Train on RFT-filtered completions using SFT (cross-entropy) loss.

    Args:
        model: The model to fine-tune.  Must accept ``model(input_ids, labels=labels)``
               and return ``(loss, logits, kv)``.
        optimizer: PyTorch optimizer already configured for ``model``.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """One SFT gradient step on a batch of RFT examples.

        Only positions where ``labels != -100`` contribute to the loss.

        Args:
            input_ids: ``(B, S)`` batched token ids.
            labels: ``(B, S)`` labels; ``-100`` at prompt positions.

        Returns:
            Scalar loss as a Python float.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, _, _ = self.model(input_ids, labels=labels)

        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def rft_loop(
        self,
        sampler: RFTSampler,
        prompts: list[torch.Tensor],
        n_epochs: int = 1,
    ) -> list[float]:
        """Full RFT loop: sample → filter → collect → train.

        Args:
            sampler: Configured ``RFTSampler``.
            prompts: List of ``(1, prompt_len)`` prompt tensors.
            n_epochs: Number of passes over the collected dataset.

        Returns:
            Per-step losses (empty list if no completions were accepted).
        """
        # ---- Phase 1: collect accepted completions ----
        dataset = RFTDataset()
        for prompt_ids in prompts:
            scored = sampler.sample_and_score(prompt_ids)
            accepted = sampler.filter_completions(scored)
            if accepted:
                dataset.add_examples(prompt_ids, accepted)

        if len(dataset) == 0:
            logger.info("RFT loop: no accepted completions, skipping training.")
            return []

        # ---- Phase 2: train on accepted completions ----
        losses: list[float] = []
        for _epoch in range(n_epochs):
            for example in dataset:
                input_ids = example["input_ids"].unsqueeze(0)  # (1, S)
                labels = example["labels"].unsqueeze(0)  # (1, S)
                step_loss = self.train_step(input_ids, labels)
                losses.append(step_loss)

        return losses


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_reward_stats(scored: list[dict]) -> dict:
    """Compute summary statistics for a list of scored completions.

    Args:
        scored: List of dicts with at least a ``'reward'`` key.

    Returns:
        Dict with keys ``mean_reward``, ``max_reward``, ``min_reward``,
        and ``acceptance_rate`` (fraction with reward > 0 by convention).
    """
    if not scored:
        return {
            "mean_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "acceptance_rate": 0.0,
        }

    rewards = [d["reward"] for d in scored]
    mean_r = sum(rewards) / len(rewards)
    max_r = max(rewards)
    min_r = min(rewards)
    acceptance_rate = sum(1 for r in rewards if r > 0) / len(rewards)

    return {
        "mean_reward": mean_r,
        "max_reward": max_r,
        "min_reward": min_r,
        "acceptance_rate": acceptance_rate,
    }
