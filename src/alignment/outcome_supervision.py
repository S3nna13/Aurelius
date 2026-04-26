"""Outcome Supervision: Outcome Reward Models (ORM) for final-answer verification.

ORMs assign binary rewards based on whether the final answer is correct,
contrasted with PRMs which score intermediate reasoning steps. Key features:
  - Answer verification via configurable verify_fn
  - Binary reward assignment (correct_reward / wrong_reward)
  - Monte Carlo estimation of outcome probability via multiple completions
  - Unbiased pass@k estimation (HumanEval-style)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Computes the probability that at least 1 of k randomly-drawn samples
    is correct, given n total samples with c correct ones.

    Formula: 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)

    Args:
        n: Total number of samples generated.
        c: Number of correct samples.
        k: Number of samples drawn (k <= n).

    Returns:
        Probability in [0.0, 1.0].
        Returns 0.0 if c == 0.
        Returns 1.0 if c == n.
    """
    if c == 0:
        return 0.0
    if c == n:
        return 1.0
    if k > n:
        k = n

    # Product of (n-c-i)/(n-i) for i in 0..k-1
    prob_none_correct = 1.0
    for i in range(k):
        if n - i == 0:
            break
        prob_none_correct *= (n - c - i) / (n - i)

    return 1.0 - prob_none_correct


# ---------------------------------------------------------------------------
# ORMConfig
# ---------------------------------------------------------------------------


@dataclass
class ORMConfig:
    """Configuration for Outcome Reward Model training and inference."""

    correct_reward: float = 1.0  # reward for a correct final answer
    wrong_reward: float = 0.0  # reward for an incorrect final answer
    n_samples: int = 16  # default number of MC samples
    temperature: float = 0.8  # sampling temperature for MC rollouts


# ---------------------------------------------------------------------------
# OutcomeVerifier
# ---------------------------------------------------------------------------


class OutcomeVerifier:
    """Verifies whether a model response matches a ground-truth answer.

    Args:
        verify_fn: Callable ``(response_ids, ground_truth) -> bool`` that
                   returns True when the response is considered correct.
    """

    def __init__(self, verify_fn: Callable[[torch.Tensor, Any], bool]) -> None:
        self.verify_fn = verify_fn

    def verify(self, response_ids: torch.Tensor, ground_truth: Any) -> bool:
        """Check a single response against the ground truth.

        Args:
            response_ids: 1-D or 2-D (1, T) token tensor representing the
                          model's generated response.
            ground_truth: Reference answer in any form accepted by verify_fn.

        Returns:
            True if the response is correct, False otherwise.
        """
        return bool(self.verify_fn(response_ids, ground_truth))

    def batch_verify(
        self,
        responses: list[torch.Tensor],
        ground_truths: list[Any],
    ) -> torch.Tensor:
        """Verify a batch of responses.

        Args:
            responses:     List of n response tensors.
            ground_truths: List of n ground-truth values (one per response).

        Returns:
            Boolean tensor of shape (n,) — True where the response is correct.
        """
        results = [self.verify(resp, gt) for resp, gt in zip(responses, ground_truths)]
        return torch.tensor(results, dtype=torch.bool)


# ---------------------------------------------------------------------------
# OutcomeRewardModel
# ---------------------------------------------------------------------------


class OutcomeRewardModel:
    """Assigns scalar rewards based on final-answer correctness.

    This is a *non-parametric* reward model: it uses a deterministic verifier
    rather than a learned neural network.  For the learned neural variant see
    ``outcome_reward.OutcomeRewardModel``.

    Args:
        verifier:       OutcomeVerifier used to check correctness.
        correct_reward: Reward returned for a correct answer (default 1.0).
        wrong_reward:   Reward returned for an incorrect answer (default 0.0).
        partial_credit: Reserved for future use; currently unused.
    """

    def __init__(
        self,
        verifier: OutcomeVerifier,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        partial_credit: bool = False,
    ) -> None:
        self.verifier = verifier
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.partial_credit = partial_credit

    def compute_reward(
        self,
        response_ids: torch.Tensor,
        ground_truth: Any,
    ) -> float:
        """Compute the outcome reward for a single response.

        Args:
            response_ids: Token tensor for the generated response.
            ground_truth: Reference answer.

        Returns:
            correct_reward if the response is correct, wrong_reward otherwise.
        """
        correct = self.verifier.verify(response_ids, ground_truth)
        return self.correct_reward if correct else self.wrong_reward

    def estimate_pass_at_k(
        self,
        responses: list[torch.Tensor],
        ground_truth: Any,
        k: int,
    ) -> float:
        """Estimate pass@k using the unbiased estimator.

        Args:
            responses:    List of n generated responses (token tensors).
            ground_truth: Reference answer.
            k:            Number of attempts to consider.

        Returns:
            Estimated probability in [0.0, 1.0] that at least one of k
            randomly-chosen responses is correct.
        """
        n = len(responses)
        correct_mask = self.verifier.batch_verify(responses, [ground_truth] * n)
        c = int(correct_mask.sum().item())
        return pass_at_k(n, c, k)

    def monte_carlo_reward(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        ground_truth: Any,
        n_samples: int = 16,
        temperature: float = 0.8,
    ) -> dict:
        """Estimate outcome quality via Monte Carlo sampling.

        Generates n_samples completions, verifies each, and computes
        pass@k statistics for k in {1, 2, 4, 8} (capped at n_samples).

        Args:
            model:        nn.Module with a ``generate`` method accepting
                          ``(prompt_ids, max_new_tokens, temperature)``.
            prompt_ids:   (1, T) or (T,) prompt token tensor.
            ground_truth: Reference answer.
            n_samples:    Number of completions to generate (default 16).
            temperature:  Sampling temperature (default 0.8).

        Returns:
            Dict with keys:
                'pass@1'        : float
                'pass@2'        : float  (if n_samples >= 2)
                'pass@4'        : float  (if n_samples >= 4)
                'pass@8'        : float  (if n_samples >= 8)
                'correct_count' : int
                'mean_reward'   : float
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # (1, T)

        responses: list[torch.Tensor] = []
        for _ in range(n_samples):
            with torch.no_grad():
                response = model.generate(
                    prompt_ids,
                    max_new_tokens=128,
                    temperature=temperature,
                )
            responses.append(response.squeeze(0))

        correct_mask = self.verifier.batch_verify(responses, [ground_truth] * n_samples)
        correct_count = int(correct_mask.sum().item())
        mean_reward = (
            correct_count / n_samples * self.correct_reward
            + (n_samples - correct_count) / n_samples * self.wrong_reward
        )

        result: dict = {
            "correct_count": correct_count,
            "mean_reward": mean_reward,
        }

        for k in (1, 2, 4, 8):
            if k <= n_samples:
                result[f"pass@{k}"] = pass_at_k(n_samples, correct_count, k)

        return result


# ---------------------------------------------------------------------------
# ORMTrainer
# ---------------------------------------------------------------------------


class ORMTrainer:
    """Trains a parametric reward model using binary cross-entropy on ORM labels.

    The reward model is trained to predict correctness (1 = correct, 0 = wrong)
    from padded response token sequences.

    Args:
        reward_model: nn.Module that maps (B, T) token ids to (B,) or (B, 1)
                      logit scores.
        optimizer:    PyTorch optimizer over reward_model.parameters().
        verifier:     OutcomeVerifier used to generate binary labels.
    """

    def __init__(
        self,
        reward_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        verifier: OutcomeVerifier,
    ) -> None:
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.verifier = verifier

    def create_training_batch(
        self,
        responses: list[torch.Tensor],
        ground_truths: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a padded batch with binary labels.

        Args:
            responses:     List of n response tensors (possibly different lengths).
            ground_truths: List of n ground-truth values.

        Returns:
            Tuple of:
                response_ids_padded: (n, max_len) long tensor, right-padded with 0.
                binary_labels:       (n,) float tensor of 0.0 / 1.0.
        """
        binary_labels = self.verifier.batch_verify(responses, ground_truths).float()

        max_len = max(r.shape[-1] for r in responses)
        padded = []
        for r in responses:
            seq = r.reshape(-1)  # flatten to 1-D
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                seq = F.pad(seq, (0, pad_len), value=0)
            padded.append(seq)

        response_ids_padded = torch.stack(padded, dim=0)  # (n, max_len)
        return response_ids_padded, binary_labels

    def train_step(
        self,
        response_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Perform one gradient update step.

        Args:
            response_ids: (B, T) padded token tensor.
            labels:       (B,) binary float labels (0.0 or 1.0).

        Returns:
            Dict with keys:
                'loss'          : float — binary cross-entropy
                'accuracy'      : float in [0, 1]
                'positive_rate' : float — fraction of positive labels in batch
        """
        self.reward_model.train()
        self.optimizer.zero_grad()

        logits = self.reward_model(response_ids)  # (B,) or (B, 1)
        logits = logits.squeeze(-1)  # ensure (B,)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            accuracy = (preds == labels).float().mean().item()
            positive_rate = labels.mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "positive_rate": positive_rate,
        }
