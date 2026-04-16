"""Aurelius — Chain-of-Hindsight (CoH) fine-tuning.

Implementation of Liu et al. (2023) Chain-of-Hindsight alignment method.
Trains models to improve their outputs using explicit feedback, using sequences
of the form: [prompt | bad_response | feedback | good_response].

This is a pure supervised-learning alignment method with no RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoHConfig:
    """Configuration for Chain-of-Hindsight training."""

    # Loss weighting for good response tokens
    coh_weight: float = 1.0

    # Feedback type: 'scalar' | 'text' | 'ranking'
    feedback_type: str = "scalar"

    # Sequence length limits
    max_bad_response_len: int = 256
    max_good_response_len: int = 512

    # Minimum reward gap to form a useful training pair
    min_reward_gap: float = 0.1


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Default feedback separator token id (used when no feedback_ids provided)
_DEFAULT_SEPARATOR_TOKEN_ID: int = 1


def rank_responses(
    responses: List[str],
    rewards: List[float],
) -> List[Tuple[str, float]]:
    """Sort (response, reward) pairs by reward ascending (worst first).

    Args:
        responses: List of response strings.
        rewards:   Corresponding reward scores.

    Returns:
        List of (response, reward) tuples sorted by reward ascending.
    """
    paired = list(zip(responses, rewards))
    paired.sort(key=lambda x: x[1])
    return paired


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def create_hindsight_dataset(
    prompts: List[str],
    responses_list: List[List[str]],
    rewards_list: List[List[float]],
    config: CoHConfig,
) -> List[dict]:
    """Build CoH training examples from prompts with multiple ranked responses.

    For each prompt, picks the worst and best responses and creates a CoH
    training example.  Pairs where reward_gap < config.min_reward_gap are
    filtered out.

    Args:
        prompts:        List of prompt strings.
        responses_list: List of response lists (one per prompt).
        rewards_list:   List of reward lists aligned with responses_list.
        config:         CoHConfig instance.

    Returns:
        List of dicts with keys: input_ids, labels, reward_gap.
        All tensors are 1-D LongTensors of token ids (using ordinal encoding
        of chars for the text fields, as a stand-in for a real tokenizer).
    """
    examples: List[dict] = []

    for prompt, responses, rewards in zip(prompts, responses_list, rewards_list):
        if len(responses) < 2:
            continue

        ranked = rank_responses(responses, rewards)
        worst_resp, worst_reward = ranked[0]
        best_resp, best_reward = ranked[-1]

        reward_gap = best_reward - worst_reward
        if reward_gap < config.min_reward_gap:
            continue

        # Build token id sequences using simple char-ordinal encoding
        prompt_ids = _text_to_ids(prompt)
        bad_ids = _text_to_ids(worst_resp)[: config.max_bad_response_len]
        good_ids = _text_to_ids(best_resp)[: config.max_good_response_len]

        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)
        bad_t = torch.tensor(bad_ids, dtype=torch.long)
        good_t = torch.tensor(good_ids, dtype=torch.long)

        input_ids, labels = _build_coh_sequence_tensors(
            prompt_t, bad_t, good_t, feedback_ids=None
        )

        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "reward_gap": reward_gap,
            }
        )

    return examples


def _text_to_ids(text: str) -> List[int]:
    """Convert text to a list of token ids via char ordinals (stand-in tokenizer)."""
    return [ord(c) % 512 for c in text]


# ---------------------------------------------------------------------------
# Low-level sequence builder (shared between CoHTrainer and dataset builder)
# ---------------------------------------------------------------------------

def _build_coh_sequence_tensors(
    prompt_ids: torch.Tensor,
    bad_response_ids: torch.Tensor,
    good_response_ids: torch.Tensor,
    feedback_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate CoH sequence and build labels mask.

    Layout: [prompt | bad_response | feedback | good_response]

    Labels are -100 (masked) everywhere except the good_response portion,
    so loss is only computed on good_response tokens.

    Args:
        prompt_ids:       1-D LongTensor.
        bad_response_ids: 1-D LongTensor.
        good_response_ids:1-D LongTensor.
        feedback_ids:     Optional 1-D LongTensor; if None, a single default
                          separator token is inserted.

    Returns:
        (input_ids, labels) — both 1-D LongTensors of the same length.
    """
    if feedback_ids is None:
        feedback_ids = torch.tensor(
            [_DEFAULT_SEPARATOR_TOKEN_ID], dtype=torch.long
        )

    # Full sequence
    input_ids = torch.cat(
        [prompt_ids, bad_response_ids, feedback_ids, good_response_ids], dim=0
    )

    # Labels: mask all except good_response
    good_start = len(prompt_ids) + len(bad_response_ids) + len(feedback_ids)
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[good_start:] = good_response_ids

    return input_ids, labels


# ---------------------------------------------------------------------------
# CoHTrainer
# ---------------------------------------------------------------------------

class CoHTrainer:
    """Chain-of-Hindsight trainer.

    Constructs CoH training sequences and computes the supervised loss that
    teaches the model to produce better responses after seeing bad ones with
    explicit feedback.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_pad_id: int = 0,
        feedback_token_id: Optional[int] = None,
        coh_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer_pad_id = tokenizer_pad_id
        self.feedback_token_id = feedback_token_id
        self.coh_weight = coh_weight

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def build_coh_sequence(
        self,
        prompt_ids: torch.Tensor,
        bad_response_ids: torch.Tensor,
        good_response_ids: torch.Tensor,
        feedback_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a CoH training sequence and its label mask.

        Concatenation order: [prompt | bad_response | feedback | good_response]

        Args:
            prompt_ids:        1-D LongTensor of prompt token ids.
            bad_response_ids:  1-D LongTensor of low-reward response token ids.
            good_response_ids: 1-D LongTensor of high-reward response token ids.
            feedback_ids:      Optional 1-D LongTensor of feedback token ids.
                               If None, a single separator token is used.

        Returns:
            (input_ids, labels):
                input_ids — 1-D LongTensor of the full concatenated sequence.
                labels    — 1-D LongTensor, -100 everywhere except good_response.
        """
        if feedback_ids is None and self.feedback_token_id is not None:
            feedback_ids = torch.tensor(
                [self.feedback_token_id], dtype=torch.long
            )

        return _build_coh_sequence_tensors(
            prompt_ids, bad_response_ids, good_response_ids, feedback_ids
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_coh_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss over non-masked positions only.

        Args:
            logits: Shape (..., seq_len, vocab_size) — model output logits.
            labels: Shape (..., seq_len) — -100 means ignore; other values are
                    valid target token ids (the good_response region).

        Returns:
            Scalar loss tensor weighted by coh_weight.
        """
        # Flatten to 2-D for F.cross_entropy
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)

        # cross_entropy with ignore_index=-100 naturally skips masked tokens
        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        return self.coh_weight * loss

    # ------------------------------------------------------------------
    # Feedback text generation
    # ------------------------------------------------------------------

    def build_hindsight_feedback(
        self,
        reward_score: float,
        threshold: float = 0.5,
    ) -> str:
        """Return a feedback string based on a scalar reward score.

        Args:
            reward_score: Scalar reward value.
            threshold:    Decision boundary; scores >= threshold are "good".

        Returns:
            Feedback string.
        """
        if reward_score >= threshold:
            return "This is a good response."
        return "This response can be improved."

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        responses: List[torch.Tensor],
        rewards: List[float],
    ) -> dict:
        """Perform a single CoH training step.

        Sorts responses by reward (worst → best), then builds a CoH sequence
        using the worst as the "bad" response and the best as the "good"
        response.

        Args:
            prompt_ids: 1-D LongTensor of prompt token ids.
            responses:  List of 1-D LongTensors, one per candidate response.
            rewards:    List of scalar reward scores aligned with *responses*.

        Returns:
            dict with keys:
                loss           — scalar float.
                n_tokens       — number of good_response tokens used for loss.
                mean_reward_gap — float (best_reward - worst_reward).
        """
        if len(responses) < 2:
            raise ValueError("train_step requires at least 2 responses.")

        # Sort by reward ascending: index 0 = worst, -1 = best
        order = sorted(range(len(rewards)), key=lambda i: rewards[i])
        sorted_responses = [responses[i] for i in order]
        sorted_rewards = [rewards[i] for i in order]

        bad_response_ids = sorted_responses[0]
        good_response_ids = sorted_responses[-1]
        mean_reward_gap = sorted_rewards[-1] - sorted_rewards[0]

        input_ids, labels = self.build_coh_sequence(
            prompt_ids, bad_response_ids, good_response_ids
        )

        # Forward pass: model expects (B, seq_len), so add batch dim
        input_ids_b = input_ids.unsqueeze(0)  # (1, seq_len)
        labels_b = labels.unsqueeze(0)        # (1, seq_len)

        # Model forward; support both (loss, logits, ...) and plain logits returns
        output = self.model(input_ids_b)
        if isinstance(output, tuple):
            logits = output[1]
        else:
            logits = output

        # Shift logits/labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels_b[:, 1:].contiguous()

        # n_tokens = number of valid (non-masked) positions after shifting
        n_tokens = int((shift_labels != -100).sum().item())

        loss = self.compute_coh_loss(shift_logits, shift_labels)

        return {
            "loss": loss,
            "n_tokens": n_tokens,
            "mean_reward_gap": mean_reward_gap,
        }
