"""Contrastive Decoding (Li et al., 2022) — suppress amateur model artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveDecodeConfig:
    """Configuration for Contrastive Decoding."""

    alpha: float = 0.1  # plausibility threshold
    temperature: float = 1.0  # sampling temperature
    max_new_tokens: int = 64


def compute_plausibility_mask(
    expert_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return boolean mask: True where expert_prob >= alpha * max(expert_prob).

    Works on last-dim logits of shape (..., V). Converts to probs internally.

    Args:
        expert_logits: Raw logits, shape (B, V) or (V,).
        alpha: Plausibility threshold factor in (0, 1].

    Returns:
        Boolean tensor of same shape: True for plausible tokens.
    """
    expert_probs = F.softmax(expert_logits, dim=-1)
    max_prob = expert_probs.max(dim=-1, keepdim=True).values
    return expert_probs >= alpha * max_prob


def contrastive_score(
    expert_logits: torch.Tensor,
    amateur_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute contrastive score: log_softmax(expert) - log_softmax(amateur), masked.

    Args:
        expert_logits: (B, V) raw logits from expert model.
        amateur_logits: (B, V) raw logits from amateur model.
        mask: (B, V) boolean plausibility mask.

    Returns:
        Modified scores of shape (B, V). Implausible positions set to -inf.
    """
    expert_log_probs = F.log_softmax(expert_logits, dim=-1)
    amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)
    score = expert_log_probs - amateur_log_probs
    score = score.masked_fill(~mask, float("-inf"))
    return score


def adaptive_alpha(step: int, total_steps: int) -> float:
    """Linearly increase alpha from 0.05 to 0.5 over generation.

    Args:
        step: Current generation step (0-indexed).
        total_steps: Total number of generation steps.

    Returns:
        Interpolated alpha value as a float.
    """
    if total_steps <= 1:
        return 0.05
    t = step / (total_steps - 1)
    return 0.05 + t * (0.5 - 0.05)


def contrastive_step(
    expert_model: nn.Module,
    amateur_model: nn.Module,
    input_ids: torch.Tensor,
    config: ContrastiveDecodeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One generation step of contrastive decoding.

    Runs both models, computes contrastive scores, samples next token.

    Args:
        expert_model: Expert AureliusTransformer.
        amateur_model: Amateur AureliusTransformer.
        input_ids: (B, T) current sequence.
        config: ContrastiveDecodeConfig.

    Returns:
        (next_token, score) where next_token is (B, 1) and score is (B, V).
    """
    _, expert_logits, _ = expert_model(input_ids)
    _, amateur_logits, _ = amateur_model(input_ids)

    # Last position logits: (B, V)
    e_logits = expert_logits[:, -1, :]
    a_logits = amateur_logits[:, -1, :]

    mask = compute_plausibility_mask(e_logits, config.alpha)
    scores = contrastive_score(e_logits, a_logits, mask)

    # Apply temperature and sample
    scaled = scores / max(config.temperature, 1e-8)
    # Replace -inf with very large negative for softmax stability
    probs = F.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

    return next_token, scores


class ContrastiveDecoder:
    """Contrastive Decoding generator combining expert and amateur models.

    Args:
        expert_model: The stronger (expert) AureliusTransformer.
        amateur_model: The weaker (amateur) AureliusTransformer.
        config: ContrastiveDecodeConfig controlling generation hyperparameters.
    """

    def __init__(
        self,
        expert_model: nn.Module,
        amateur_model: nn.Module,
        config: ContrastiveDecodeConfig,
    ) -> None:
        self.expert = expert_model
        self.amateur = amateur_model
        self.config = config

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run autoregressive contrastive decoding.

        Args:
            input_ids: Prompt token ids, shape (B, T).

        Returns:
            Full sequence (prompt + generated) token tensor of shape (B, T + max_new_tokens).
        """
        seq = input_ids.clone()
        n = self.config.max_new_tokens

        for step in range(n):
            _, expert_logits, _ = self.expert(seq)
            _, amateur_logits, _ = self.amateur(seq)

            e_logits = expert_logits[:, -1, :]
            a_logits = amateur_logits[:, -1, :]

            alpha = adaptive_alpha(step, n)
            mask = compute_plausibility_mask(e_logits, alpha)
            scores = contrastive_score(e_logits, a_logits, mask)

            scaled = scores / max(self.config.temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            seq = torch.cat([seq, next_token], dim=1)

        return seq
