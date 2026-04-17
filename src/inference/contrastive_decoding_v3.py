"""
Contrastive Decoding (Li et al. 2022) — pure PyTorch implementation.

Key idea: improve text quality by subtracting an "amateur" model's log-probs
from the "expert" model's, then applying an adaptive plausibility constraint
so only tokens that are already plausible under the expert are considered.

Reference: https://arxiv.org/abs/2210.15097
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adaptive Plausibility Filter
# ---------------------------------------------------------------------------

class AdaptivePlausibilityFilter(nn.Module):
    """Filter tokens to the plausible set for a given alpha threshold.

    A token x is plausible at position t if:
        p_expert(x | context) >= alpha * max_x' p_expert(x' | context)

    Args:
        alpha: threshold in (0, 1).  Smaller alpha → larger candidate set.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def forward(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """Return boolean mask (True = plausible) of shape (batch, vocab).

        Implausible positions in expert_logits are NOT modified here —
        the mask is the primary output.  Callers use it to set scores to -inf.

        Args:
            expert_logits: (B, V) raw logits from the expert model.

        Returns:
            mask: (B, V) bool tensor; True where the token is plausible.
        """
        probs = F.softmax(expert_logits, dim=-1)          # (B, V)
        max_prob = probs.max(dim=-1, keepdim=True).values  # (B, 1)
        mask = probs >= self.alpha * max_prob              # (B, V)
        return mask


# ---------------------------------------------------------------------------
# Contrastive Logits
# ---------------------------------------------------------------------------

class ContrastiveLogits(nn.Module):
    """Core contrastive scoring module.

    score(x) = log_softmax_expert(x) - log_softmax_amateur(x)

    only for tokens that pass the adaptive plausibility filter; all others
    are set to -inf so they can never be sampled.

    Args:
        alpha: plausibility threshold forwarded to AdaptivePlausibilityFilter.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.filter = AdaptivePlausibilityFilter(alpha=alpha)

    def forward(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive logits.

        Args:
            expert_logits:  (B, V) raw expert logits.
            amateur_logits: (B, V) raw amateur logits.

        Returns:
            scores: (B, V) contrastive scores; implausible positions are -inf.
        """
        mask = self.filter(expert_logits)                  # (B, V) bool

        log_p_expert = F.log_softmax(expert_logits, dim=-1)
        log_p_amateur = F.log_softmax(amateur_logits, dim=-1)

        scores = log_p_expert - log_p_amateur              # (B, V)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores


# ---------------------------------------------------------------------------
# Amateur Model Wrapper
# ---------------------------------------------------------------------------

class AmateurModelWrapper(nn.Module):
    """Lightweight wrapper that scales amateur logits by 1/temperature.

    A higher temperature on the amateur (> 1) makes its distribution flatter,
    which reduces contrast.  A temperature < 1 sharpens it and increases
    contrast — but the wrapper divides by temperature, so temperature < 1
    actually *amplifies* logits (sharper amateur, better contrast).

    Args:
        model:       callable (input_ids) → (B, T, V) logits.
        temperature: scales amateur logits DOWN (logits / temperature).
    """

    def __init__(self, model: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.model = model
        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(input_ids) / self.temperature

    def freeze(self) -> None:
        """Freeze all parameters so they are excluded from gradient updates."""
        for param in self.model.parameters():
            param.requires_grad_(False)


# ---------------------------------------------------------------------------
# Contrastive Decoder
# ---------------------------------------------------------------------------

class ContrastiveDecoder(nn.Module):
    """Autoregressive text generation using contrastive decoding.

    At each step:
      1. Run expert and amateur models on the current token sequence.
      2. Take the logits at the *last* position.
      3. Compute contrastive scores via ContrastiveLogits.
      4. Apply temperature scaling to contrastive scores.
      5. Greedily pick argmax (temperature just re-scales, argmax unchanged
         unless you add stochastic sampling — left as deterministic here).
      6. Append the chosen token and repeat.

    Both models run under torch.no_grad() (inference only).

    Args:
        expert_model:  callable (B, T) → (B, T, V).
        amateur_model: callable (B, T) → (B, T, V).
        alpha:         plausibility threshold.
        temperature:   applied to contrastive scores before argmax.
    """

    def __init__(
        self,
        expert_model: Callable[[torch.Tensor], torch.Tensor],
        amateur_model: Callable[[torch.Tensor], torch.Tensor],
        alpha: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.expert_model = expert_model
        self.amateur_model = amateur_model
        self.scorer = ContrastiveLogits(alpha=alpha)
        self.temperature = temperature

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Generate tokens autoregressively using contrastive decoding.

        Args:
            input_ids:      (B, T) integer token ids.
            max_new_tokens: number of tokens to generate.

        Returns:
            (B, T + max_new_tokens) token ids.
        """
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # --- expert ---
                expert_out = self.expert_model(generated)   # (B, T', V)
                expert_logits = expert_out[:, -1, :]        # (B, V)

                # --- amateur ---
                amateur_out = self.amateur_model(generated) # (B, T', V)
                amateur_logits = amateur_out[:, -1, :]      # (B, V)

                # --- contrastive scores ---
                scores = self.scorer(expert_logits, amateur_logits)  # (B, V)
                scores = scores / self.temperature

                # greedy decoding: argmax over plausible tokens
                next_token = scores.argmax(dim=-1, keepdim=True)    # (B, 1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveDecodingConfig:
    """Configuration for contrastive decoding."""

    alpha: float = 0.1
    temperature: float = 1.0
    amateur_temperature: float = 1.0
    max_new_tokens: int = 50

    def validate(self) -> None:
        """Validate all configuration fields.

        Raises:
            AssertionError: if any field is out of range.
        """
        assert 0.0 < self.alpha < 1.0, (
            f"alpha must be in (0, 1), got {self.alpha}"
        )
        assert self.temperature > 0.0, (
            f"temperature must be > 0, got {self.temperature}"
        )
        assert self.amateur_temperature > 0.0, (
            f"amateur_temperature must be > 0, got {self.amateur_temperature}"
        )


# ---------------------------------------------------------------------------
# Contrastive Scoring Metrics
# ---------------------------------------------------------------------------

class ContrastiveScoringMetrics:
    """Evaluate quality of contrastive vs. greedy decoding outputs."""

    def __init__(self) -> None:
        pass

    def vocabulary_diversity(self, generated_ids: torch.Tensor) -> float:
        """Ratio of unique tokens to total tokens.

        Args:
            generated_ids: (B, T) or (T,) integer token ids.

        Returns:
            float in [0, 1].
        """
        flat = generated_ids.reshape(-1)
        if flat.numel() == 0:
            return 0.0
        unique = flat.unique().numel()
        return float(unique) / float(flat.numel())

    def top1_probability(self, logits: torch.Tensor) -> float:
        """Mean maximum softmax probability across all positions.

        Args:
            logits: (B, V) or (B, T, V).

        Returns:
            float in [0, 1].
        """
        if logits.dim() == 3:
            # flatten batch and time
            B, T, V = logits.shape
            logits = logits.reshape(B * T, V)
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values   # (N,)
        return float(max_probs.mean().item())

    def score_gap(self, contrastive_logits: torch.Tensor) -> float:
        """Mean gap between top-1 and top-2 contrastive scores.

        Larger gap → the model is more decisive.

        Args:
            contrastive_logits: (B, V) scores (may contain -inf).

        Returns:
            float >= 0.
        """
        # Replace -inf with a large negative finite value for topk
        finite = contrastive_logits.clone()
        finite[contrastive_logits == float("-inf")] = -1e9

        if finite.shape[-1] < 2:
            return 0.0

        top2 = torch.topk(finite, k=2, dim=-1).values  # (B, 2)
        gaps = top2[:, 0] - top2[:, 1]                  # (B,)
        return float(gaps.mean().item())
