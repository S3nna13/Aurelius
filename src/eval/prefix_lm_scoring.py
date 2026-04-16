"""Prefix LM scoring for few-shot evaluation (LAMBADA, HellaSwag, etc.).

Computes log-probabilities of completions conditioned on prefixes using
pure PyTorch — no external evaluation libraries required.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_VALID_REDUCTIONS = {"mean", "sum", "none"}


@dataclass
class ScoringConfig:
    """Configuration for prefix-LM completion scoring.

    Attributes:
        reduction: How to aggregate per-token log-probs over the completion.
            One of "mean" (default), "sum", or "none".
        log_base: Logarithm base used for reported scores (default 2.0).
        normalize_by_length: Whether to normalise scores by completion length
            when reduction="mean" (default True).  When reduction is not
            "mean" this flag has no additional effect.
    """
    reduction: str = "mean"
    log_base: float = 2.0
    normalize_by_length: bool = True

    def __post_init__(self) -> None:
        if self.reduction not in _VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {_VALID_REDUCTIONS}, got {self.reduction!r}"
            )
        if self.log_base <= 0:
            raise ValueError(f"log_base must be positive, got {self.log_base}")


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def compute_token_log_probs(logits: Tensor, token_ids: Tensor) -> Tensor:
    """Compute per-token log-probabilities from raw logits.

    Args:
        logits:    Float tensor of shape (T, vocab_size).
        token_ids: Long tensor of shape (T,) containing ground-truth token ids.

    Returns:
        Float tensor of shape (T,) where position t holds
        log p(token_ids[t] | logits[t]) in natural log (nats).
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (T, vocab), got shape {logits.shape}")
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"token_ids must be 1-D with length T={logits.shape[0]}, "
            f"got shape {token_ids.shape}"
        )

    log_probs = F.log_softmax(logits, dim=-1)           # (T, vocab)
    gathered = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # (T,)
    return gathered


def score_completion(
    prefix_logprobs: Tensor,
    completion_mask: Tensor,
    config: ScoringConfig,
) -> Tensor:
    """Aggregate per-token log-probs over completion positions.

    Args:
        prefix_logprobs: Float tensor of shape (T,), per-token log-probs in nats.
        completion_mask: Boolean tensor of shape (T,); True at completion positions.
        config:          ScoringConfig controlling reduction and log base.

    Returns:
        * reduction="mean" or "sum": scalar tensor.
        * reduction="none": 1-D tensor of length n_completion_tokens.
    All values are expressed in log_base (not nats).
    """
    if prefix_logprobs.ndim != 1:
        raise ValueError(f"prefix_logprobs must be 1-D, got shape {prefix_logprobs.shape}")
    if completion_mask.shape != prefix_logprobs.shape:
        raise ValueError(
            f"completion_mask shape {completion_mask.shape} must match "
            f"prefix_logprobs shape {prefix_logprobs.shape}"
        )

    # Natural-log-to-log_base conversion factor: log_b(x) = ln(x) / ln(b)
    ln_base = math.log(config.log_base)
    conversion = 1.0 / ln_base

    completion_lp = prefix_logprobs[completion_mask]  # (n_completion,)

    if config.reduction == "none":
        return completion_lp * conversion

    if config.reduction == "sum":
        return (completion_lp.sum() * conversion)

    # reduction == "mean"
    if completion_lp.numel() == 0:
        return torch.tensor(float("-inf"))
    return (completion_lp.mean() * conversion)


def rank_completions(scores: Tensor) -> Tensor:
    """Rank completions from best (0) to worst.

    Args:
        scores: 1-D float tensor of shape (n_completions,).

    Returns:
        Long tensor of shape (n_completions,) where rank 0 = highest score.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    # argsort ascending, then argsort again → rank (0 = smallest descending index)
    descending_order = torch.argsort(scores, descending=True)  # indices sorted best→worst
    ranks = torch.empty_like(descending_order)
    ranks[descending_order] = torch.arange(len(scores), device=scores.device)
    return ranks


# ---------------------------------------------------------------------------
# High-level scorer
# ---------------------------------------------------------------------------

class PrefixLMScorer:
    """Score and rank completions using a causal language model.

    The model_fn is a callable that accepts a 1-D or 2-D integer input_ids
    tensor and returns logits of shape (..., T, vocab_size).
    """

    def __init__(self, model_fn: Callable, config: ScoringConfig) -> None:
        self.model_fn = model_fn
        self.config = config

    def score(self, input_ids: Tensor, completion_start: int) -> Tensor:
        """Score a single (prefix + completion) sequence.

        Args:
            input_ids:        1-D long tensor of shape (T,).
            completion_start: Index (inclusive) where the completion begins.

        Returns:
            Scalar tensor — the score of the completion in log_base.
        """
        logits = self.model_fn(input_ids)  # (T, vocab) or (1, T, vocab)
        if logits.ndim == 3:
            logits = logits.squeeze(0)
        token_lp = compute_token_log_probs(logits, input_ids)

        mask = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        mask[completion_start:] = True

        return score_completion(token_lp, mask, self.config)

    def score_batch(
        self,
        inputs: List[Tensor],
        completion_starts: List[int],
    ) -> Tensor:
        """Score a batch of (prefix + completion) sequences independently.

        Args:
            inputs:             List of 1-D long tensors, one per completion.
            completion_starts:  Corresponding completion start indices.

        Returns:
            Float tensor of shape (n,) with one score per input.
        """
        scores = [
            self.score(inp, start)
            for inp, start in zip(inputs, completion_starts)
        ]
        return torch.stack(scores)

    def rank(
        self,
        inputs: List[Tensor],
        completion_starts: List[int],
    ) -> Tensor:
        """Score then rank completions (rank 0 = best / highest score).

        Returns:
            Long tensor of shape (n,) with ranks.
        """
        scores = self.score_batch(inputs, completion_starts)
        return rank_completions(scores)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def perplexity_from_log_probs(log_probs: Tensor, base: float = 2.0) -> float:
    """Compute perplexity from per-token log-probabilities.

    Args:
        log_probs: 1-D float tensor of per-token log-probs in the given base.
        base:      The logarithm base used in log_probs (default 2.0).

    Returns:
        Perplexity = base^(-mean(log_probs)) as a Python float.
    """
    mean_lp = log_probs.mean().item()
    return float(base ** (-mean_lp))
