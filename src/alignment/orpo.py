"""Odds Ratio Preference Optimization helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _log_odds_from_log_probs(log_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    max_log_prob = math.log(1.0 - eps)
    probs = torch.exp(log_probs.clamp(max=max_log_prob)).clamp(min=eps, max=1.0 - eps)
    return torch.log(probs) - torch.log1p(-probs)


@dataclass(frozen=True)
class ORPOMetrics:
    loss: torch.Tensor
    nll: torch.Tensor
    preference_term: torch.Tensor
    log_odds_ratio: torch.Tensor


def orpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_nll: torch.Tensor | None = None,
    alpha: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the ORPO objective."""
    return orpo_metrics(
        chosen_logps=chosen_logps,
        rejected_logps=rejected_logps,
        chosen_nll=chosen_nll,
        alpha=alpha,
        reduction=reduction,
    ).loss


def orpo_metrics(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_nll: torch.Tensor | None = None,
    alpha: float = 1.0,
    reduction: str = "mean",
) -> ORPOMetrics:
    """Return ORPO loss and its components."""
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError(
            f"chosen_logps and rejected_logps must match, got {chosen_logps.shape} and {rejected_logps.shape}"
        )
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    if chosen_nll is None:
        chosen_nll = torch.zeros_like(chosen_logps)
    elif chosen_nll.shape != chosen_logps.shape:
        raise ValueError(
            f"chosen_nll must match log-prob shape, got {chosen_nll.shape} and {chosen_logps.shape}"
        )

    chosen_log_odds = _log_odds_from_log_probs(chosen_logps)
    rejected_log_odds = _log_odds_from_log_probs(rejected_logps)
    log_odds_ratio = chosen_log_odds - rejected_log_odds
    preference_term = -F.logsigmoid(log_odds_ratio)
    losses = chosen_nll + alpha * preference_term

    if reduction == "mean":
        loss = losses.mean()
    elif reduction == "sum":
        loss = losses.sum()
    elif reduction == "none":
        loss = losses
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return ORPOMetrics(
        loss=loss,
        nll=chosen_nll,
        preference_term=preference_term,
        log_odds_ratio=log_odds_ratio,
    )
