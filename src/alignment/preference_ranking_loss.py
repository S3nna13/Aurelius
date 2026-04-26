"""Preference ranking losses for reward-model training and DPO variants.

Pure PyTorch. No foreign imports.

Provides:
- bradley_terry_loss: pairwise -log sigmoid(r_chosen - r_rejected)
- margin_ranking_loss: hinge-style with configurable margin
- listnet_loss: listwise cross-entropy over preference distributions (Cao 2007)
- ordinal_ranking_loss: pairwise listwise loss consistent with integer ranks
- dpo_pair_loss: DPO implicit reward loss over policy/reference logprobs
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _check_same_shape(a: Tensor, b: Tensor, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: {name_a} has shape {tuple(a.shape)} but "
            f"{name_b} has shape {tuple(b.shape)}"
        )


def bradley_terry_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> Tensor:
    """Pairwise Bradley-Terry preference loss.

    loss = mean -log sigmoid(r_chosen - r_rejected)

    Uses softplus for numerical stability: -log sigmoid(x) = softplus(-x).
    """
    _check_same_shape(chosen_rewards, rejected_rewards, "chosen_rewards", "rejected_rewards")
    diff = chosen_rewards - rejected_rewards
    return F.softplus(-diff).mean()


def margin_ranking_loss(
    chosen_rewards: Tensor, rejected_rewards: Tensor, margin: float = 1.0
) -> Tensor:
    """Hinge margin ranking loss.

    loss = mean max(0, margin - (r_chosen - r_rejected))
    """
    _check_same_shape(chosen_rewards, rejected_rewards, "chosen_rewards", "rejected_rewards")
    diff = chosen_rewards - rejected_rewards
    return torch.clamp(margin - diff, min=0.0).mean()


def listnet_loss(predicted_scores: Tensor, true_scores: Tensor) -> Tensor:
    """ListNet listwise cross-entropy loss (Cao 2007).

    Both inputs shape [B, K]. Computes cross-entropy between softmax(true) and
    softmax(predicted), averaged over batch.
    """
    if predicted_scores.dim() != 2 or true_scores.dim() != 2:
        raise ValueError(
            f"listnet_loss expects 2D tensors [B, K], got shapes "
            f"{tuple(predicted_scores.shape)} and {tuple(true_scores.shape)}"
        )
    _check_same_shape(predicted_scores, true_scores, "predicted_scores", "true_scores")
    true_prob = F.softmax(true_scores, dim=-1)
    pred_log_prob = F.log_softmax(predicted_scores, dim=-1)
    # cross-entropy: -sum true_prob * log pred_prob
    per_row = -(true_prob * pred_log_prob).sum(dim=-1)
    return per_row.mean()


def ordinal_ranking_loss(predicted: Tensor, ranks: Tensor) -> Tensor:
    """Listwise ordinal ranking loss.

    For each row in [B, K], for every pair (i, j) where ranks[i] < ranks[j]
    (i.e. i is preferred over j -- lower rank index = better), penalize
    -log sigmoid(predicted[i] - predicted[j]).

    Returns mean over valid pairs and batch.
    """
    if predicted.dim() != 2 or ranks.dim() != 2:
        raise ValueError(
            f"ordinal_ranking_loss expects 2D tensors [B, K], got shapes "
            f"{tuple(predicted.shape)} and {tuple(ranks.shape)}"
        )
    _check_same_shape(predicted, ranks, "predicted", "ranks")

    # pred_i - pred_j for all pairs: [B, K, K]
    pred_diff = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
    ranks_f = ranks.to(pred_diff.dtype)
    # mask: rank_i < rank_j (i preferred over j)
    pair_mask = (ranks_f.unsqueeze(-1) < ranks_f.unsqueeze(-2)).to(pred_diff.dtype)
    # -log sigmoid(pred_i - pred_j) via softplus(-x)
    pair_loss = F.softplus(-pred_diff) * pair_mask
    denom = pair_mask.sum().clamp_min(1.0)
    return pair_loss.sum() / denom


def dpo_pair_loss(
    policy_chosen_logprobs: Tensor,
    policy_rejected_logprobs: Tensor,
    ref_chosen_logprobs: Tensor,
    ref_rejected_logprobs: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """DPO-style implicit reward loss.

    loss = mean -log sigmoid( beta * (
        (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    ))
    """
    _check_same_shape(
        policy_chosen_logprobs,
        policy_rejected_logprobs,
        "policy_chosen_logprobs",
        "policy_rejected_logprobs",
    )
    _check_same_shape(
        ref_chosen_logprobs,
        ref_rejected_logprobs,
        "ref_chosen_logprobs",
        "ref_rejected_logprobs",
    )
    _check_same_shape(
        policy_chosen_logprobs,
        ref_chosen_logprobs,
        "policy_chosen_logprobs",
        "ref_chosen_logprobs",
    )
    chosen_logratio = policy_chosen_logprobs - ref_chosen_logprobs
    rejected_logratio = policy_rejected_logprobs - ref_rejected_logprobs
    logits = beta * (chosen_logratio - rejected_logratio)
    return F.softplus(-logits).mean()


__all__ = [
    "bradley_terry_loss",
    "margin_ranking_loss",
    "listnet_loss",
    "ordinal_ranking_loss",
    "dpo_pair_loss",
]
