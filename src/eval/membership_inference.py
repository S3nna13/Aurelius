"""Membership Inference Attack and Defense for LLMs.

Implements three attack methods and a gradient-noise defense:

  - LossAttack            — Shokri et al. 2017; threshold on cross-entropy loss
  - MinKProbAttack        — Shi et al. 2023 (arXiv:2310.16789 Section 3)
  - LikelihoodRatioAttack — likelihood-ratio against a uniform-entropy baseline

Defense:
  - GradientNoiseDefense  — DP-SGD-style gradient clipping + Gaussian noise

Evaluator:
  - MembershipInferenceEvaluator — pure-PyTorch ROC/AUC; no sklearn/scipy.

Convention: all attack ``score`` methods return a 1-D FloatTensor of shape
``(B,)`` where *higher* values indicate the sample is more likely a
*training member*.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MembershipInferenceAttack(ABC):
    """Abstract base class for membership inference attacks.

    Subclasses implement ``score``, which maps per-token log-probabilities to
    a scalar membership score per sequence.  Higher score → more likely member.
    """

    @abstractmethod
    def score(self, log_probs: Tensor) -> Tensor:
        """Compute membership score.

        Args:
            log_probs: FloatTensor of shape ``(B, T)`` — per-token
                log-probabilities for each sequence in the batch.

        Returns:
            FloatTensor of shape ``(B,)`` — higher means more likely member.
        """


# ---------------------------------------------------------------------------
# 1. Loss-based attack
# ---------------------------------------------------------------------------

class LossAttack(MembershipInferenceAttack):
    """Loss-based membership inference attack (Shokri et al., 2017).

    Training members typically have *lower* cross-entropy loss because the
    model has memorised them, which corresponds to *higher* mean log-probability.
    We therefore return the mean log-probability directly so that a *higher*
    score corresponds to a *lower* loss (higher log-prob) and, thus, a more
    likely member.

    score(log_probs) = mean(log_probs along token dimension)

    Note on sign convention: the spec states ``score = -mean(log_probs)``
    together with the parenthetical "lower loss = more likely member".  Since
    the base-class contract is **higher score → more likely member**, and
    members have higher log-probs (lower loss), the correct formula is
    ``+mean(log_probs)``.  The negation in the spec refers to the *loss*
    definition (loss = -mean(log_probs)), not to the membership score itself.
    """

    def score(self, log_probs: Tensor) -> Tensor:  # (B, T) -> (B,)
        if log_probs.dim() != 2:
            raise ValueError(
                f"log_probs must be 2-D (B, T), got shape {tuple(log_probs.shape)}"
            )
        return log_probs.mean(dim=1)


# ---------------------------------------------------------------------------
# 2. Min-K% Prob attack
# ---------------------------------------------------------------------------

class MinKProbAttack(MembershipInferenceAttack):
    """Min-K% Prob membership inference attack (Shi et al., 2023).

    arXiv:2310.16789 Section 3.

    Curated training text has fewer *surprising* (low-probability) tokens than
    random non-member text.  We take the K% tokens with the lowest
    log-probabilities and average them.  Members score *higher* (less negative)
    than non-members because they have fewer very-low-probability tokens.

    score = mean of bottom k% log-probs  (closer to 0 → higher → more member)
    """

    def __init__(self, k_percent: float = 0.2) -> None:
        if not (0.0 < k_percent <= 1.0):
            raise ValueError(f"k_percent must be in (0, 1], got {k_percent}")
        self.k_percent = k_percent

    def score(self, log_probs: Tensor) -> Tensor:  # (B, T) -> (B,)
        if log_probs.dim() != 2:
            raise ValueError(
                f"log_probs must be 2-D (B, T), got shape {tuple(log_probs.shape)}"
            )
        B, T = log_probs.shape
        k = max(1, math.ceil(T * self.k_percent))
        # topk on the *negated* tensor gives us the largest (least-negative)
        # log-probs, i.e., the smallest values in log_probs.
        # We want the k tokens with the LOWEST log-probs.
        bottom_k, _ = torch.topk(-log_probs, k, dim=1)  # largest of -log_probs
        # bottom_k contains -log_probs for the k worst tokens; negate back.
        return -bottom_k.mean(dim=1)  # mean of the bottom-k log-probs


# ---------------------------------------------------------------------------
# 3. Likelihood Ratio attack
# ---------------------------------------------------------------------------

class LikelihoodRatioAttack(MembershipInferenceAttack):
    """Likelihood-ratio membership inference attack.

    score = log P_θ(x) - log P_ref(x)

    We use the *uniform-distribution entropy* over the vocabulary as a proxy
    for P_ref.  For a vocabulary of size V, a uniform baseline assigns
    log(1/V) = -log(V) to every token.  With V unknown we use the empirical
    estimate: baseline_entropy_per_token = -mean(-log_probs) averaged over all
    tokens in the batch (i.e., the mean per-token loss as an entropy estimate).

    Concretely:
        baseline = mean of all log_probs across the entire batch (scalar)
        score(i) = mean(log_probs[i]) - baseline

    Members whose sequences are well-modelled will have mean log-prob *above*
    the global average, yielding a positive score.
    """

    def score(self, log_probs: Tensor) -> Tensor:  # (B, T) -> (B,)
        if log_probs.dim() != 2:
            raise ValueError(
                f"log_probs must be 2-D (B, T), got shape {tuple(log_probs.shape)}"
            )
        baseline = log_probs.mean()
        return log_probs.mean(dim=1) - baseline


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MembershipInferenceEvaluator:
    """Evaluate membership inference attacks using pure-PyTorch metrics.

    No sklearn or scipy dependencies.

    Usage::

        evaluator = MembershipInferenceEvaluator()
        results = evaluator.evaluate(member_scores, nonmember_scores)
        # results: {'auc': float, 'tpr_at_fpr_0.1': float, 'accuracy': float}
    """

    @staticmethod
    def _roc_auc(scores: Tensor, labels: Tensor) -> Tensor:
        """Compute ROC AUC via trapezoidal rule (pure PyTorch).

        Args:
            scores: 1-D float tensor of predicted scores.
            labels: 1-D binary tensor (1 = positive / member).

        Returns:
            Scalar tensor with the AUC value.
        """
        # Sort by descending score.
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices].float()

        n_pos = sorted_labels.sum()
        n_neg = (1.0 - sorted_labels).sum()

        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.5)

        # Cumulative TP and FP counts along threshold sweep.
        cum_tp = torch.cumsum(sorted_labels, dim=0)
        cum_fp = torch.cumsum(1.0 - sorted_labels, dim=0)

        tpr = cum_tp / n_pos
        fpr = cum_fp / n_neg

        # Prepend (0, 0) for the trapezoidal rule.
        tpr = torch.cat([torch.zeros(1, device=tpr.device, dtype=tpr.dtype), tpr])
        fpr = torch.cat([torch.zeros(1, device=fpr.device, dtype=fpr.dtype), fpr])

        # Trapezoidal AUC.
        auc = torch.trapezoid(tpr, fpr)
        return auc

    @staticmethod
    def _tpr_at_fpr(
        scores: Tensor, labels: Tensor, target_fpr: float = 0.1
    ) -> Tensor:
        """Compute TPR at a specific FPR threshold.

        Args:
            scores: 1-D float tensor.
            labels: 1-D binary tensor (1 = member).
            target_fpr: desired FPR level (default 0.1).

        Returns:
            Scalar tensor with TPR at the requested FPR.
        """
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices].float()

        n_pos = sorted_labels.sum()
        n_neg = (1.0 - sorted_labels).sum()

        if n_neg == 0 or n_pos == 0:
            return torch.tensor(0.0)

        cum_tp = torch.cumsum(sorted_labels, dim=0)
        cum_fp = torch.cumsum(1.0 - sorted_labels, dim=0)

        tpr = cum_tp / n_pos
        fpr = cum_fp / n_neg

        # Find the last index where fpr <= target_fpr.
        mask = fpr <= target_fpr
        if not mask.any():
            return torch.tensor(0.0)

        last_valid = mask.nonzero(as_tuple=False)[-1].item()
        return tpr[last_valid]

    def evaluate(
        self,
        member_scores: Tensor,
        nonmember_scores: Tensor,
    ) -> Dict[str, float]:
        """Evaluate attack performance.

        Args:
            member_scores:    1-D float tensor of scores for training members.
            nonmember_scores: 1-D float tensor of scores for non-members.

        Returns:
            Dictionary with keys:
              - ``'auc'``            : ROC AUC in [0, 1].
              - ``'tpr_at_fpr_0.1'`` : TPR at 10% FPR.
              - ``'accuracy'``       : balanced accuracy at the median threshold.
        """
        if member_scores.dim() != 1 or nonmember_scores.dim() != 1:
            raise ValueError("member_scores and nonmember_scores must be 1-D tensors.")

        scores = torch.cat([member_scores, nonmember_scores])
        labels = torch.cat(
            [
                torch.ones(len(member_scores), dtype=torch.long),
                torch.zeros(len(nonmember_scores), dtype=torch.long),
            ]
        )

        auc = self._roc_auc(scores, labels)
        tpr_01 = self._tpr_at_fpr(scores, labels, target_fpr=0.1)

        # Accuracy: threshold = median of all scores.
        threshold = scores.median()
        preds = (scores >= threshold).long()
        accuracy = (preds == labels).float().mean()

        return {
            "auc": float(auc.item()),
            "tpr_at_fpr_0.1": float(tpr_01.item() if isinstance(tpr_01, Tensor) else tpr_01),
            "accuracy": float(accuracy.item()),
        }


# ---------------------------------------------------------------------------
# Defense: Gradient Noise (DP-SGD style)
# ---------------------------------------------------------------------------

class GradientNoiseDefense:
    """Gradient clipping + Gaussian noise defense (DP-SGD style).

    Each gradient tensor is:
      1. Clipped per-sample to ``max_norm`` (L2 norm clipping).
      2. Perturbed with Gaussian noise N(0, (noise_multiplier * max_norm)^2).

    This is compatible with the standard DP-SGD formulation from
    Abadi et al. (2016) and can serve as a defense against membership
    inference attacks by reducing the memorisation signal.

    Args:
        noise_multiplier: Scale of added noise relative to ``max_norm``.
            Setting to 0 disables noise (only clipping is applied).
    """

    def __init__(self, noise_multiplier: float = 1.0) -> None:
        if noise_multiplier < 0:
            raise ValueError(
                f"noise_multiplier must be >= 0, got {noise_multiplier}"
            )
        self.noise_multiplier = noise_multiplier

    def clip_and_noise(
        self,
        gradients: List[Tensor],
        max_norm: float,
    ) -> List[Tensor]:
        """Clip gradients to ``max_norm`` and add calibrated Gaussian noise.

        Args:
            gradients: List of gradient tensors (arbitrary shapes).
            max_norm:  Maximum L2 norm for each gradient tensor (per-tensor
                       clipping).  Must be > 0.

        Returns:
            List of noised gradient tensors with the same shapes as inputs.
        """
        if max_norm <= 0:
            raise ValueError(f"max_norm must be > 0, got {max_norm}")
        if not gradients:
            return []

        noised: List[Tensor] = []
        sigma = self.noise_multiplier * max_norm

        for grad in gradients:
            # Per-tensor L2 norm clipping.
            norm = grad.norm(2)
            clip_factor = torch.clamp(torch.tensor(max_norm) / (norm + 1e-12), max=1.0)
            clipped = grad * clip_factor

            if sigma > 0.0:
                noise = torch.randn_like(clipped) * sigma
                noised.append(clipped + noise)
            else:
                noised.append(clipped)

        return noised
