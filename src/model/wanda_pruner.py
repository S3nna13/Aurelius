"""Wanda pruning: activation-aware weight pruning via |W| * ||X||_2 scoring.

Reference: Sun et al. 2023 — "A Simple and Effective Pruning Approach for Large
Language Models" (https://arxiv.org/abs/2306.11695).

Wanda scores each weight W_ij as:
    score(W_ij) = |W_ij| * ||X_j||_2

where ||X_j||_2 is the L2 norm of column j of the calibration input activations,
averaged over all calibration samples.  This activation-aware criterion is more
effective than magnitude-only pruning because it accounts for how much each input
feature actually contributes to the output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WandaConfig:
    """Configuration for Wanda pruning."""

    sparsity_ratio: float = 0.5
    """Fraction of weights to prune (set to zero).  Must be in [0, 1)."""

    semi_structured: bool = False
    """If True, apply N:M structured sparsity instead of global unstructured pruning."""

    n: int = 2
    """N in N:M sparsity — number of weights to *keep* per group of M."""

    m: int = 4
    """M in N:M sparsity — group size along the in_features dimension."""

    calibration_samples: int = 128
    """Expected number of calibration activation samples (informational only)."""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class WandaScorer:
    """Accumulates calibration activations and computes Wanda importance scores.

    Usage::

        scorer = WandaScorer(config)
        for X in calibration_loader:          # X: [B, T, in] or [B, in]
            scorer.accumulate(X)
        norms = scorer.get_activation_norms()  # [in_features]
        scores = scorer.score(W)               # [out_features, in_features]
    """

    def __init__(self, config: WandaConfig) -> None:
        self.config = config
        self._norm_sq: Tensor | None = None  # [in_features] running sum of ||x_j||^2
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def accumulate(self, X: Tensor) -> None:
        """Accumulate squared column norms from a batch of activations.

        Args:
            X: Activation tensor of shape ``[batch, in_features]`` or
               ``[batch, seq_len, in_features]``.  Any leading dimensions
               beyond the last are treated as the sample axis.
        """
        if X.dim() == 2:
            # [B, in_features] — each row is one sample
            flat = X  # [B, in_features]
        elif X.dim() == 3:
            # [B, T, in_features] — flatten batch and time
            B, T, F = X.shape
            flat = X.reshape(B * T, F)
        else:
            # Generic: collapse all but last dim
            flat = X.reshape(-1, X.shape[-1])

        # Sum of squared values per column: shape [in_features]
        sq_sum = (flat.detach().float() ** 2).sum(dim=0)

        if self._norm_sq is None:
            self._norm_sq = sq_sum
        else:
            self._norm_sq = self._norm_sq + sq_sum

        self._n_samples += flat.shape[0]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_activation_norms(self) -> Tensor:
        """Return per-column L2 norms ``||X_j||_2`` averaged over samples.

        Returns:
            Tensor of shape ``[in_features]`` with non-negative values.

        Raises:
            RuntimeError: If no activations have been accumulated yet.
        """
        if self._norm_sq is None or self._n_samples == 0:
            raise RuntimeError(
                "No activations accumulated. Call accumulate() with calibration data first."
            )
        # sqrt(sum_sq / n_samples) — element-wise L2 norm per column
        return torch.sqrt(self._norm_sq / self._n_samples)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._norm_sq = None
        self._n_samples = 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, W: Tensor) -> Tensor:
        """Compute Wanda importance scores for weight matrix W.

        score(W_ij) = |W_ij| * ||X_j||_2

        Args:
            W: Weight matrix of shape ``[out_features, in_features]``.

        Returns:
            Importance scores of the same shape as W.
        """
        norms = self.get_activation_norms()  # [in_features]
        # Broadcast: norms unsqueezed to [1, in_features]
        return W.abs().float() * norms.unsqueeze(0)


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------


class WandaPruner:
    """Applies Wanda importance-score-based pruning to a weight matrix.

    Example::

        config  = WandaConfig(sparsity_ratio=0.5)
        scorer  = WandaScorer(config)
        pruner  = WandaPruner(config)

        for X in calibration_data:
            scorer.accumulate(X)

        W_pruned = pruner.prune(W, scorer)
        print(pruner.sparsity(W_pruned))  # ≈ 0.5
    """

    def __init__(self, config: WandaConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Unstructured pruning
    # ------------------------------------------------------------------

    def prune_unstructured(self, W: Tensor, scores: Tensor) -> Tensor:
        """Zero out the bottom ``sparsity_ratio`` fraction of weights by score.

        Args:
            W:      Weight matrix ``[out_features, in_features]``.
            scores: Importance scores, same shape as W.

        Returns:
            Pruned weight matrix (same shape; low-score entries zeroed).
        """
        W = W.clone()
        n_prune = int(W.numel() * self.config.sparsity_ratio)
        if n_prune == 0:
            return W

        # Flat view for threshold computation
        flat_scores = scores.reshape(-1)
        # kth smallest = threshold; anything <= threshold gets pruned
        threshold = torch.kthvalue(flat_scores, n_prune).values
        mask = scores > threshold  # keep above threshold

        # Handle ties at the threshold: we need exactly n_prune zeros.
        # Mark positions that equal the threshold as candidates for pruning.
        tie_mask = scores == threshold
        n_already_pruned = (~mask).sum().item()
        n_extra_needed = n_prune - int(n_already_pruned)

        if n_extra_needed > 0 and tie_mask.any():
            # Pick n_extra_needed tie positions to prune (in flat order)
            tie_indices = tie_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            prune_ties = tie_indices[:n_extra_needed]
            mask_flat = mask.reshape(-1)
            mask_flat[prune_ties] = False
            mask = mask_flat.reshape(W.shape)

        W[~mask] = 0.0
        return W

    # ------------------------------------------------------------------
    # Semi-structured (N:M) pruning
    # ------------------------------------------------------------------

    def prune_semi_structured(self, W: Tensor, scores: Tensor) -> Tensor:
        """Apply N:M structured sparsity along the in_features dimension.

        In every group of M consecutive columns (in_features axis), retain the
        top N weights by Wanda score and zero the remaining M-N weights.

        Args:
            W:      Weight matrix ``[out_features, in_features]``.
            scores: Importance scores, same shape as W.

        Returns:
            Pruned weight matrix with exactly (M-N)/M sparsity per group.
        """
        n, m = self.config.n, self.config.m
        out_features, in_features = W.shape

        if in_features % m != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by M ({m}) "
                "for N:M semi-structured pruning."
            )

        W = W.clone()
        # Reshape to [out_features, in_features // m, m]
        W_grouped = W.reshape(out_features, in_features // m, m)
        scores_grouped = scores.reshape(out_features, in_features // m, m)

        # Within each group of M values, find the top-N indices
        # topk returns values + indices; indices are along the last dim
        _, top_indices = torch.topk(scores_grouped, k=n, dim=-1)

        # Build a keep mask: False everywhere, True at top-N positions
        keep_mask = torch.zeros_like(W_grouped, dtype=torch.bool)
        keep_mask.scatter_(-1, top_indices, True)

        W_grouped[~keep_mask] = 0.0
        return W_grouped.reshape(out_features, in_features)

    # ------------------------------------------------------------------
    # Unified prune entry-point
    # ------------------------------------------------------------------

    def prune(self, W: Tensor, scorer: WandaScorer) -> Tensor:
        """Score W using scorer then apply the configured pruning strategy.

        Args:
            W:      Weight matrix ``[out_features, in_features]``.
            scorer: A WandaScorer with accumulated calibration statistics.

        Returns:
            Pruned weight matrix (same dtype and shape as W).
        """
        original_dtype = W.dtype
        scores = scorer.score(W)  # float32

        if self.config.semi_structured:
            W_pruned = self.prune_semi_structured(W.float(), scores)
        else:
            W_pruned = self.prune_unstructured(W.float(), scores)

        return W_pruned.to(original_dtype)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sparsity(self, W: Tensor) -> float:
        """Return the fraction of zero weights in W.

        Args:
            W: Weight tensor of any shape.

        Returns:
            Float in [0, 1] representing the fraction of zero elements.
        """
        if W.numel() == 0:
            return 0.0
        return float((W == 0).sum().item()) / W.numel()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.model import MODEL_COMPONENT_REGISTRY  # noqa: E402

MODEL_COMPONENT_REGISTRY["wanda_pruner"] = WandaPruner
