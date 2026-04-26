"""Out-of-Distribution (OOD) detection for LLMs.

Implements four scoring methods operating on feature representations (hidden
states) or logits:

  - MaxSoftmaxScorer  — Hendrycks & Gimpel 2017 (MSP)
  - EnergyScorer      — Liu et al. 2020
  - MahalanobisScorer — Lee et al. 2018
  - KNNScorer         — Sun et al. 2022

All scorers return higher scores for *more in-distribution* samples so that a
single threshold comparison works uniformly: score < threshold → OOD.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class OODScorer(ABC):
    """Abstract base class for OOD scorers.

    Convention: ``score`` returns a 1-D FloatTensor of shape ``(B,)`` where
    *higher* values indicate the sample is more likely *in-distribution*.
    ``fit`` is a no-op for scorers that require no training data; it must
    still accept ``(features, labels)`` for a uniform API.
    """

    def fit(self, features: Tensor, labels: Tensor | None = None) -> OODScorer:
        """Fit scorer to in-distribution data (no-op for non-parametric scorers)."""
        return self

    @abstractmethod
    def score(self, features: Tensor) -> Tensor:
        """Return (B,) float tensor; higher = more in-distribution."""


# ---------------------------------------------------------------------------
# 1. MaxSoftmax (MSP)
# ---------------------------------------------------------------------------


class MaxSoftmaxScorer(OODScorer):
    """Maximum softmax probability scorer (Hendrycks & Gimpel, 2017).

    Args:
        None

    Input to ``score``: logits of shape ``(B, C)``.
    Output: ``(B,)`` — max softmax probability in [0, 1].
    """

    def score(self, features: Tensor) -> Tensor:  # features = logits here
        """Compute max softmax probability for each sample.

        Args:
            features: ``(B, C)`` logit tensor.

        Returns:
            ``(B,)`` float tensor with values in (0, 1].
        """
        if features.dim() != 2:
            raise ValueError(
                f"MaxSoftmaxScorer.score expects (B, C) logits; got shape {tuple(features.shape)}"
            )
        probs = F.softmax(features, dim=-1)
        return probs.max(dim=-1).values


# ---------------------------------------------------------------------------
# 2. Energy Score
# ---------------------------------------------------------------------------


class EnergyScorer(OODScorer):
    """Energy-based OOD scorer (Liu et al., 2020).

    score(logits) = -log Σ_c exp(logits_c / T)

    Negated so that higher score → more in-distribution (lower energy).

    Args:
        temperature: Softmax temperature T (default 1.0).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def score(self, features: Tensor) -> Tensor:
        """Compute negative energy for each sample.

        Args:
            features: ``(B, C)`` logit tensor.

        Returns:
            ``(B,)`` float tensor (more negative = higher energy = more OOD).
        """
        if features.dim() != 2:
            raise ValueError(
                f"EnergyScorer.score expects (B, C) logits; got shape {tuple(features.shape)}"
            )
        # log-sum-exp over class dimension, scaled by temperature
        energy = torch.logsumexp(features / self.temperature, dim=-1)
        # negate: in-dist samples have lower energy → higher score
        return -energy


# ---------------------------------------------------------------------------
# 3. Mahalanobis Distance
# ---------------------------------------------------------------------------


class MahalanobisScorer(OODScorer):
    """Mahalanobis distance OOD scorer (Lee et al., 2018).

    Fit: compute per-class means and a shared tied covariance from
         in-distribution feature vectors.
    Score: negative minimum Mahalanobis distance to any class centroid.

    Args:
        reg: Regularisation added to diagonal of covariance before inversion
             (default 1e-5) to prevent singular matrices.
    """

    def __init__(self, reg: float = 1e-5) -> None:
        if reg < 0:
            raise ValueError(f"reg must be >= 0, got {reg}")
        self.reg = reg
        self._means: Tensor | None = None  # (C, D)
        self._precision: Tensor | None = None  # (D, D)

    def fit(self, features: Tensor, labels: Tensor | None = None) -> MahalanobisScorer:
        """Fit per-class means and shared precision matrix.

        Args:
            features: ``(N, D)`` float tensor of in-distribution features.
            labels:   ``(N,)`` integer class labels. If ``None``, all samples
                      are treated as a single class.

        Returns:
            self
        """
        if features.dim() != 2:
            raise ValueError(
                f"MahalanobisScorer.fit expects (N, D) features; got {tuple(features.shape)}"
            )
        features = features.float()
        if labels is None:
            labels = torch.zeros(features.size(0), dtype=torch.long, device=features.device)

        classes = labels.unique()
        D = features.size(1)

        means = []
        centered_all = []
        for c in classes:
            mask = labels == c
            fc = features[mask]
            mu = fc.mean(dim=0)
            means.append(mu)
            centered_all.append(fc - mu)

        self._means = torch.stack(means, dim=0)  # (C, D)

        centered = torch.cat(centered_all, dim=0)  # (N, D)
        N = centered.size(0)
        cov = (centered.T @ centered) / max(N - 1, 1)  # (D, D)
        # Regularise diagonal
        cov = cov + self.reg * torch.eye(D, dtype=cov.dtype, device=cov.device)
        # Use pseudoinverse for robustness with near-singular matrices
        self._precision = torch.linalg.pinv(cov)  # (D, D)
        return self

    def score(self, features: Tensor) -> Tensor:
        """Compute negative minimum Mahalanobis distance (higher = more in-dist).

        Args:
            features: ``(B, D)`` float tensor.

        Returns:
            ``(B,)`` float tensor.

        Raises:
            RuntimeError: if ``fit`` has not been called.
        """
        if self._means is None or self._precision is None:
            raise RuntimeError("MahalanobisScorer must be fit before scoring.")
        if features.dim() != 2:
            raise ValueError(
                f"MahalanobisScorer.score expects (B, D) features; got {tuple(features.shape)}"
            )
        features = features.float()
        # diff: (B, C, D)
        diff = features.unsqueeze(1) - self._means.unsqueeze(0)
        # Mahalanobis: (B, C)
        # m[b,c] = diff[b,c] @ precision @ diff[b,c]
        Pd = diff @ self._precision  # (B, C, D)
        dist_sq = (Pd * diff).sum(dim=-1)  # (B, C)
        dist_sq = dist_sq.clamp(min=0.0)  # numerical safety
        min_dist = dist_sq.min(dim=-1).values  # (B,)
        return -min_dist


# ---------------------------------------------------------------------------
# 4. KNN Distance
# ---------------------------------------------------------------------------


class KNNScorer(OODScorer):
    """k-Nearest-Neighbour OOD scorer (Sun et al., 2022).

    Fit: store in-distribution feature vectors.
    Score: negative distance to the k-th nearest neighbour (cosine or L2).
          Higher score → smaller distance → more in-distribution.

    Args:
        k:      Number of neighbours (default 5).
        metric: ``'cosine'`` or ``'l2'`` (default ``'cosine'``).
    """

    def __init__(self, k: int = 5, metric: str = "cosine") -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got '{metric}'")
        self.k = k
        self.metric = metric
        self._train_features: Tensor | None = None

    def fit(self, features: Tensor, labels: Tensor | None = None) -> KNNScorer:
        """Store in-distribution features.

        Args:
            features: ``(N, D)`` float tensor.
            labels:   Ignored (kept for API uniformity).

        Returns:
            self
        """
        if features.dim() != 2:
            raise ValueError(f"KNNScorer.fit expects (N, D) features; got {tuple(features.shape)}")
        self._train_features = features.float().clone()
        return self

    def score(self, features: Tensor) -> Tensor:
        """Compute negative k-NN distance to training set.

        Args:
            features: ``(B, D)`` float tensor.

        Returns:
            ``(B,)`` float tensor (higher = more in-distribution).

        Raises:
            RuntimeError: if ``fit`` has not been called.
        """
        if self._train_features is None:
            raise RuntimeError("KNNScorer must be fit before scoring.")
        if features.dim() != 2:
            raise ValueError(
                f"KNNScorer.score expects (B, D) features; got {tuple(features.shape)}"
            )
        features = features.float()
        train = self._train_features

        if self.metric == "cosine":
            # Normalise both sets
            q = F.normalize(features, p=2, dim=-1)  # (B, D)
            r = F.normalize(train, p=2, dim=-1)  # (N, D)
            # cosine similarity: (B, N)
            sim = q @ r.T
            # distance = 1 - similarity
            dist = 1.0 - sim  # (B, N)
        else:  # l2
            # ||q - r||^2 = ||q||^2 + ||r||^2 - 2 q·r
            q_sq = (features**2).sum(dim=-1, keepdim=True)  # (B, 1)
            r_sq = (train**2).sum(dim=-1, keepdim=True).T  # (1, N)
            cross = features @ train.T  # (B, N)
            dist = (q_sq + r_sq - 2 * cross).clamp(min=0.0).sqrt()

        k_eff = min(self.k, dist.size(1))
        # kth smallest distance per query
        knn_dist, _ = torch.topk(dist, k_eff, dim=-1, largest=False)
        kth_dist = knn_dist[:, -1]  # (B,)
        return -kth_dist


# ---------------------------------------------------------------------------
# OODDetector — wrapper that applies a scorer + threshold
# ---------------------------------------------------------------------------


class OODDetector:
    """High-level OOD detector combining a scorer with a threshold.

    Args:
        scorer:    An ``OODScorer`` instance (already instantiated).
        threshold: Samples with ``score < threshold`` are flagged as OOD.
    """

    def __init__(self, scorer: OODScorer, threshold: float) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def fit(
        self,
        in_dist_features: Tensor,
        labels: Tensor | None = None,
    ) -> OODDetector:
        """Delegate to the underlying scorer's ``fit`` method.

        Args:
            in_dist_features: ``(N, D)`` or ``(N, C)`` feature/logit tensor.
            labels:           Optional ``(N,)`` class labels.

        Returns:
            self
        """
        self.scorer.fit(in_dist_features, labels)
        return self

    def score(self, features: Tensor) -> Tensor:
        """Return raw OOD scores from the underlying scorer.

        Args:
            features: ``(B, D)`` or ``(B, C)`` tensor.

        Returns:
            ``(B,)`` float tensor (higher = more in-distribution).
        """
        return self.scorer.score(features)

    def predict(self, features: Tensor) -> Tensor:
        """Predict OOD membership.

        Args:
            features: ``(B, D)`` or ``(B, C)`` tensor.

        Returns:
            ``(B,)`` BoolTensor — ``True`` means OOD.
        """
        scores = self.score(features)
        return scores < self.threshold
