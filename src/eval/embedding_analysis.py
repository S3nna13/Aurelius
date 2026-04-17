"""Embedding analysis utilities for AureliusTransformer.

Provides isotropy measurement, intrinsic dimensionality estimation,
k-means clustering, and nearest-neighbor retrieval — pure PyTorch only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingAnalysisConfig:
    """Configuration for embedding analysis routines."""

    n_components: int = 50      # PCA / dimensionality components
    n_neighbors: int = 10       # neighbours for ID estimation / retrieval
    normalize: bool = True      # L2-normalise before analysis when True


# ---------------------------------------------------------------------------
# Isotropy
# ---------------------------------------------------------------------------

class IsotropyAnalyzer:
    """Measures how uniformly spread embeddings are in vector space."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _covariance(embeddings: Tensor) -> Tensor:
        """Zero-mean covariance matrix of shape (d, d)."""
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)
        n = x.shape[0]
        return (x.T @ x) / max(n - 1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_isotropy(self, embeddings: Tensor) -> float:
        """Isotropy = min_eigenvalue / max_eigenvalue of covariance.

        Returns a value in [0, 1]:
        * 1  — perfectly isotropic (all eigenvalues equal)
        * 0  — completely collapsed (degenerate covariance)
        """
        cov = self._covariance(embeddings)
        # eigh returns ascending eigenvalues for symmetric matrices
        eigenvalues = torch.linalg.eigh(cov).eigenvalues
        # Clamp to avoid numerical negatives close to zero
        eigenvalues = eigenvalues.clamp(min=0.0)
        max_ev = eigenvalues.max().item()
        if max_ev == 0.0:
            return 0.0
        min_ev = eigenvalues.min().item()
        return float(min_ev / max_ev)

    def average_cosine_similarity(self, embeddings: Tensor) -> float:
        """Mean pairwise cosine similarity.

        Samples up to 1000 pairs when N > 100 to keep computation tractable.
        """
        x = embeddings.float()
        n = x.shape[0]

        # Normalise rows
        norms = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = x / norms

        if n > 100:
            # Sample 1000 random pairs
            rng = torch.Generator(device=x.device)
            idx_a = torch.randint(0, n, (1000,), generator=rng, device=x.device)
            idx_b = torch.randint(0, n, (1000,), generator=rng, device=x.device)
            sims = (x_norm[idx_a] * x_norm[idx_b]).sum(dim=1)
        else:
            # Full pairwise (N x N), exclude diagonal
            sim_matrix = x_norm @ x_norm.T
            mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
            sims = sim_matrix[mask]

        return float(sims.mean().item())

    def spectral_entropy(self, embeddings: Tensor) -> float:
        """Shannon entropy of the normalised eigenvalue spectrum.

        H = -sum(p_i * log(p_i + eps))  where p_i are probabilities
        derived from the eigenvalues of the covariance matrix.
        """
        cov = self._covariance(embeddings)
        eigenvalues = torch.linalg.eigh(cov).eigenvalues
        eigenvalues = eigenvalues.clamp(min=0.0)
        total = eigenvalues.sum()
        if total == 0.0:
            return 0.0
        p = eigenvalues / total
        entropy = -(p * torch.log(p + 1e-10)).sum()
        return float(entropy.item())


# ---------------------------------------------------------------------------
# Intrinsic Dimensionality
# ---------------------------------------------------------------------------

class IntrinsicDimensionEstimator:
    """Estimates the intrinsic dimensionality of an embedding space."""

    def __init__(self, n_neighbors: int = 10) -> None:
        self.n_neighbors = n_neighbors

    def twonn_estimate(self, embeddings: Tensor) -> float:
        """TwoNN intrinsic dimensionality estimator.

        For each point finds its two nearest neighbours, computes the ratio
        mu = d2 / d1 of the distances, then estimates ID as::

            ID = 1 / (mean(log(mu)) + eps)
        """
        x = embeddings.float()
        n = x.shape[0]

        # Pairwise squared Euclidean distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq_norms = (x * x).sum(dim=1)
        dist2 = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (x @ x.T)
        dist2 = dist2.clamp(min=0.0)

        # Set diagonal to large value so self is never selected
        dist2.fill_diagonal_(float("inf"))

        # Sort and take the two smallest distances per row
        sorted_dists, _ = dist2.sort(dim=1)
        d1 = sorted_dists[:, 0].clamp(min=1e-12).sqrt()
        d2 = sorted_dists[:, 1].clamp(min=1e-12).sqrt()

        mu = d2 / d1.clamp(min=1e-12)
        # Exclude points where mu == 1 (degenerate)
        mu = mu.clamp(min=1.0 + 1e-8)
        log_mu = torch.log(mu).mean()
        id_estimate = 1.0 / (log_mu.item() + 1e-10)
        return float(id_estimate)

    def pca_explained_variance_ratio(self, embeddings: Tensor) -> Tensor:
        """Cumulative explained variance ratio via SVD.

        Returns a 1-D tensor of length min(N, d) where the last value
        is approximately 1.0.
        """
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)
        # Economy SVD
        _, s, _ = torch.linalg.svd(x, full_matrices=False)
        var = s ** 2
        total_var = var.sum()
        if total_var == 0.0:
            return torch.zeros(var.shape[0])
        ratio = var / total_var
        cumulative = torch.cumsum(ratio, dim=0)
        return cumulative


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class EmbeddingClustering:
    """Simple Lloyd's k-means clustering for embedding tensors."""

    def __init__(self, n_clusters: int) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters

    def kmeans(
        self,
        embeddings: Tensor,
        n_iter: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """Lloyd's k-means.

        Returns
        -------
        cluster_ids : LongTensor of shape (N,)
        centroids   : FloatTensor of shape (n_clusters, d)
        """
        x = embeddings.float()
        n, d = x.shape
        k = min(self.n_clusters, n)

        # Initialise centroids by picking k random points (no replacement)
        perm = torch.randperm(n, device=x.device)[:k]
        centroids = x[perm].clone()

        cluster_ids = torch.zeros(n, dtype=torch.long, device=x.device)

        for _ in range(n_iter):
            # Assignment step: nearest centroid (Euclidean)
            # dist(x_i, c_j)^2 = ||x_i||^2 + ||c_j||^2 - 2 x_i·c_j
            sq_x = (x * x).sum(dim=1, keepdim=True)          # (N,1)
            sq_c = (centroids * centroids).sum(dim=1)          # (k,)
            cross = x @ centroids.T                            # (N,k)
            dists2 = sq_x + sq_c.unsqueeze(0) - 2.0 * cross  # (N,k)
            cluster_ids = dists2.argmin(dim=1)                 # (N,)

            # Update step: recompute centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, dtype=x.dtype, device=x.device)
            new_centroids.scatter_add_(
                0,
                cluster_ids.unsqueeze(1).expand(-1, d),
                x,
            )
            counts.scatter_add_(
                0,
                cluster_ids,
                torch.ones(n, dtype=x.dtype, device=x.device),
            )
            # Avoid division by zero for empty clusters (keep old centroid)
            mask = counts > 0
            new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
            new_centroids[~mask] = centroids[~mask]
            centroids = new_centroids

        return cluster_ids, centroids

    def silhouette_score(
        self,
        embeddings: Tensor,
        cluster_ids: Tensor,
    ) -> float:
        """Simplified silhouette score.

        For each sample: s = (b - a) / max(a, b)
        where a = mean intra-cluster distance and
              b = min mean inter-cluster distance.

        Returns mean over all samples in [-1, 1].
        """
        x = embeddings.float()
        ids = cluster_ids.long()
        n = x.shape[0]
        unique_clusters = ids.unique()
        k = unique_clusters.shape[0]

        if k == 1:
            # All points in one cluster — silhouette undefined, return 0
            return 0.0

        scores = torch.zeros(n, dtype=x.dtype, device=x.device)

        for i in range(n):
            xi = x[i]
            ci = ids[i].item()

            # Intra-cluster distance (a)
            intra_mask = ids == ci
            intra_mask[i] = False  # exclude self
            if intra_mask.sum() == 0:
                a = 0.0
            else:
                a = float((xi - x[intra_mask]).norm(dim=1).mean().item())

            # Inter-cluster distances (b) — min mean distance to other clusters
            b = float("inf")
            for cj in unique_clusters:
                cj = cj.item()
                if cj == ci:
                    continue
                inter_mask = ids == cj
                mean_dist = float((xi - x[inter_mask]).norm(dim=1).mean().item())
                if mean_dist < b:
                    b = mean_dist

            denom = max(a, b)
            if denom == 0.0:
                scores[i] = 0.0
            else:
                scores[i] = (b - a) / denom

        return float(scores.mean().item())


# ---------------------------------------------------------------------------
# Nearest-Neighbour Retrieval
# ---------------------------------------------------------------------------

class NearestNeighborRetriever:
    """Cosine-similarity based nearest-neighbour retrieval."""

    def __init__(self, index: Tensor) -> None:
        """
        Parameters
        ----------
        index : Tensor of shape (M, d)
            Reference embeddings to search against.
        """
        idx = index.float()
        norms = idx.norm(dim=1, keepdim=True).clamp(min=1e-12)
        self._index_norm = idx / norms          # (M, d)
        self._index_raw = idx

    def search(
        self,
        queries: Tensor,
        k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """Return top-k indices and cosine similarity scores.

        Returns
        -------
        indices : LongTensor  (Q, k)
        scores  : FloatTensor (Q, k)  — cosine similarities in [-1, 1]
        """
        q = queries.float()
        q_norms = q.norm(dim=1, keepdim=True).clamp(min=1e-12)
        q_norm = q / q_norms  # (Q, d)

        sim = q_norm @ self._index_norm.T  # (Q, M)
        k_clamped = min(k, sim.shape[1])
        scores, indices = sim.topk(k_clamped, dim=1, largest=True, sorted=True)
        return indices, scores

    def recall_at_k(
        self,
        queries: Tensor,
        ground_truth: Tensor,
        k: int = 5,
    ) -> float:
        """Fraction of queries where the true index appears in top-k results.

        Parameters
        ----------
        queries      : Tensor (Q, d)
        ground_truth : LongTensor (Q,) — true indices into the index
        k            : int

        Returns float in [0, 1].
        """
        indices, _ = self.search(queries, k=k)  # (Q, k)
        gt = ground_truth.long().unsqueeze(1)    # (Q, 1)
        hits = (indices == gt).any(dim=1)        # (Q,)
        return float(hits.float().mean().item())
