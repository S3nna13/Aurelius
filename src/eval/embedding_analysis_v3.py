"""Embedding space analysis tools — v3 — for AureliusTransformer.

Implements:
  * IsotropyMetrics           — partition-function, cosine-sim, partition score
  * IntrinsicDimensionality   — TwoNN, PCA component count, participation ratio
  * AnisotropyCorrector       — ZCA-whitening corrector with round-trip support
  * EmbeddingClusterAnalysis  — Lloyd's k-means, silhouette, intra-cluster variance
  * EmbeddingAnalysisSuite    — one-stop wrapper with full_report()
  * EmbeddingAnalysisConfig   — dataclass holding default hyper-parameters

Pure PyTorch only — no third-party packages beyond torch / stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingAnalysisConfig:
    """Hyper-parameters shared across all analysis routines."""

    n_dirs: int = 50
    threshold: float = 0.95
    n_clusters: int = 4
    n_iters: int = 20


# ---------------------------------------------------------------------------
# Isotropy Metrics
# ---------------------------------------------------------------------------


class IsotropyMetrics:
    """Three complementary measures of embedding-space isotropy.

    Methods
    -------
    compute_isotropy          — Arora-style partition-function ratio I(E) ∈ [0, 1]
    compute_average_cosine_similarity — mean pairwise cosine sim ∈ [-1, 1]
    compute_partition_score   — Mu & Viswanath 2018 variant ∈ [0, 1]
    """

    # ------------------------------------------------------------------
    # compute_isotropy
    # ------------------------------------------------------------------

    def compute_isotropy(
        self,
        embeddings: Tensor,
        n_dirs: int = 100,
        seed: int = 0,
    ) -> float:
        """Partition-function isotropy.

        I(E) = min_c Z(c) / max_c Z(c)
        where Z(c) = sum_i exp(c · e_i) and c ranges over *n_dirs*
        random unit vectors.

        Returns a value in [0, 1]; 1 = perfectly isotropic.
        """
        x = embeddings.float()
        n, d = x.shape

        # L2-normalise embeddings so the partition function is scale-invariant
        # and the metric reflects directional spread only (as in Arora et al.)
        x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        # Sample n_dirs random unit vectors: shape (n_dirs, d)
        c = torch.randn(n_dirs, d, generator=gen, device=x.device)
        c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # Raw dot products: (n_dirs, n)
        scores = c @ x.T

        # Numerically stable log-sum-exp per direction
        s_max = scores.max(dim=1, keepdim=True).values  # (n_dirs, 1)
        z = (scores - s_max).exp().sum(dim=1)  # (n_dirs,)

        z_min = z.min().item()
        z_max = z.max().item()
        if z_max == 0.0:
            return 0.0
        return float(z_min / z_max)

    # ------------------------------------------------------------------
    # compute_average_cosine_similarity
    # ------------------------------------------------------------------

    def compute_average_cosine_similarity(
        self,
        embeddings: Tensor,
        n_pairs: int = 1000,
        seed: int = 0,
    ) -> float:
        """Mean cosine similarity over random (or all) pairs.

        Returns a value in [-1, 1]. For purely random unit vectors in
        high-dimensional space the expected value is ≈ 0.
        """
        x = embeddings.float()
        n = x.shape[0]

        if n <= 1:
            return 0.0

        norms = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        x_n = x / norms

        # Full pairwise for small N; otherwise sample pairs
        if n * (n - 1) <= n_pairs * 2:
            sim = x_n @ x_n.T  # (n, n)
            mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
            return float(sim[mask].mean().item())

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        idx_a = torch.randint(0, n, (n_pairs,), generator=gen, device=x.device)
        idx_b = torch.randint(0, n, (n_pairs,), generator=gen, device=x.device)
        sims = (x_n[idx_a] * x_n[idx_b]).sum(dim=1)
        return float(sims.mean().item())

    # ------------------------------------------------------------------
    # compute_partition_score
    # ------------------------------------------------------------------

    def compute_partition_score(
        self,
        embeddings: Tensor,
        n_dirs: int = 100,
        seed: int = 0,
    ) -> float:
        """Partition-based isotropy (Mu & Viswanath 2018).

        For each random direction c, the fraction of points with a
        positive projection should be 0.5 if embeddings are isotropic.

        score = 1 − 2 * mean_dirs |frac_pos − 0.5|  ∈ [0, 1]
        """
        x = embeddings.float()
        n, d = x.shape

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        c = torch.randn(n_dirs, d, generator=gen, device=x.device)
        c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-12)

        proj = c @ x.T  # (n_dirs, n)
        frac_pos = (proj > 0).float().mean(dim=1)  # (n_dirs,)
        deviation = (frac_pos - 0.5).abs().mean().item()
        return float(1.0 - 2.0 * deviation)


# ---------------------------------------------------------------------------
# Intrinsic Dimensionality
# ---------------------------------------------------------------------------


class IntrinsicDimensionality:
    """Three estimators of the intrinsic dimensionality of an embedding space.

    Methods
    -------
    twonn_estimate         — Two-NN estimator (Facco et al. 2017)
    pca_explained_variance — smallest k s.t. cumvar ≥ threshold
    participation_ratio    — (Σλ)² / Σλ²  ∈ [1, d]
    """

    # ------------------------------------------------------------------
    # twonn_estimate
    # ------------------------------------------------------------------

    def twonn_estimate(self, embeddings: Tensor) -> float:
        """Two-Nearest-Neighbours intrinsic dimensionality estimator.

        ID = 1 / mean( log(r2 / r1) )

        where r1, r2 are distances to the 1st and 2nd nearest neighbours.
        Returns a positive float.
        """
        x = embeddings.float()
        n, d = x.shape

        # Pairwise squared Euclidean distances
        sq = (x * x).sum(dim=1)  # (n,)
        dist2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (x @ x.T)).clamp(min=0.0)  # (n, n)

        # Exclude self-distances
        dist2.fill_diagonal_(float("inf"))

        # Two smallest per row
        k = min(2, n - 1)
        smallest = dist2.topk(k, dim=1, largest=False).values  # (n, k)

        r1 = smallest[:, 0].clamp(min=1e-12).sqrt()
        r2 = smallest[:, min(1, k - 1)].clamp(min=1e-12).sqrt()

        ratio = (r2 / r1.clamp(min=1e-12)).clamp(min=1.0 + 1e-8)
        log_ratio = torch.log(ratio)
        mean_log = log_ratio.mean().item()
        if mean_log <= 0.0:
            return float(d)
        return float(1.0 / mean_log)

    # ------------------------------------------------------------------
    # pca_explained_variance
    # ------------------------------------------------------------------

    def pca_explained_variance(
        self,
        embeddings: Tensor,
        threshold: float = 0.95,
    ) -> int:
        """Smallest number of PCA components explaining *threshold* of variance.

        Returns an integer in [1, d].
        """
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)

        _, s, _ = torch.linalg.svd(x, full_matrices=False)  # s: (min(n,d),)
        var = s**2
        total = var.sum()
        if total == 0.0:
            return 1

        cumvar = torch.cumsum(var / total, dim=0)
        hits = (cumvar >= threshold).nonzero(as_tuple=False)
        if hits.numel() == 0:
            return int(cumvar.shape[0])
        return int(hits[0, 0].item()) + 1  # 1-indexed count

    # ------------------------------------------------------------------
    # participation_ratio
    # ------------------------------------------------------------------

    def participation_ratio(self, embeddings: Tensor) -> float:
        """Participation ratio: PR = (Σλ)² / Σλ².

        PR ∈ [1, d]; measures effective number of active dimensions.
        """
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)
        n = x.shape[0]

        cov = (x.T @ x) / max(n - 1, 1)  # (d, d)
        lam = torch.linalg.eigh(cov).eigenvalues.clamp(min=0.0)  # (d,)

        sum_lam = lam.sum().item()
        sum_lam2 = (lam**2).sum().item()
        if sum_lam2 == 0.0:
            return 1.0
        return float((sum_lam**2) / sum_lam2)


# ---------------------------------------------------------------------------
# Anisotropy Corrector
# ---------------------------------------------------------------------------


class AnisotropyCorrector:
    """ZCA-whitening corrector for anisotropic embeddings.

    Usage
    -----
    corrector = AnisotropyCorrector()
    corrected = corrector.fit_transform(raw_embeddings)  # [N, d] → [N, d]
    original  = corrector.inverse_transform(corrected)   # round-trips
    """

    def __init__(self) -> None:
        self._mean: Tensor | None = None
        self._W: Tensor | None = None  # whitening matrix  (d, d)
        self._W_inv: Tensor | None = None  # de-whitening matrix (d, d)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, embeddings: Tensor) -> AnisotropyCorrector:
        """Estimate mean and ZCA-whitening transform from *embeddings*.

        Stores:
          _mean  — per-dimension mean  (d,)
          _W     — whitening matrix    (d, d)
          _W_inv — de-whitening matrix (d, d)
        """
        x = embeddings.float()
        n, d = x.shape
        self._mean = x.mean(dim=0)  # (d,)

        x_c = x - self._mean.unsqueeze(0)  # centred

        # Economy SVD of the centred data: X_c = U S Vt
        _U, s, Vt = torch.linalg.svd(x_c, full_matrices=False)
        V = Vt.T  # (d, k)

        # Covariance eigenvalues: λ = s² / (n-1)
        lam = (s**2) / max(n - 1, 1)  # (k,)
        eps = 1e-8

        inv_std = 1.0 / (lam.sqrt() + eps)  # (k,)
        std = lam.sqrt() + eps  # (k,)

        # ZCA: W = V diag(1/√λ) Vᵀ
        self._W = V @ torch.diag(inv_std) @ V.T  # (d, d)
        # Inverse: W⁻¹ = V diag(√λ) Vᵀ
        self._W_inv = V @ torch.diag(std) @ V.T  # (d, d)
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, embeddings: Tensor) -> Tensor:
        """Apply mean-centering + whitening. Shape is preserved."""
        if self._mean is None or self._W is None:
            raise RuntimeError("AnisotropyCorrector.fit() must be called before transform().")
        x = embeddings.float()
        x_c = x - self._mean.to(x.device).unsqueeze(0)
        return x_c @ self._W.to(x.device).T

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(self, embeddings: Tensor) -> Tensor:
        """Fit on *embeddings* and immediately transform them."""
        return self.fit(embeddings).transform(embeddings)

    # ------------------------------------------------------------------
    # inverse_transform
    # ------------------------------------------------------------------

    def inverse_transform(self, embeddings: Tensor) -> Tensor:
        """Invert the whitening transform, recovering the original scale."""
        if self._mean is None or self._W_inv is None:
            raise RuntimeError(
                "AnisotropyCorrector.fit() must be called before inverse_transform()."
            )
        x = embeddings.float()
        x_unwhitened = x @ self._W_inv.to(x.device).T
        return x_unwhitened + self._mean.to(x.device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Embedding Cluster Analysis
# ---------------------------------------------------------------------------


class EmbeddingClusterAnalysis:
    """K-means clustering and quality metrics for embedding tensors.

    Parameters
    ----------
    n_clusters : int
        Number of clusters k ≥ 1.
    """

    def __init__(self, n_clusters: int) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self._centroids: Tensor | None = None

    # ------------------------------------------------------------------
    # kmeans_fit
    # ------------------------------------------------------------------

    def kmeans_fit(
        self,
        embeddings: Tensor,
        n_iters: int = 50,
        seed: int = 0,
    ) -> Tensor:
        """Lloyd's k-means algorithm in pure PyTorch.

        Returns
        -------
        labels : LongTensor of shape (N,) with values in [0, n_clusters)
        """
        x = embeddings.float()
        n, d = x.shape
        k = min(self.n_clusters, n)

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        # Initialise centroids by picking k random distinct points
        perm = torch.randperm(n, generator=gen, device=x.device)[:k]
        centroids = x[perm].clone()  # (k, d)

        labels = torch.zeros(n, dtype=torch.long, device=x.device)

        for _ in range(n_iters):
            # --- Assignment step (squared Euclidean distance) ---
            # ||x_i - c_j||² = ||x_i||² + ||c_j||² - 2 x_i·c_j
            sq_x = (x * x).sum(dim=1, keepdim=True)  # (n, 1)
            sq_c = (centroids * centroids).sum(dim=1)  # (k,)
            cross = x @ centroids.T  # (n, k)
            dist2 = (sq_x + sq_c.unsqueeze(0) - 2.0 * cross).clamp(min=0.0)
            labels = dist2.argmin(dim=1)  # (n,)

            # --- Update step ---
            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(k, dtype=x.dtype, device=x.device)
            new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), x)
            counts.scatter_add_(0, labels, torch.ones(n, dtype=x.dtype, device=x.device))
            non_empty = counts > 0
            new_c[non_empty] = new_c[non_empty] / counts[non_empty].unsqueeze(1)
            new_c[~non_empty] = centroids[~non_empty]  # keep stale centroid
            centroids = new_c

        self._centroids = centroids
        return labels

    # ------------------------------------------------------------------
    # silhouette_score
    # ------------------------------------------------------------------

    def silhouette_score(self, embeddings: Tensor, labels: Tensor) -> float:
        """Mean silhouette coefficient.

        s_i = (b_i − a_i) / max(a_i, b_i)

        where a_i = mean intra-cluster distance and b_i = min mean
        distance to any other cluster.

        Returns a float in [-1, 1].
        """
        x = embeddings.float()
        ids = labels.long()
        n = x.shape[0]
        unique = ids.unique()

        if unique.shape[0] == 1:
            # Single cluster — silhouette is undefined; return 0
            return 0.0

        scores = torch.zeros(n, dtype=x.dtype, device=x.device)

        for i in range(n):
            xi = x[i]
            ci = ids[i].item()

            # Intra-cluster mean distance (a)
            intra_mask = (ids == ci).clone()
            intra_mask[i] = False
            if intra_mask.sum() == 0:
                a = 0.0
            else:
                a = float((xi - x[intra_mask]).norm(dim=1).mean().item())

            # Min inter-cluster mean distance (b)
            b = float("inf")
            for cj_t in unique:
                cj = cj_t.item()
                if cj == ci:
                    continue
                inter_mask = ids == cj
                if inter_mask.sum() == 0:
                    continue
                mean_dist = float((xi - x[inter_mask]).norm(dim=1).mean().item())
                if mean_dist < b:
                    b = mean_dist

            if b == float("inf"):
                b = 0.0

            denom = max(a, b)
            scores[i] = 0.0 if denom == 0.0 else float((b - a) / denom)

        return float(scores.mean().item())

    # ------------------------------------------------------------------
    # intra_cluster_variance
    # ------------------------------------------------------------------

    def intra_cluster_variance(self, embeddings: Tensor, labels: Tensor) -> float:
        """Mean within-cluster variance (mean squared distance to centroid).

        Returns a non-negative float.
        """
        x = embeddings.float()
        ids = labels.long()
        unique = ids.unique()

        total = 0.0
        for ct in unique:
            c = ct.item()
            mask = ids == c
            members = x[mask]
            if members.shape[0] == 0:
                continue
            centroid = members.mean(dim=0)
            var = ((members - centroid.unsqueeze(0)) ** 2).sum(dim=1).mean().item()
            total += var

        return float(total / max(len(unique), 1))


# ---------------------------------------------------------------------------
# Embedding Analysis Suite
# ---------------------------------------------------------------------------


class EmbeddingAnalysisSuite:
    """Unified wrapper that runs all embedding analyses in one call.

    Attributes
    ----------
    isotropy      : IsotropyMetrics
    dimensionality: IntrinsicDimensionality
    corrector     : AnisotropyCorrector
    clustering    : EmbeddingClusterAnalysis
    config        : EmbeddingAnalysisConfig
    """

    def __init__(self, config: EmbeddingAnalysisConfig | None = None) -> None:
        self.config = config or EmbeddingAnalysisConfig()
        self.isotropy = IsotropyMetrics()
        self.dimensionality = IntrinsicDimensionality()
        self.corrector = AnisotropyCorrector()
        self.clustering = EmbeddingClusterAnalysis(n_clusters=self.config.n_clusters)

    # ------------------------------------------------------------------
    # full_report
    # ------------------------------------------------------------------

    def full_report(self, embeddings: Tensor) -> dict[str, float]:
        """Compute every metric and return a flat dict of floats.

        Keys
        ----
        isotropy_score, avg_cosine_similarity, partition_score,
        twonn_id, pca_dims, participation_ratio,
        silhouette_score, intra_cluster_variance
        """
        cfg = self.config

        iso = self.isotropy.compute_isotropy(embeddings, n_dirs=cfg.n_dirs)
        avg_cos = self.isotropy.compute_average_cosine_similarity(embeddings)
        part = self.isotropy.compute_partition_score(embeddings, n_dirs=cfg.n_dirs)

        twonn = self.dimensionality.twonn_estimate(embeddings)
        pca_k = float(
            self.dimensionality.pca_explained_variance(embeddings, threshold=cfg.threshold)
        )
        pr = self.dimensionality.participation_ratio(embeddings)

        labels = self.clustering.kmeans_fit(embeddings, n_iters=cfg.n_iters)
        sil = self.clustering.silhouette_score(embeddings, labels)
        icv = self.clustering.intra_cluster_variance(embeddings, labels)

        return {
            "isotropy_score": iso,
            "avg_cosine_similarity": avg_cos,
            "partition_score": part,
            "twonn_id": twonn,
            "pca_dims": pca_k,
            "participation_ratio": pr,
            "silhouette_score": sil,
            "intra_cluster_variance": icv,
        }
