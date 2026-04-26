"""Embedding space analysis tools for AureliusTransformer.

Implements isotropy measurement, intrinsic dimensionality estimation,
anisotropy correction, and clustering — pure PyTorch only.
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
    """Configuration for embedding analysis routines."""

    n_dirs: int = 50
    threshold: float = 0.95
    n_clusters: int = 4
    n_iters: int = 20


# ---------------------------------------------------------------------------
# Isotropy Metrics
# ---------------------------------------------------------------------------


class IsotropyMetrics:
    """Measures how uniformly spread embeddings are in vector space.

    Three complementary measures:
    * compute_isotropy       — partition-function ratio (Arora et al.)
    * compute_average_cosine_similarity — mean cosine sim of random pairs
    * compute_partition_score           — Mu & Viswanath 2018
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
        """Partition-function isotropy: I(E) = min_c Z(c) / max_c Z(c).

        Sample *n_dirs* random unit vectors c; for each compute
        Z(c) = sum_i exp(c · e_i).  Returns min/max ratio in [0, 1].
        """
        x = embeddings.float()
        n, d = x.shape

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        # Sample random unit vectors: (n_dirs, d)
        c = torch.randn(n_dirs, d, generator=gen, device=x.device)
        c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # Scores: (n_dirs, n)
        scores = c @ x.T  # (n_dirs, n)

        # log Z(c) = log sum_i exp(c·e_i)
        # Compare Z values in log-space so different directions are on the same scale.
        # Per-direction max subtraction makes z values incomparable across directions.
        log_z = torch.logsumexp(scores, dim=1)  # (n_dirs,)

        log_z_min = log_z.min()
        log_z_max = log_z.max()
        if log_z_min == log_z_max:
            return 1.0
        return float((log_z_min - log_z_max).exp().item())

    # ------------------------------------------------------------------
    # compute_average_cosine_similarity
    # ------------------------------------------------------------------

    def compute_average_cosine_similarity(
        self,
        embeddings: Tensor,
        n_pairs: int = 1000,
        seed: int = 0,
    ) -> float:
        """Mean cosine similarity over random pairs in [-1, 1]."""
        x = embeddings.float()
        n, d = x.shape

        norms = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = x / norms

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        if n <= 1:
            return 0.0

        if n * (n - 1) <= n_pairs * 2:
            # Full pairwise excluding diagonal
            sim = x_norm @ x_norm.T  # (n, n)
            mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
            return float(sim[mask].mean().item())

        # Sample random pairs (allow same index; statistically negligible)
        idx_a = torch.randint(0, n, (n_pairs,), generator=gen, device=x.device)
        idx_b = torch.randint(0, n, (n_pairs,), generator=gen, device=x.device)
        sims = (x_norm[idx_a] * x_norm[idx_b]).sum(dim=1)
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

        For each random direction c, compute the fraction of points whose
        projection onto c is positive. The partition score is the
        deviation of this fraction from 0.5, averaged across directions
        and mapped to [0, 1]:
            score = 1 - 2 * mean_dirs |frac_pos - 0.5|
        A perfectly isotropic set yields score ≈ 1.
        """
        x = embeddings.float()
        n, d = x.shape

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        c = torch.randn(n_dirs, d, generator=gen, device=x.device)
        c = c / c.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # Projections: (n_dirs, n)
        proj = c @ x.T

        # Fraction of positive projections per direction
        frac_pos = (proj > 0).float().mean(dim=1)  # (n_dirs,)
        deviation = (frac_pos - 0.5).abs().mean().item()
        return float(1.0 - 2.0 * deviation)


# ---------------------------------------------------------------------------
# Intrinsic Dimensionality
# ---------------------------------------------------------------------------


class IntrinsicDimensionality:
    """Estimates the intrinsic dimensionality of an embedding space."""

    # ------------------------------------------------------------------
    # twonn_estimate
    # ------------------------------------------------------------------

    def twonn_estimate(self, embeddings: Tensor) -> float:
        """Two-Nearest-Neighbours intrinsic dimensionality estimator.

        ID = 1 / mean(log(r2 / r1))

        where r1, r2 are the distances to the 1st and 2nd nearest
        neighbours of each point.
        """
        x = embeddings.float()
        n, d = x.shape

        # Pairwise squared Euclidean distances via the identity
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq_norms = (x * x).sum(dim=1)
        dist2 = (sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (x @ x.T)).clamp(min=0.0)

        # Exclude self: set diagonal to inf
        dist2.fill_diagonal_(float("inf"))

        # Two smallest distances per row
        k = min(2, n - 1)
        topk = dist2.topk(k, dim=1, largest=False).values  # (n, 2)

        r1 = topk[:, 0].clamp(min=1e-12).sqrt()
        r2 = topk[:, min(1, k - 1)].clamp(min=1e-12).sqrt()

        # Exclude degenerate points (r1 ≈ r2)
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
        """Number of PCA components needed to explain *threshold* of variance.

        Returns an integer in [1, d].
        """
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)

        _, s, _ = torch.linalg.svd(x, full_matrices=False)
        var = s**2
        total = var.sum()
        if total == 0.0:
            return 1

        cumvar = torch.cumsum(var / total, dim=0)
        # First index where cumulative variance >= threshold
        indices = (cumvar >= threshold).nonzero(as_tuple=False)
        if indices.numel() == 0:
            return int(cumvar.shape[0])
        return int(indices[0, 0].item()) + 1  # 1-indexed count

    # ------------------------------------------------------------------
    # participation_ratio
    # ------------------------------------------------------------------

    def participation_ratio(self, embeddings: Tensor) -> float:
        """Participation ratio: PR = (sum λ)² / sum(λ²).

        Returns a value in [1, d] where d = embedding dimension.
        Measures the effective number of dimensions used.
        """
        x = embeddings.float()
        x = x - x.mean(dim=0, keepdim=True)
        n = x.shape[0]

        # Covariance eigenvalues
        cov = (x.T @ x) / max(n - 1, 1)
        eigenvalues = torch.linalg.eigh(cov).eigenvalues.clamp(min=0.0)

        sum_lam = eigenvalues.sum().item()
        sum_lam2 = (eigenvalues**2).sum().item()
        if sum_lam2 == 0.0:
            return 1.0
        return float((sum_lam**2) / sum_lam2)


# ---------------------------------------------------------------------------
# Anisotropy Corrector
# ---------------------------------------------------------------------------


class AnisotropyCorrector:
    """Corrects anisotropy via mean-centering and ZCA-whitening.

    Fit computes the mean and principal directions via SVD.
    Transform projects embeddings into the whitened space.
    """

    def __init__(self) -> None:
        self._mean: Tensor | None = None
        self._W: Tensor | None = None  # whitening matrix (d, d)
        self._W_inv: Tensor | None = None  # inverse whitening matrix

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, embeddings: Tensor) -> AnisotropyCorrector:
        """Compute mean and whitening transform from *embeddings*.

        Uses ZCA whitening: W = U diag(1/sqrt(s+eps)) U^T
        where U, s come from the economy SVD of the centred data matrix.
        """
        x = embeddings.float()
        self._mean = x.mean(dim=0)  # (d,)

        x_c = x - self._mean.unsqueeze(0)
        n = x_c.shape[0]

        U, s, Vt = torch.linalg.svd(x_c, full_matrices=False)
        # Singular values → eigenvalues of covariance (scale by 1/n)
        # For whitening we need sqrt of eigenvalues of cov
        # cov eigenvalues: lam = s^2 / n
        # We whiten so each dimension has unit variance
        lam = (s**2) / max(n - 1, 1)
        inv_std = 1.0 / (lam.sqrt() + 1e-8)

        # ZCA whitening: W = V diag(1/sqrt(lam)) V^T
        # Vt rows are eigenvectors; Vt.T columns are eigenvectors
        V = Vt.T  # (d, k)
        self._W = V @ torch.diag(inv_std) @ V.T  # (d, d)
        self._W_inv = V @ torch.diag(lam.sqrt() + 1e-8) @ V.T  # (d, d)
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, embeddings: Tensor) -> Tensor:
        """Apply mean-centering and whitening.

        Returns a Tensor of the same shape as *embeddings*.
        """
        if self._mean is None or self._W is None:
            raise RuntimeError("AnisotropyCorrector must be fit before transform.")
        x = embeddings.float()
        x_c = x - self._mean.to(x.device).unsqueeze(0)
        return x_c @ self._W.to(x.device).T

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(self, embeddings: Tensor) -> Tensor:
        """Fit on *embeddings* then transform it in one call."""
        return self.fit(embeddings).transform(embeddings)

    # ------------------------------------------------------------------
    # inverse_transform
    # ------------------------------------------------------------------

    def inverse_transform(self, embeddings: Tensor) -> Tensor:
        """Invert the whitening transform.

        Returns a Tensor of the same shape as *embeddings*.
        """
        if self._mean is None or self._W_inv is None:
            raise RuntimeError("AnisotropyCorrector must be fit before inverse_transform.")
        x = embeddings.float()
        x_unwhitened = x @ self._W_inv.to(x.device).T
        return x_unwhitened + self._mean.to(x.device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Embedding Cluster Analysis
# ---------------------------------------------------------------------------


class EmbeddingClusterAnalysis:
    """K-means clustering and cluster quality metrics for embeddings."""

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
        """Lloyd's k-means algorithm.

        Returns
        -------
        labels : LongTensor of shape (N,)
        """
        x = embeddings.float()
        n, d = x.shape
        k = min(self.n_clusters, n)

        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)

        # Initialise centroids by picking k random points (no replacement)
        perm = torch.randperm(n, generator=gen, device=x.device)[:k]
        centroids = x[perm].clone()

        labels = torch.zeros(n, dtype=torch.long, device=x.device)

        for _ in range(n_iters):
            # Assignment: nearest centroid by squared Euclidean distance
            sq_x = (x * x).sum(dim=1, keepdim=True)  # (n, 1)
            sq_c = (centroids * centroids).sum(dim=1)  # (k,)
            cross = x @ centroids.T  # (n, k)
            dist2 = (sq_x + sq_c.unsqueeze(0) - 2.0 * cross).clamp(min=0.0)
            labels = dist2.argmin(dim=1)  # (n,)

            # Update: recompute centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(k, dtype=x.dtype, device=x.device)
            new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), x)
            counts.scatter_add_(0, labels, torch.ones(n, dtype=x.dtype, device=x.device))
            mask = counts > 0
            new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
            new_centroids[~mask] = centroids[~mask]
            centroids = new_centroids

        self._centroids = centroids
        return labels

    # ------------------------------------------------------------------
    # silhouette_score
    # ------------------------------------------------------------------

    def silhouette_score(self, embeddings: Tensor, labels: Tensor) -> float:
        """Mean silhouette coefficient: mean (b - a) / max(a, b).

        Returns a value in [-1, 1].
        """
        x = embeddings.float()
        ids = labels.long()
        n = x.shape[0]
        unique_clusters = ids.unique()

        if unique_clusters.shape[0] == 1:
            # All in one cluster — silhouette undefined
            return 0.0

        scores = torch.zeros(n, dtype=x.dtype, device=x.device)

        for i in range(n):
            xi = x[i]
            ci = ids[i].item()

            intra_mask = (ids == ci).clone()
            intra_mask[i] = False
            if intra_mask.sum() == 0:
                a = 0.0
            else:
                a = float((xi - x[intra_mask]).norm(dim=1).mean().item())

            b = float("inf")
            for cj_t in unique_clusters:
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
            if denom == 0.0:
                scores[i] = 0.0
            else:
                scores[i] = (b - a) / denom

        return float(scores.mean().item())

    # ------------------------------------------------------------------
    # intra_cluster_variance
    # ------------------------------------------------------------------

    def intra_cluster_variance(self, embeddings: Tensor, labels: Tensor) -> float:
        """Mean within-cluster variance (squared distance from centroid).

        Returns a non-negative float.
        """
        x = embeddings.float()
        ids = labels.long()
        x.shape[0]
        unique_clusters = ids.unique()

        total_var = 0.0
        for ct in unique_clusters:
            c = ct.item()
            mask = ids == c
            members = x[mask]
            if members.shape[0] == 0:
                continue
            centroid = members.mean(dim=0)
            var = ((members - centroid.unsqueeze(0)) ** 2).sum(dim=1).mean().item()
            total_var += var

        return float(total_var / max(len(unique_clusters), 1))


# ---------------------------------------------------------------------------
# Embedding Analysis Suite
# ---------------------------------------------------------------------------


class EmbeddingAnalysisSuite:
    """Orchestrates all embedding analysis routines in one object.

    Attributes
    ----------
    isotropy      : IsotropyMetrics
    dimensionality: IntrinsicDimensionality
    corrector     : AnisotropyCorrector
    clustering    : EmbeddingClusterAnalysis
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
        """Compute all metrics and return as a flat dict.

        Keys returned
        -------------
        isotropy_score, avg_cosine_similarity, partition_score,
        twonn_id, pca_dims, participation_ratio,
        silhouette_score, intra_cluster_variance
        """
        cfg = self.config

        iso = self.isotropy.compute_isotropy(embeddings, n_dirs=cfg.n_dirs)
        avg_cos = self.isotropy.compute_average_cosine_similarity(embeddings)
        partition = self.isotropy.compute_partition_score(embeddings, n_dirs=cfg.n_dirs)

        twonn = self.dimensionality.twonn_estimate(embeddings)
        pca_dims = float(
            self.dimensionality.pca_explained_variance(embeddings, threshold=cfg.threshold)
        )
        pr = self.dimensionality.participation_ratio(embeddings)

        labels = self.clustering.kmeans_fit(embeddings, n_iters=cfg.n_iters)
        sil = self.clustering.silhouette_score(embeddings, labels)
        icv = self.clustering.intra_cluster_variance(embeddings, labels)

        return {
            "isotropy_score": iso,
            "avg_cosine_similarity": avg_cos,
            "partition_score": partition,
            "twonn_id": twonn,
            "pca_dims": pca_dims,
            "participation_ratio": pr,
            "silhouette_score": sil,
            "intra_cluster_variance": icv,
        }
