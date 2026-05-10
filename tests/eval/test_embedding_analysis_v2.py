"""Tests for src/eval/embedding_analysis_v2.py

Covers:
- IsotropyMetrics: range, Gaussian near-1, cosine sim near 0
- IntrinsicDimensionality: twonn positive, pca <= d, participation ratio in [1, d]
- AnisotropyCorrector: zero-mean after fit_transform, round-trip, shape preserved
- EmbeddingClusterAnalysis: label shape, range, non-negative variance, silhouette range
- EmbeddingAnalysisSuite: full_report dict keys present, values finite
- Corrected embeddings have higher isotropy than raw anisotropic embeddings
"""

from __future__ import annotations

import math

import pytest
import torch
from aurelius.eval.embedding_analysis_v2 import (
    AnisotropyCorrector,
    EmbeddingAnalysisConfig,
    EmbeddingAnalysisSuite,
    EmbeddingClusterAnalysis,
    IntrinsicDimensionality,
    IsotropyMetrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
torch.manual_seed(42)
N, D = 32, 16
N_CLUSTERS = 4

_FULL_REPORT_KEYS = {
    "isotropy_score",
    "avg_cosine_similarity",
    "partition_score",
    "twonn_id",
    "pca_dims",
    "participation_ratio",
    "silhouette_score",
    "intra_cluster_variance",
}

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rand_emb() -> torch.Tensor:
    """Random Gaussian embeddings (N, D)."""
    return torch.randn(N, D)


@pytest.fixture()
def unit_emb() -> torch.Tensor:
    """L2-normalised random unit vectors (N, D)."""
    x = torch.randn(N, D)
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)


@pytest.fixture()
def aniso_emb() -> torch.Tensor:
    """Heavily anisotropic embeddings: most variance along one axis."""
    x = torch.randn(N, D)
    # Scale first 2 dims by 50x; remaining dims have unit scale
    x[:, 0] *= 50.0
    x[:, 1] *= 50.0
    return x


@pytest.fixture()
def clustered_emb() -> torch.Tensor:
    """Four tight, well-separated clusters."""
    centres = torch.tensor(
        [
            [5.0, 0.0],
            [-5.0, 0.0],
            [0.0, 5.0],
            [0.0, -5.0],
        ]
    )
    parts = []
    for c in centres:
        noise = torch.randn(N // N_CLUSTERS, 2) * 0.2
        parts.append(c.unsqueeze(0) + noise)
    data2d = torch.cat(parts, dim=0)
    # Pad to D dimensions with small noise
    pad = torch.randn(data2d.shape[0], D - 2) * 0.01
    return torch.cat([data2d, pad], dim=1)


# ===========================================================================
# 1. IsotropyMetrics — isotropy in [0, 1]
# ===========================================================================


def test_isotropy_in_unit_interval(rand_emb):
    metrics = IsotropyMetrics()
    score = metrics.compute_isotropy(rand_emb)
    assert 0.0 <= score <= 1.0, f"isotropy {score} not in [0, 1]"


# ===========================================================================
# 2. IsotropyMetrics — Gaussian → near 1
# ===========================================================================


def test_isotropy_gaussian_near_one():
    """Large Gaussian sample should give isotropy close to 1."""
    torch.manual_seed(7)
    x = torch.randn(512, D)
    metrics = IsotropyMetrics()
    score = metrics.compute_isotropy(x, n_dirs=200)
    assert score > 0.5, f"Expected isotropy > 0.5 for large Gaussian, got {score}"


# ===========================================================================
# 3. IsotropyMetrics — average cosine similarity near 0 for random unit vectors
# ===========================================================================


def test_avg_cosine_similarity_near_zero(unit_emb):
    metrics = IsotropyMetrics()
    sim = metrics.compute_average_cosine_similarity(unit_emb)
    # Random unit vectors in high-d space → mean ≈ 0
    assert abs(sim) < 0.3, f"Expected avg cos sim near 0 for random unit vecs, got {sim}"


# ===========================================================================
# 4. IntrinsicDimensionality — twonn_estimate positive float
# ===========================================================================


def test_twonn_returns_positive(rand_emb):
    dim = IntrinsicDimensionality()
    est = dim.twonn_estimate(rand_emb)
    assert isinstance(est, float), "twonn_estimate should return float"
    assert est > 0.0, f"twonn_estimate should be positive, got {est}"


# ===========================================================================
# 5. IntrinsicDimensionality — pca_explained_variance <= d
# ===========================================================================


def test_pca_explained_variance_le_d(rand_emb):
    dim = IntrinsicDimensionality()
    k = dim.pca_explained_variance(rand_emb, threshold=0.95)
    assert isinstance(k, int), "pca_explained_variance should return int"
    assert 1 <= k <= D, f"pca_explained_variance {k} not in [1, {D}]"


# ===========================================================================
# 6. IntrinsicDimensionality — participation_ratio in [1, d]
# ===========================================================================


def test_participation_ratio_in_range(rand_emb):
    dim = IntrinsicDimensionality()
    pr = dim.participation_ratio(rand_emb)
    assert isinstance(pr, float), "participation_ratio should return float"
    # Allow small floating-point slack below 1
    assert pr >= 0.99, f"participation_ratio {pr} < 1"
    assert pr <= D + 1e-4, f"participation_ratio {pr} > d={D}"


# ===========================================================================
# 7. AnisotropyCorrector — fit_transform zero-means result
# ===========================================================================


def test_corrector_fit_transform_zero_mean(rand_emb):
    corrector = AnisotropyCorrector()
    transformed = corrector.fit_transform(rand_emb)
    mean = transformed.mean(dim=0)
    assert mean.abs().max().item() < 1e-4, (
        f"fit_transform output should have zero mean, max abs mean = {mean.abs().max().item()}"
    )


# ===========================================================================
# 8. AnisotropyCorrector — inverse_transform round-trips
# ===========================================================================


def test_corrector_inverse_transform_roundtrip(rand_emb):
    corrector = AnisotropyCorrector()
    transformed = corrector.fit_transform(rand_emb)
    reconstructed = corrector.inverse_transform(transformed)
    diff = (reconstructed - rand_emb.float()).abs().max().item()
    assert diff < 1e-3, f"Round-trip error too large: {diff}"


# ===========================================================================
# 9. AnisotropyCorrector — output shape unchanged
# ===========================================================================


def test_corrector_transform_shape_unchanged(rand_emb):
    corrector = AnisotropyCorrector()
    out = corrector.fit_transform(rand_emb)
    assert out.shape == rand_emb.shape, (
        f"Shape after transform {out.shape} != input shape {rand_emb.shape}"
    )


# ===========================================================================
# 10. EmbeddingClusterAnalysis — kmeans labels shape [N]
# ===========================================================================


def test_kmeans_labels_shape(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    assert labels.shape == (N,), f"labels shape {labels.shape} != ({N},)"


# ===========================================================================
# 11. EmbeddingClusterAnalysis — labels all in [0, n_clusters)
# ===========================================================================


def test_kmeans_labels_in_valid_range(rand_emb):
    k = N_CLUSTERS
    clust = EmbeddingClusterAnalysis(n_clusters=k)
    labels = clust.kmeans_fit(rand_emb)
    assert labels.min().item() >= 0, "labels contain negative values"
    assert labels.max().item() < k, f"labels contain value >= n_clusters ({k})"


# ===========================================================================
# 12. EmbeddingClusterAnalysis — intra_cluster_variance non-negative
# ===========================================================================


def test_intra_cluster_variance_non_negative(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    icv = clust.intra_cluster_variance(rand_emb, labels)
    assert icv >= 0.0, f"intra_cluster_variance is negative: {icv}"


# ===========================================================================
# 13. silhouette_score in [-1, 1]
# ===========================================================================


def test_silhouette_score_in_range(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    score = clust.silhouette_score(rand_emb, labels)
    assert -1.0 <= score <= 1.0, f"silhouette score {score} not in [-1, 1]"


# ===========================================================================
# 14. EmbeddingAnalysisSuite — full_report returns dict with all keys
# ===========================================================================


def test_full_report_keys(rand_emb):
    cfg = EmbeddingAnalysisConfig(n_dirs=20, n_clusters=N_CLUSTERS, n_iters=10)
    suite = EmbeddingAnalysisSuite(cfg)
    report = suite.full_report(rand_emb)
    missing = _FULL_REPORT_KEYS - set(report.keys())
    assert not missing, f"full_report missing keys: {missing}"


# ===========================================================================
# 15. full_report values are finite floats
# ===========================================================================


def test_full_report_values_finite(rand_emb):
    cfg = EmbeddingAnalysisConfig(n_dirs=20, n_clusters=N_CLUSTERS, n_iters=10)
    suite = EmbeddingAnalysisSuite(cfg)
    report = suite.full_report(rand_emb)
    for key, val in report.items():
        assert isinstance(val, float), f"{key} is not float: {type(val)}"
        assert math.isfinite(val), f"{key} is not finite: {val}"


# ===========================================================================
# 16. Corrected embeddings have higher isotropy than raw anisotropic embeddings
# ===========================================================================


def test_corrected_embeddings_higher_isotropy(aniso_emb):
    metrics = IsotropyMetrics()
    corrector = AnisotropyCorrector()

    iso_before = metrics.compute_isotropy(aniso_emb, n_dirs=100)
    corrected = corrector.fit_transform(aniso_emb)
    iso_after = metrics.compute_isotropy(corrected, n_dirs=100)

    assert iso_after > iso_before, (
        f"Expected isotropy to improve after correction: "
        f"before={iso_before:.4f}, after={iso_after:.4f}"
    )


# ===========================================================================
# Bonus: EmbeddingAnalysisConfig defaults
# ===========================================================================


def test_config_defaults():
    cfg = EmbeddingAnalysisConfig()
    assert cfg.n_dirs == 50
    assert cfg.threshold == 0.95
    assert cfg.n_clusters == 4
    assert cfg.n_iters == 20


# ===========================================================================
# Bonus: partition_score in [0, 1]
# ===========================================================================


def test_partition_score_in_unit_interval(rand_emb):
    metrics = IsotropyMetrics()
    score = metrics.compute_partition_score(rand_emb, n_dirs=50)
    assert 0.0 <= score <= 1.0, f"partition_score {score} not in [0, 1]"
