"""Tests for src/eval/embedding_analysis_v3.py

Covers (16+ tests):
 1.  IsotropyMetrics.compute_isotropy         — result in [0, 1]
 2.  IsotropyMetrics.compute_isotropy         — large Gaussian → near 1
 3.  IsotropyMetrics.compute_average_cosine_similarity — near 0 for random unit vecs
 4.  IsotropyMetrics.compute_partition_score  — result in [0, 1]
 5.  IntrinsicDimensionality.twonn_estimate   — positive float
 6.  IntrinsicDimensionality.pca_explained_variance — result ≤ d
 7.  IntrinsicDimensionality.participation_ratio    — result in [1, d]
 8.  AnisotropyCorrector.fit_transform        — output is zero-mean
 9.  AnisotropyCorrector.inverse_transform    — round-trips to original
10.  AnisotropyCorrector.transform            — output shape matches input
11.  EmbeddingClusterAnalysis.kmeans_fit      — labels shape (N,)
12.  EmbeddingClusterAnalysis.kmeans_fit      — labels all in [0, n_clusters)
13.  EmbeddingClusterAnalysis.intra_cluster_variance — non-negative
14.  EmbeddingClusterAnalysis.silhouette_score       — in [-1, 1]
15.  EmbeddingAnalysisSuite.full_report       — all required keys present
16.  EmbeddingAnalysisSuite.full_report       — all values are finite floats
17.  Corrected embeddings have higher isotropy than raw anisotropic embeddings
18.  EmbeddingAnalysisConfig                  — default values
"""
from __future__ import annotations

import math

import pytest
import torch

from aurelius.eval.embedding_analysis_v3 import (
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

N = 32
D = 16
N_CLUSTERS = 4

_FULL_REPORT_KEYS = frozenset({
    "isotropy_score",
    "avg_cosine_similarity",
    "partition_score",
    "twonn_id",
    "pca_dims",
    "participation_ratio",
    "silhouette_score",
    "intra_cluster_variance",
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rand_emb() -> torch.Tensor:
    """Random Gaussian embeddings (N, D)."""
    torch.manual_seed(0)
    return torch.randn(N, D)


@pytest.fixture()
def unit_emb() -> torch.Tensor:
    """L2-normalised unit vectors (N, D)."""
    torch.manual_seed(1)
    x = torch.randn(N, D)
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)


@pytest.fixture()
def aniso_emb() -> torch.Tensor:
    """Heavily anisotropic embeddings (most variance in first two dims)."""
    torch.manual_seed(2)
    x = torch.randn(N, D)
    x[:, 0] = x[:, 0] * 50.0
    x[:, 1] = x[:, 1] * 50.0
    return x


@pytest.fixture()
def clustered_emb() -> torch.Tensor:
    """Four tight, well-separated clusters in D-dimensional space."""
    torch.manual_seed(3)
    centres = torch.tensor([
        [6.0, 0.0], [-6.0, 0.0], [0.0, 6.0], [0.0, -6.0]
    ])
    parts = []
    per_cluster = N // N_CLUSTERS
    for c in centres:
        noise = torch.randn(per_cluster, 2) * 0.15
        parts.append(c.unsqueeze(0) + noise)
    data2d = torch.cat(parts, dim=0)                            # (N, 2)
    pad = torch.randn(data2d.shape[0], D - 2) * 0.01
    return torch.cat([data2d, pad], dim=1)                      # (N, D)


# ===========================================================================
# 1. IsotropyMetrics — isotropy in [0, 1]
# ===========================================================================

def test_isotropy_in_unit_interval(rand_emb):
    m = IsotropyMetrics()
    score = m.compute_isotropy(rand_emb)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"isotropy {score!r} not in [0, 1]"


# ===========================================================================
# 2. IsotropyMetrics — large Gaussian → isotropy near 1
# ===========================================================================

def test_isotropy_gaussian_near_one():
    torch.manual_seed(7)
    x = torch.randn(512, D)
    m = IsotropyMetrics()
    score = m.compute_isotropy(x, n_dirs=200)
    assert score > 0.5, f"Expected isotropy > 0.5 for large Gaussian, got {score}"


# ===========================================================================
# 3. IsotropyMetrics — average cosine sim near 0 for random unit vectors
# ===========================================================================

def test_avg_cosine_sim_near_zero_for_unit_vecs(unit_emb):
    m = IsotropyMetrics()
    sim = m.compute_average_cosine_similarity(unit_emb)
    assert abs(sim) < 0.3, (
        f"Expected avg cos sim ≈ 0 for random unit vectors, got {sim}"
    )


# ===========================================================================
# 4. IsotropyMetrics — partition_score in [0, 1]
# ===========================================================================

def test_partition_score_in_unit_interval(rand_emb):
    m = IsotropyMetrics()
    score = m.compute_partition_score(rand_emb, n_dirs=50)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"partition_score {score!r} not in [0, 1]"


# ===========================================================================
# 5. IntrinsicDimensionality — twonn_estimate positive float
# ===========================================================================

def test_twonn_returns_positive_float(rand_emb):
    dim = IntrinsicDimensionality()
    est = dim.twonn_estimate(rand_emb)
    assert isinstance(est, float), f"twonn_estimate should return float, got {type(est)}"
    assert est > 0.0, f"twonn_estimate should be positive, got {est}"


# ===========================================================================
# 6. IntrinsicDimensionality — pca_explained_variance ≤ d
# ===========================================================================

def test_pca_explained_variance_le_d(rand_emb):
    dim = IntrinsicDimensionality()
    k = dim.pca_explained_variance(rand_emb, threshold=0.95)
    assert isinstance(k, int), f"pca_explained_variance should return int, got {type(k)}"
    assert 1 <= k <= D, f"pca_explained_variance {k} not in [1, {D}]"


# ===========================================================================
# 7. IntrinsicDimensionality — participation_ratio in [1, d]
# ===========================================================================

def test_participation_ratio_in_valid_range(rand_emb):
    dim = IntrinsicDimensionality()
    pr = dim.participation_ratio(rand_emb)
    assert isinstance(pr, float), f"participation_ratio should return float, got {type(pr)}"
    # Allow a small floating-point margin below 1
    assert pr >= 0.99, f"participation_ratio {pr} < 1"
    assert pr <= D + 1e-4, f"participation_ratio {pr} > d={D}"


# ===========================================================================
# 8. AnisotropyCorrector — fit_transform produces zero-mean output
# ===========================================================================

def test_fit_transform_zero_mean(rand_emb):
    corr = AnisotropyCorrector()
    out = corr.fit_transform(rand_emb)
    max_abs_mean = out.mean(dim=0).abs().max().item()
    assert max_abs_mean < 1e-4, (
        f"fit_transform output should be zero-mean; max|mean| = {max_abs_mean}"
    )


# ===========================================================================
# 9. AnisotropyCorrector — inverse_transform round-trips
# ===========================================================================

def test_corrector_inverse_transform_roundtrip(rand_emb):
    corr = AnisotropyCorrector()
    transformed = corr.fit_transform(rand_emb)
    recovered = corr.inverse_transform(transformed)
    max_err = (recovered - rand_emb.float()).abs().max().item()
    assert max_err < 1e-3, f"Round-trip max error {max_err} exceeds threshold"


# ===========================================================================
# 10. AnisotropyCorrector — transform output shape unchanged
# ===========================================================================

def test_corrector_shape_preserved(rand_emb):
    corr = AnisotropyCorrector()
    out = corr.fit_transform(rand_emb)
    assert out.shape == rand_emb.shape, (
        f"Shape after fit_transform {out.shape} != input {rand_emb.shape}"
    )


# ===========================================================================
# 11. EmbeddingClusterAnalysis — kmeans_fit labels shape [N]
# ===========================================================================

def test_kmeans_labels_shape(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    assert labels.shape == (N,), f"labels shape {labels.shape} != ({N},)"


# ===========================================================================
# 12. EmbeddingClusterAnalysis — labels all in [0, n_clusters)
# ===========================================================================

def test_kmeans_labels_in_valid_range(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    assert int(labels.min().item()) >= 0, "labels contain a negative value"
    assert int(labels.max().item()) < N_CLUSTERS, (
        f"labels contain a value >= n_clusters={N_CLUSTERS}"
    )


# ===========================================================================
# 13. EmbeddingClusterAnalysis — intra_cluster_variance non-negative
# ===========================================================================

def test_intra_cluster_variance_non_negative(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    icv = clust.intra_cluster_variance(rand_emb, labels)
    assert isinstance(icv, float)
    assert icv >= 0.0, f"intra_cluster_variance is negative: {icv}"


# ===========================================================================
# 14. EmbeddingClusterAnalysis — silhouette_score in [-1, 1]
# ===========================================================================

def test_silhouette_score_in_range(rand_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(rand_emb)
    score = clust.silhouette_score(rand_emb, labels)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0, f"silhouette score {score} not in [-1, 1]"


# ===========================================================================
# 15. EmbeddingAnalysisSuite — full_report returns dict with all required keys
# ===========================================================================

def test_full_report_has_all_keys(rand_emb):
    cfg = EmbeddingAnalysisConfig(n_dirs=20, n_clusters=N_CLUSTERS, n_iters=10)
    suite = EmbeddingAnalysisSuite(cfg)
    report = suite.full_report(rand_emb)
    missing = _FULL_REPORT_KEYS - set(report.keys())
    assert not missing, f"full_report missing keys: {missing}"


# ===========================================================================
# 16. EmbeddingAnalysisSuite — all values are finite floats
# ===========================================================================

def test_full_report_values_finite(rand_emb):
    cfg = EmbeddingAnalysisConfig(n_dirs=20, n_clusters=N_CLUSTERS, n_iters=10)
    suite = EmbeddingAnalysisSuite(cfg)
    report = suite.full_report(rand_emb)
    for key, val in report.items():
        assert isinstance(val, float), f"'{key}' value is {type(val)}, expected float"
        assert math.isfinite(val), f"'{key}' value is not finite: {val}"


# ===========================================================================
# 17. Corrected embeddings have higher isotropy than raw anisotropic embeddings
# ===========================================================================

def test_corrected_embeddings_higher_isotropy(aniso_emb):
    m = IsotropyMetrics()
    corr = AnisotropyCorrector()

    iso_before = m.compute_isotropy(aniso_emb, n_dirs=100)
    corrected = corr.fit_transform(aniso_emb)
    iso_after = m.compute_isotropy(corrected, n_dirs=100)

    assert iso_after > iso_before, (
        f"Isotropy should improve after correction: "
        f"before={iso_before:.4f}, after={iso_after:.4f}"
    )


# ===========================================================================
# 18. EmbeddingAnalysisConfig — default values
# ===========================================================================

def test_config_defaults():
    cfg = EmbeddingAnalysisConfig()
    assert cfg.n_dirs == 50
    assert cfg.threshold == 0.95
    assert cfg.n_clusters == 4
    assert cfg.n_iters == 20


# ===========================================================================
# Bonus: well-separated clusters yield positive silhouette score
# ===========================================================================

def test_silhouette_positive_for_well_separated_clusters(clustered_emb):
    clust = EmbeddingClusterAnalysis(n_clusters=N_CLUSTERS)
    labels = clust.kmeans_fit(clustered_emb, n_iters=50, seed=0)
    score = clust.silhouette_score(clustered_emb, labels)
    # Well-separated clusters should yield a clearly positive silhouette
    assert score > 0.0, f"Expected positive silhouette for well-separated clusters, got {score}"


# ===========================================================================
# Bonus: transform raises before fit
# ===========================================================================

def test_transform_raises_before_fit(rand_emb):
    corr = AnisotropyCorrector()
    with pytest.raises(RuntimeError):
        corr.transform(rand_emb)


# ===========================================================================
# Bonus: inverse_transform raises before fit
# ===========================================================================

def test_inverse_transform_raises_before_fit(rand_emb):
    corr = AnisotropyCorrector()
    with pytest.raises(RuntimeError):
        corr.inverse_transform(rand_emb)
