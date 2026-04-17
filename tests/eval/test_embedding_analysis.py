"""Tests for src/eval/embedding_analysis.py"""
from __future__ import annotations

import math

import pytest
import torch

from aurelius.eval.embedding_analysis import (
    EmbeddingAnalysisConfig,
    EmbeddingClustering,
    IntrinsicDimensionEstimator,
    IsotropyAnalyzer,
    NearestNeighborRetriever,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

torch.manual_seed(42)

N, D = 64, 32  # moderate size for speed


@pytest.fixture()
def random_embeddings() -> torch.Tensor:
    return torch.randn(N, D)


@pytest.fixture()
def isotropic_embeddings() -> torch.Tensor:
    """Gaussian embeddings — should be nearly isotropic."""
    return torch.randn(200, D)


@pytest.fixture()
def collapsed_embeddings() -> torch.Tensor:
    """All embeddings in the same direction (rank-1) — minimal isotropy."""
    direction = torch.randn(D)
    direction = direction / direction.norm()
    scales = torch.randn(200, 1).abs() + 0.1
    return scales * direction.unsqueeze(0)


# ===========================================================================
# EmbeddingAnalysisConfig
# ===========================================================================

def test_config_defaults() -> None:
    cfg = EmbeddingAnalysisConfig()
    assert cfg.n_components == 50
    assert cfg.n_neighbors == 10
    assert cfg.normalize is True


# ===========================================================================
# IsotropyAnalyzer
# ===========================================================================

class TestIsotropyAnalyzer:
    def test_compute_isotropy_in_unit_interval(self, random_embeddings: torch.Tensor) -> None:
        analyzer = IsotropyAnalyzer()
        value = analyzer.compute_isotropy(random_embeddings)
        assert 0.0 <= value <= 1.0, f"isotropy {value} not in [0, 1]"

    def test_isotropic_embeddings_high_isotropy(
        self, isotropic_embeddings: torch.Tensor
    ) -> None:
        analyzer = IsotropyAnalyzer()
        value = analyzer.compute_isotropy(isotropic_embeddings)
        # Random Gaussian should produce reasonable spread; typically > 0.01
        assert value > 0.0, "Expected non-zero isotropy for random Gaussian embeddings"

    def test_collapsed_embeddings_low_isotropy(
        self, collapsed_embeddings: torch.Tensor
    ) -> None:
        analyzer = IsotropyAnalyzer()
        value = analyzer.compute_isotropy(collapsed_embeddings)
        # Rank-1 matrix: only one non-zero eigenvalue → ratio close to 0
        assert value < 0.05, f"Expected near-zero isotropy for rank-1 embeddings, got {value}"

    def test_average_cosine_similarity_in_range(
        self, random_embeddings: torch.Tensor
    ) -> None:
        analyzer = IsotropyAnalyzer()
        value = analyzer.average_cosine_similarity(random_embeddings)
        assert -1.0 <= value <= 1.0, f"cosine similarity {value} out of [-1, 1]"

    def test_average_cosine_similarity_large_batch(self) -> None:
        """Triggers the sampling path (N > 100)."""
        analyzer = IsotropyAnalyzer()
        large = torch.randn(200, D)
        value = analyzer.average_cosine_similarity(large)
        assert -1.0 <= value <= 1.0

    def test_spectral_entropy_non_negative(
        self, random_embeddings: torch.Tensor
    ) -> None:
        analyzer = IsotropyAnalyzer()
        value = analyzer.spectral_entropy(random_embeddings)
        assert value >= 0.0, f"spectral entropy {value} is negative"

    def test_spectral_entropy_collapsed_lower_than_random(
        self,
        isotropic_embeddings: torch.Tensor,
        collapsed_embeddings: torch.Tensor,
    ) -> None:
        analyzer = IsotropyAnalyzer()
        h_iso = analyzer.spectral_entropy(isotropic_embeddings)
        h_col = analyzer.spectral_entropy(collapsed_embeddings)
        assert h_iso > h_col, "Isotropic embeddings should have higher spectral entropy"


# ===========================================================================
# IntrinsicDimensionEstimator
# ===========================================================================

class TestIntrinsicDimensionEstimator:
    def test_twonn_estimate_positive(self, random_embeddings: torch.Tensor) -> None:
        estimator = IntrinsicDimensionEstimator(n_neighbors=2)
        id_est = estimator.twonn_estimate(random_embeddings)
        assert id_est > 0.0, f"TwoNN estimate should be positive, got {id_est}"

    def test_pca_explained_variance_ratio_monotone(
        self, random_embeddings: torch.Tensor
    ) -> None:
        estimator = IntrinsicDimensionEstimator()
        cumvar = estimator.pca_explained_variance_ratio(random_embeddings)
        diffs = cumvar[1:] - cumvar[:-1]
        assert (diffs >= -1e-6).all(), "Cumulative variance ratio must be non-decreasing"

    def test_pca_explained_variance_ratio_last_near_one(
        self, random_embeddings: torch.Tensor
    ) -> None:
        estimator = IntrinsicDimensionEstimator()
        cumvar = estimator.pca_explained_variance_ratio(random_embeddings)
        assert abs(cumvar[-1].item() - 1.0) < 1e-4, (
            f"Last cumulative variance ratio should be ~1.0, got {cumvar[-1].item()}"
        )


# ===========================================================================
# EmbeddingClustering
# ===========================================================================

class TestEmbeddingClustering:
    def test_kmeans_returns_correct_shapes(
        self, random_embeddings: torch.Tensor
    ) -> None:
        k = 4
        clustering = EmbeddingClustering(n_clusters=k)
        cluster_ids, centroids = clustering.kmeans(random_embeddings)
        assert cluster_ids.shape == (N,), f"cluster_ids shape {cluster_ids.shape} != ({N},)"
        assert centroids.shape == (k, D), f"centroids shape {centroids.shape} != ({k}, {D})"

    def test_cluster_ids_valid_range(self, random_embeddings: torch.Tensor) -> None:
        k = 4
        clustering = EmbeddingClustering(n_clusters=k)
        cluster_ids, _ = clustering.kmeans(random_embeddings)
        assert cluster_ids.min() >= 0, "cluster_ids contains negative values"
        assert cluster_ids.max() < k, f"cluster_ids contains value >= n_clusters ({k})"

    def test_silhouette_score_in_range(self, random_embeddings: torch.Tensor) -> None:
        clustering = EmbeddingClustering(n_clusters=3)
        cluster_ids, _ = clustering.kmeans(random_embeddings)
        score = clustering.silhouette_score(random_embeddings, cluster_ids)
        assert -1.0 <= score <= 1.0, f"silhouette score {score} out of [-1, 1]"


# ===========================================================================
# NearestNeighborRetriever
# ===========================================================================

class TestNearestNeighborRetriever:
    @pytest.fixture()
    def retriever_and_index(self) -> tuple[NearestNeighborRetriever, torch.Tensor]:
        index = torch.randn(50, D)
        return NearestNeighborRetriever(index), index

    def test_search_returns_correct_shapes(
        self,
        retriever_and_index: tuple[NearestNeighborRetriever, torch.Tensor],
    ) -> None:
        retriever, _ = retriever_and_index
        queries = torch.randn(8, D)
        k = 5
        indices, scores = retriever.search(queries, k=k)
        assert indices.shape == (8, k), f"indices shape {indices.shape} != (8, {k})"
        assert scores.shape == (8, k), f"scores shape {scores.shape} != (8, {k})"

    def test_search_indices_in_valid_range(
        self,
        retriever_and_index: tuple[NearestNeighborRetriever, torch.Tensor],
    ) -> None:
        retriever, index = retriever_and_index
        M = index.shape[0]
        queries = torch.randn(8, D)
        indices, _ = retriever.search(queries, k=5)
        assert indices.min() >= 0, "Indices contain negatives"
        assert indices.max() < M, f"Indices exceed index size {M}"

    def test_recall_at_k_in_unit_interval(
        self,
        retriever_and_index: tuple[NearestNeighborRetriever, torch.Tensor],
    ) -> None:
        retriever, index = retriever_and_index
        M = index.shape[0]
        queries = torch.randn(10, D)
        gt = torch.randint(0, M, (10,))
        recall = retriever.recall_at_k(queries, gt, k=5)
        assert 0.0 <= recall <= 1.0, f"recall@k {recall} not in [0, 1]"

    def test_recall_at_k_perfect_when_queries_in_index(self) -> None:
        """Queries that ARE the indexed vectors should have recall@k = 1."""
        index = torch.randn(20, D)
        retriever = NearestNeighborRetriever(index)
        # Use 5 of the index vectors as queries; ground truth = their positions
        query_indices = [0, 3, 7, 12, 19]
        queries = index[query_indices]
        gt = torch.tensor(query_indices, dtype=torch.long)
        recall = retriever.recall_at_k(queries, gt, k=1)
        assert recall == 1.0, f"Expected recall=1.0 when queries are in index, got {recall}"
