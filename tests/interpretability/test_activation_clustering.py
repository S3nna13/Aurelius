"""
tests/interpretability/test_activation_clustering.py

Unit tests for src/interpretability/activation_clustering.py.

Uses a tiny AureliusTransformer so tests run fast on CPU.
Tiny config: n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
             head_dim=16, d_ff=128, vocab_size=256, max_seq_len=32.
N=16 samples, T=8 tokens per sample.
"""

from __future__ import annotations

import math

import torch

from src.interpretability.activation_clustering import (
    ClusteringConfig,
    ClusterResult,
    cluster_token_sequences,
    extract_layer_activations,
    kmeans_cluster,
    nearest_cluster_examples,
    pairwise_distances,
    silhouette_score,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

TINY_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)

N = 16  # number of samples
T = 8  # sequence length
D = 64  # d_model
K = 4  # n_clusters
VOCAB = 256

torch.manual_seed(42)


def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(TINY_CONFIG)


def _make_input_ids() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, VOCAB, (N, T))


def _make_activations(n: int = N, d: int = D) -> torch.Tensor:
    torch.manual_seed(13)
    return torch.randn(n, d)


# ---------------------------------------------------------------------------
# 1. pairwise_distances shape is (N, M)
# ---------------------------------------------------------------------------


def test_pairwise_distances_shape():
    torch.manual_seed(1)
    a = torch.randn(N, D)
    b = torch.randn(K, D)
    dist = pairwise_distances(a, b)
    assert dist.shape == (N, K), f"Expected shape ({N}, {K}), got {dist.shape}"


# ---------------------------------------------------------------------------
# 2. pairwise_distances diagonal is 0 when a == b
# ---------------------------------------------------------------------------


def test_pairwise_distances_diagonal_zero():
    torch.manual_seed(2)
    a = torch.randn(N, D)
    dist = pairwise_distances(a, a)
    diag = dist.diagonal()
    assert torch.allclose(diag, torch.zeros(N), atol=1e-4), (
        f"Diagonal should be zero, got max {diag.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 3. kmeans_cluster returns ClusterResult
# ---------------------------------------------------------------------------


def test_kmeans_returns_cluster_result():
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    assert isinstance(result, ClusterResult)


# ---------------------------------------------------------------------------
# 4. kmeans_cluster labels shape is (N,)
# ---------------------------------------------------------------------------


def test_kmeans_labels_shape():
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    assert result.labels.shape == (N,), f"Expected labels shape ({N},), got {result.labels.shape}"


# ---------------------------------------------------------------------------
# 5. kmeans_cluster unique labels count <= n_clusters
# ---------------------------------------------------------------------------


def test_kmeans_unique_labels_count():
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    n_unique = result.labels.unique().shape[0]
    assert n_unique <= K, f"Expected at most {K} unique labels, got {n_unique}"


# ---------------------------------------------------------------------------
# 6. kmeans_cluster inertia is finite and non-negative
# ---------------------------------------------------------------------------


def test_kmeans_inertia_finite_nonneg():
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    assert isinstance(result.inertia, float)
    assert result.inertia >= 0.0, f"Inertia must be non-negative, got {result.inertia}"
    assert math.isfinite(result.inertia), f"Inertia must be finite, got {result.inertia}"


# ---------------------------------------------------------------------------
# 7. silhouette_score returns float in [-1, 1]
# ---------------------------------------------------------------------------


def test_silhouette_score_range():
    torch.manual_seed(5)
    acts = torch.randn(20, D)
    labels = torch.randint(0, K, (20,))
    score = silhouette_score(acts, labels)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0, f"Silhouette score must be in [-1, 1], got {score}"


# ---------------------------------------------------------------------------
# 8. silhouette_score with well-separated clusters -> score > 0
# ---------------------------------------------------------------------------


def test_silhouette_score_well_separated():
    """Clusters placed far apart in d-dimensional space should yield score > 0."""
    torch.manual_seed(6)
    n_per_cluster = 10
    n_clusters_local = 4
    total = n_per_cluster * n_clusters_local

    acts = torch.randn(total, D) * 0.05  # tight noise
    labels = torch.zeros(total, dtype=torch.long)
    for c in range(n_clusters_local):
        start = c * n_per_cluster
        end = start + n_per_cluster
        acts[start:end, c] += 100.0
        labels[start:end] = c

    score = silhouette_score(acts, labels)
    assert score > 0.0, f"Expected positive silhouette for well-separated clusters, got {score}"


# ---------------------------------------------------------------------------
# 9. extract_layer_activations position="last" returns (N, d_model)
# ---------------------------------------------------------------------------


def test_extract_layer_activations_last_shape():
    torch.manual_seed(9)
    model = _make_model()
    input_ids = _make_input_ids()
    acts = extract_layer_activations(model, input_ids, layer_idx=-1, position="last")
    assert acts.shape == (N, D), f"Expected shape ({N}, {D}), got {acts.shape}"


# ---------------------------------------------------------------------------
# 10. extract_layer_activations position="mean" returns (N, d_model)
# ---------------------------------------------------------------------------


def test_extract_layer_activations_mean_shape():
    torch.manual_seed(10)
    model = _make_model()
    input_ids = _make_input_ids()
    acts = extract_layer_activations(model, input_ids, layer_idx=0, position="mean")
    assert acts.shape == (N, D), f"Expected shape ({N}, {D}), got {acts.shape}"


# ---------------------------------------------------------------------------
# 11. cluster_token_sequences returns ClusterResult
# ---------------------------------------------------------------------------


def test_cluster_token_sequences_returns_cluster_result():
    torch.manual_seed(11)
    model = _make_model()
    input_ids = _make_input_ids()
    config = ClusteringConfig(
        n_clusters=K,
        n_init=2,
        max_iter=20,
        layer_idx=-1,
        position="last",
    )
    result = cluster_token_sequences(model, input_ids, config)
    assert isinstance(result, ClusterResult)


# ---------------------------------------------------------------------------
# 12. nearest_cluster_examples returns k indices
# ---------------------------------------------------------------------------


def test_nearest_cluster_examples_returns_k_indices():
    torch.manual_seed(12)
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    top_k = 3
    indices = nearest_cluster_examples(acts, result, cluster_idx=0, top_k=top_k)
    assert indices.shape == (top_k,), f"Expected {top_k} indices, got shape {indices.shape}"


# ---------------------------------------------------------------------------
# 13. All returned indices are valid (in range [0, N))
# ---------------------------------------------------------------------------


def test_nearest_cluster_examples_indices_valid():
    torch.manual_seed(13)
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    top_k = 5
    for c in range(K):
        indices = nearest_cluster_examples(acts, result, cluster_idx=c, top_k=top_k)
        assert (indices >= 0).all(), "All indices must be >= 0"
        assert (indices < N).all(), f"All indices must be < {N}, got {indices}"


# ---------------------------------------------------------------------------
# 14. ClusteringConfig defaults
# ---------------------------------------------------------------------------


def test_clustering_config_defaults():
    cfg = ClusteringConfig()
    assert cfg.n_clusters == 4
    assert cfg.n_init == 3
    assert cfg.max_iter == 100
    assert cfg.tol == 1e-4
    assert cfg.layer_idx == -1
    assert cfg.position == "last"


# ---------------------------------------------------------------------------
# 15. centroids shape is (n_clusters, d_model)
# ---------------------------------------------------------------------------


def test_kmeans_centroids_shape():
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    assert result.centroids.shape[1] == D, (
        f"Centroids second dim should be {D}, got {result.centroids.shape[1]}"
    )
    assert result.centroids.shape[0] <= K, (
        f"Centroids first dim should be <= {K}, got {result.centroids.shape[0]}"
    )


# ---------------------------------------------------------------------------
# 16. n_iter is a positive integer
# ---------------------------------------------------------------------------


def test_kmeans_n_iter_positive_int():
    acts = _make_activations()
    result = kmeans_cluster(acts, n_clusters=K, n_init=2, max_iter=20, seed=42)
    assert isinstance(result.n_iter, int)
    assert result.n_iter >= 1, f"n_iter should be >= 1, got {result.n_iter}"
