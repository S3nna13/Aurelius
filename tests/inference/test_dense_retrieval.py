"""Tests for src/inference/dense_retrieval.py — 15 tests."""

from __future__ import annotations

import pytest
import torch

from src.inference.dense_retrieval import (
    DenseRetrievalConfig,
    DenseRetriever,
    IVFIndex,
    ProductQuantizer,
    compute_similarity,
    kmeans_cluster,
)

# ---------------------------------------------------------------------------
# Test configuration (small for speed)
# ---------------------------------------------------------------------------

EMBED_DIM = 8
N_SUBSPACES = 2
N_CENTROIDS = 4
N_CLUSTERS = 4
N_PROBE = 2


def small_cfg(**kwargs) -> DenseRetrievalConfig:
    defaults = dict(
        embed_dim=EMBED_DIM,
        n_subspaces=N_SUBSPACES,
        n_centroids=N_CENTROIDS,
        n_probe=N_PROBE,
        n_clusters=N_CLUSTERS,
        metric="cosine",
    )
    defaults.update(kwargs)
    return DenseRetrievalConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = DenseRetrievalConfig()
    assert cfg.embed_dim == 64
    assert cfg.n_subspaces == 4
    assert cfg.n_centroids == 16
    assert cfg.n_probe == 4
    assert cfg.n_clusters == 8
    assert cfg.metric == "cosine"


# ---------------------------------------------------------------------------
# 2. test_compute_similarity_cosine_shape
# ---------------------------------------------------------------------------

def test_compute_similarity_cosine_shape():
    Q, N, D = 3, 10, EMBED_DIM
    query = torch.randn(Q, D)
    corpus = torch.randn(N, D)
    sim = compute_similarity(query, corpus, metric="cosine")
    assert sim.shape == (Q, N)


# ---------------------------------------------------------------------------
# 3. test_compute_similarity_self — cosine(x, x) → ~1.0
# ---------------------------------------------------------------------------

def test_compute_similarity_self():
    torch.manual_seed(0)
    x = torch.randn(5, EMBED_DIM)
    sim = compute_similarity(x, x, metric="cosine")
    # Diagonal should be ~1.0
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones(5), atol=1e-5), f"diagonal={diag}"


# ---------------------------------------------------------------------------
# 4. test_compute_similarity_l2_nonneg — l2 returns non-positive values
# ---------------------------------------------------------------------------

def test_compute_similarity_l2_nonneg():
    torch.manual_seed(1)
    Q, N, D = 3, 7, EMBED_DIM
    query = torch.randn(Q, D)
    corpus = torch.randn(N, D)
    sim = compute_similarity(query, corpus, metric="l2")
    assert sim.shape == (Q, N)
    assert (sim <= 0).all(), f"Expected all non-positive, got max={sim.max()}"


# ---------------------------------------------------------------------------
# 5. test_kmeans_cluster_shapes
# ---------------------------------------------------------------------------

def test_kmeans_cluster_shapes():
    torch.manual_seed(2)
    N, D, k = 50, EMBED_DIM, 6
    vectors = torch.randn(N, D)
    centroids, assignments = kmeans_cluster(vectors, k=k)
    assert centroids.shape == (k, D)
    assert assignments.shape == (N,)


# ---------------------------------------------------------------------------
# 6. test_kmeans_cluster_assignment_range
# ---------------------------------------------------------------------------

def test_kmeans_cluster_assignment_range():
    torch.manual_seed(3)
    N, D, k = 40, EMBED_DIM, 5
    vectors = torch.randn(N, D)
    centroids, assignments = kmeans_cluster(vectors, k=k)
    assert assignments.min().item() >= 0
    assert assignments.max().item() < k


# ---------------------------------------------------------------------------
# 7. test_pq_encode_shape
# ---------------------------------------------------------------------------

def test_pq_encode_shape():
    torch.manual_seed(4)
    cfg = small_cfg()
    pq = ProductQuantizer(cfg)
    fit_vecs = torch.randn(20, EMBED_DIM)
    pq.fit(fit_vecs)

    test_vecs = torch.randn(5, EMBED_DIM)
    codes = pq.encode(test_vecs)
    assert codes.shape == (5, N_SUBSPACES)


# ---------------------------------------------------------------------------
# 8. test_pq_decode_shape
# ---------------------------------------------------------------------------

def test_pq_decode_shape():
    torch.manual_seed(5)
    cfg = small_cfg()
    pq = ProductQuantizer(cfg)
    fit_vecs = torch.randn(20, EMBED_DIM)
    pq.fit(fit_vecs)

    test_vecs = torch.randn(5, EMBED_DIM)
    codes = pq.encode(test_vecs)
    decoded = pq.decode(codes)
    assert decoded.shape == (5, EMBED_DIM)


# ---------------------------------------------------------------------------
# 9. test_pq_encode_decode_approx — decoded ≈ original within reasonable error
# ---------------------------------------------------------------------------

def test_pq_encode_decode_approx():
    torch.manual_seed(6)
    cfg = small_cfg(n_centroids=16)  # more centroids → better approximation
    pq = ProductQuantizer(cfg)
    # Use structured data so centroids capture it well
    fit_vecs = torch.randn(100, EMBED_DIM)
    pq.fit(fit_vecs)

    # Encode and decode the same fit vectors (in-distribution)
    codes = pq.encode(fit_vecs)
    decoded = pq.decode(codes)

    # Decoded vectors should be strictly closer than a random baseline.
    # For random Gaussian data with sub_dim=4, 16 centroids per subspace,
    # reconstruction error should be clearly less than naive mean-prediction error.
    naive_error = fit_vecs.norm(dim=1).mean().item()  # predict zero vector baseline
    actual_error = (fit_vecs - decoded).norm(dim=1).mean().item()
    assert actual_error < naive_error, (
        f"PQ reconstruction ({actual_error:.4f}) should beat naive baseline ({naive_error:.4f})"
    )


# ---------------------------------------------------------------------------
# 10. test_pq_asymmetric_distance_shape
# ---------------------------------------------------------------------------

def test_pq_asymmetric_distance_shape():
    torch.manual_seed(7)
    cfg = small_cfg()
    pq = ProductQuantizer(cfg)
    fit_vecs = torch.randn(20, EMBED_DIM)
    pq.fit(fit_vecs)

    test_vecs = torch.randn(5, EMBED_DIM)
    codes = pq.encode(test_vecs)

    query = torch.randn(EMBED_DIM)
    dists = pq.asymmetric_distance(query, codes)
    assert dists.shape == (5,)


# ---------------------------------------------------------------------------
# 11. test_ivf_build_and_len
# ---------------------------------------------------------------------------

def test_ivf_build_and_len():
    torch.manual_seed(8)
    cfg = small_cfg()
    ivf = IVFIndex(cfg)
    vectors = torch.randn(32, EMBED_DIM)
    ivf.build(vectors)
    assert len(ivf) == 32


# ---------------------------------------------------------------------------
# 12. test_ivf_search_returns_top_k
# ---------------------------------------------------------------------------

def test_ivf_search_returns_top_k():
    torch.manual_seed(9)
    cfg = small_cfg()
    ivf = IVFIndex(cfg)
    vectors = torch.randn(32, EMBED_DIM)
    ivf.build(vectors)

    query = torch.randn(EMBED_DIM)
    scores, ids = ivf.search(query, top_k=3)
    assert scores.shape == (3,)
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 13. test_dense_retriever_exact_search_correct
# ---------------------------------------------------------------------------

def test_dense_retriever_exact_search_correct():
    torch.manual_seed(10)
    cfg = small_cfg()
    retriever = DenseRetriever(cfg)

    corpus = torch.randn(20, EMBED_DIM)
    retriever.index(corpus)

    # Query is exactly corpus[5]
    query = corpus[5].clone()
    results = retriever.exact_search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == 5, f"Expected id=5, got id={results[0]['id']}"


# ---------------------------------------------------------------------------
# 14. test_dense_retriever_search_result_keys
# ---------------------------------------------------------------------------

def test_dense_retriever_search_result_keys():
    torch.manual_seed(11)
    cfg = small_cfg()
    retriever = DenseRetriever(cfg)

    corpus = torch.randn(20, EMBED_DIM)
    retriever.index(corpus)

    query = torch.randn(EMBED_DIM)
    results = retriever.search(query, top_k=3)

    assert len(results) > 0
    for r in results:
        assert "score" in r
        assert "id" in r
        # "text" key should be present (may be None)
        assert "text" in r


# ---------------------------------------------------------------------------
# 15. test_dense_retriever_evaluate_recall_range
# ---------------------------------------------------------------------------

def test_dense_retriever_evaluate_recall_range():
    torch.manual_seed(12)
    cfg = small_cfg()
    retriever = DenseRetriever(cfg)

    n_corpus = 32
    n_queries = 5
    corpus = torch.randn(n_corpus, EMBED_DIM)
    retriever.index(corpus)

    # Ground truth: each query is nearest to the corresponding corpus vector
    query_vectors = corpus[:n_queries].clone()
    ground_truth_ids = list(range(n_queries))

    recall = retriever.evaluate_recall(query_vectors, ground_truth_ids, top_k=10)
    assert 0.0 <= recall <= 1.0, f"Recall out of [0, 1]: {recall}"
