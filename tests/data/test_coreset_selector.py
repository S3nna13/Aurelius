"""Tests for src/data/coreset_selector.py

Covers:
  - CoresetConfig defaults
  - k_center: budget size, uniqueness, index range, spread across clusters,
    cosine and euclidean distance variants
  - coverage: budget size, uniqueness, diversity preference
  - coverage_ratio: full-set → 1.0, subset → < 1.0
  - select() dispatcher: k_center and coverage paths
  - Integration: N=200, D=16, budget=20

Pure PyTorch only — no numpy, scipy, or sklearn.
"""

import pytest
import torch

from src.data.coreset_selector import CoresetConfig, CoresetSelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_embeddings(n: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def make_clustered_embeddings(d: int = 8, cluster_size: int = 20) -> torch.Tensor:
    """Return embeddings from two well-separated clusters.

    Cluster A is centred at  10 * e_0 and cluster B at -10 * e_0.
    With a large margin the greedy selector is guaranteed to pick from both.
    """
    torch.manual_seed(7)
    centre_a = torch.zeros(d)
    centre_a[0] = 10.0
    centre_b = torch.zeros(d)
    centre_b[0] = -10.0
    noise_a = torch.randn(cluster_size, d) * 0.1 + centre_a
    noise_b = torch.randn(cluster_size, d) * 0.1 + centre_b
    return torch.cat([noise_a, noise_b], dim=0)  # [2*cluster_size, d]


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CoresetConfig()
    assert cfg.budget == 1000
    assert cfg.method == "k_center"
    assert cfg.ngram_n == 2
    assert cfg.seed == 42
    assert cfg.distance_metric == "cosine"


# ---------------------------------------------------------------------------
# 2. k_center — returns exactly budget indices
# ---------------------------------------------------------------------------


def test_k_center_returns_budget():
    cfg = CoresetConfig(budget=10, method="k_center")
    sel = CoresetSelector(cfg)
    emb = make_embeddings(50, 8)
    result = sel.select_k_center(emb)
    assert len(result) == 10


# ---------------------------------------------------------------------------
# 3. k_center — no duplicate indices
# ---------------------------------------------------------------------------


def test_k_center_unique_indices():
    cfg = CoresetConfig(budget=15, method="k_center")
    sel = CoresetSelector(cfg)
    emb = make_embeddings(60, 8)
    result = sel.select_k_center(emb)
    assert len(result) == len(set(result)), "duplicate indices found"


# ---------------------------------------------------------------------------
# 4. k_center — all indices in [0, N)
# ---------------------------------------------------------------------------


def test_k_center_within_range():
    N, D, budget = 40, 6, 12
    cfg = CoresetConfig(budget=budget, method="k_center")
    sel = CoresetSelector(cfg)
    emb = make_embeddings(N, D)
    result = sel.select_k_center(emb)
    assert all(0 <= idx < N for idx in result)


# ---------------------------------------------------------------------------
# 5. k_center — spreads across clusters (cosine, default)
# ---------------------------------------------------------------------------


def test_k_center_spreads_well():
    cluster_size = 20
    emb = make_clustered_embeddings(d=8, cluster_size=cluster_size)
    # budget = 4 should be enough to touch both clusters
    cfg = CoresetConfig(budget=4, method="k_center", distance_metric="cosine")
    sel = CoresetSelector(cfg)
    result = sel.select_k_center(emb)

    in_a = sum(1 for i in result if i < cluster_size)
    in_b = sum(1 for i in result if i >= cluster_size)
    assert in_a >= 1, "k_center selected nothing from cluster A"
    assert in_b >= 1, "k_center selected nothing from cluster B"


# ---------------------------------------------------------------------------
# 6. k_center — cosine distance variant
# ---------------------------------------------------------------------------


def test_k_center_cosine():
    cfg = CoresetConfig(budget=8, method="k_center", distance_metric="cosine")
    sel = CoresetSelector(cfg)
    emb = make_embeddings(30, 4)
    result = sel.select_k_center(emb)
    assert len(result) == 8
    assert len(set(result)) == 8


# ---------------------------------------------------------------------------
# 7. k_center — euclidean distance variant
# ---------------------------------------------------------------------------


def test_k_center_euclidean():
    cfg = CoresetConfig(budget=8, method="k_center", distance_metric="euclidean")
    sel = CoresetSelector(cfg)
    emb = make_embeddings(30, 4)
    result = sel.select_k_center(emb)
    assert len(result) == 8
    assert len(set(result)) == 8


# ---------------------------------------------------------------------------
# 8. coverage — returns exactly budget indices
# ---------------------------------------------------------------------------


def test_coverage_returns_budget():
    seqs = [[1, 2, 3, 4, i] for i in range(30)]
    cfg = CoresetConfig(budget=10, method="coverage")
    sel = CoresetSelector(cfg)
    result = sel.select_coverage(seqs)
    assert len(result) == 10


# ---------------------------------------------------------------------------
# 9. coverage — no duplicates
# ---------------------------------------------------------------------------


def test_coverage_unique():
    seqs = [[i, i + 1, i + 2] for i in range(25)]
    cfg = CoresetConfig(budget=8, method="coverage")
    sel = CoresetSelector(cfg)
    result = sel.select_coverage(seqs)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# 10. coverage — prefers diverse sequences over duplicate ones
# ---------------------------------------------------------------------------


def test_coverage_diverse_seqs():
    # Sequences 0–4 are identical low-diversity sequences.
    # Sequences 5–9 each introduce unique bigrams.
    dup_seqs = [[10, 11, 12]] * 5
    diverse_seqs = [[i * 100, i * 100 + 1, i * 100 + 2] for i in range(5)]
    all_seqs = dup_seqs + diverse_seqs  # indices 0-4 duplicates, 5-9 diverse

    cfg = CoresetConfig(budget=5, method="coverage")
    sel = CoresetSelector(cfg)
    result = sel.select_coverage(all_seqs)

    # After picking one duplicate, the remaining four slots should come from
    # the diverse sequences (each adds more new bigrams).
    diverse_count = sum(1 for i in result if i >= 5)
    assert diverse_count >= 4, (
        f"Expected at least 4 diverse seqs selected, got {diverse_count}. Selected: {result}"
    )


# ---------------------------------------------------------------------------
# 11. coverage_ratio — selecting all → ratio == 1.0
# ---------------------------------------------------------------------------


def test_coverage_ratio_full():
    seqs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    cfg = CoresetConfig(budget=3, method="coverage")
    sel = CoresetSelector(cfg)
    result = sel.coverage_ratio([0, 1, 2], seqs)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 12. coverage_ratio — subset gives ratio < 1.0
# ---------------------------------------------------------------------------


def test_coverage_ratio_partial():
    seqs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    cfg = CoresetConfig(budget=2, method="coverage")
    sel = CoresetSelector(cfg)
    result = sel.coverage_ratio([0, 1], seqs)
    assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# 13. select() dispatches to k_center when method="k_center"
# ---------------------------------------------------------------------------


def test_select_dispatches_k_center():
    cfg = CoresetConfig(budget=5, method="k_center")
    sel = CoresetSelector(cfg)
    emb = make_embeddings(20, 4)
    result = sel.select(embeddings=emb)
    assert isinstance(result, list)
    assert len(result) == 5
    assert all(0 <= i < 20 for i in result)


# ---------------------------------------------------------------------------
# 14. select() dispatches to coverage when method="coverage"
# ---------------------------------------------------------------------------


def test_select_dispatches_coverage():
    seqs = [[i, i + 1, i + 2, i + 3] for i in range(20)]
    cfg = CoresetConfig(budget=5, method="coverage")
    sel = CoresetSelector(cfg)
    result = sel.select(token_sequences=seqs)
    assert isinstance(result, list)
    assert len(result) == 5
    assert len(set(result)) == 5


# ---------------------------------------------------------------------------
# 15. Integration — N=200, D=16, budget=20, k_center end-to-end
# ---------------------------------------------------------------------------


def test_integration_k_center():
    N, D, budget = 200, 16, 20
    torch.manual_seed(99)
    emb = torch.randn(N, D)

    cfg = CoresetConfig(budget=budget, method="k_center", distance_metric="cosine")
    sel = CoresetSelector(cfg)
    result = sel.select(embeddings=emb)

    # No duplicates
    assert len(result) == len(set(result)), "integration: duplicate indices"
    # Correct count
    assert len(result) == budget, f"integration: expected {budget}, got {len(result)}"
    # All within range
    assert all(0 <= i < N for i in result), "integration: index out of range"
    # Selected count is strictly less than full dataset
    assert len(result) < N
