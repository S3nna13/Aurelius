"""Tests for src/data/coreset_selection.py."""

import pytest
import torch

from src.data.coreset_selection import (
    CoresetConfig,
    CoresetSelector,
    greedy_coreset,
    kcenter_coreset,
    pairwise_cosine_sim,
    pairwise_l2,
    random_coreset,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N, D, K = 50, 16, 10  # dataset size, embedding dim, coreset size


@pytest.fixture
def embeddings():
    torch.manual_seed(0)
    return torch.randn(N, D)


# ---------------------------------------------------------------------------
# CoresetConfig
# ---------------------------------------------------------------------------


def test_coreset_config_defaults():
    cfg = CoresetConfig()
    assert cfg.n_select == 100
    assert cfg.method == "greedy"
    assert cfg.distance == "cosine"
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# pairwise_cosine_sim
# ---------------------------------------------------------------------------


def test_pairwise_cosine_sim_shape():
    M, N2 = 7, 11
    a = torch.randn(M, D)
    b = torch.randn(N2, D)
    out = pairwise_cosine_sim(a, b)
    assert out.shape == (M, N2)


def test_pairwise_cosine_sim_identical_vectors():
    v = torch.randn(1, D)
    sim = pairwise_cosine_sim(v, v)
    assert sim.shape == (1, 1)
    assert abs(sim.item() - 1.0) < 1e-5


def test_pairwise_cosine_sim_orthogonal_vectors():
    # two canonical basis vectors
    a = torch.zeros(1, D)
    b = torch.zeros(1, D)
    a[0, 0] = 1.0
    b[0, 1] = 1.0
    sim = pairwise_cosine_sim(a, b)
    assert abs(sim.item()) < 1e-6


# ---------------------------------------------------------------------------
# pairwise_l2
# ---------------------------------------------------------------------------


def test_pairwise_l2_shape():
    M, N2 = 5, 9
    a = torch.randn(M, D)
    b = torch.randn(N2, D)
    out = pairwise_l2(a, b)
    assert out.shape == (M, N2)


def test_pairwise_l2_identical_points():
    v = torch.randn(3, D)
    dist = pairwise_l2(v, v)
    # diagonal should be (near) zero — allow float32 rounding error
    diag = dist.diagonal()
    assert (diag < 1e-3).all()


# ---------------------------------------------------------------------------
# greedy_coreset
# ---------------------------------------------------------------------------


def test_greedy_coreset_returns_correct_shape(embeddings):
    idx = greedy_coreset(embeddings, K)
    assert idx.shape == (K,)


def test_greedy_coreset_indices_unique(embeddings):
    idx = greedy_coreset(embeddings, K)
    assert len(idx.tolist()) == len(set(idx.tolist()))


def test_greedy_coreset_indices_in_valid_range(embeddings):
    idx = greedy_coreset(embeddings, K)
    assert (idx >= 0).all() and (idx < N).all()


# ---------------------------------------------------------------------------
# kcenter_coreset
# ---------------------------------------------------------------------------


def test_kcenter_coreset_returns_correct_shape(embeddings):
    idx = kcenter_coreset(embeddings, K)
    assert idx.shape == (K,)


def test_kcenter_coreset_indices_unique(embeddings):
    idx = kcenter_coreset(embeddings, K)
    assert len(idx.tolist()) == len(set(idx.tolist()))


# ---------------------------------------------------------------------------
# random_coreset
# ---------------------------------------------------------------------------


def test_random_coreset_returns_unique_indices(embeddings):
    idx = random_coreset(embeddings, K, seed=42)
    assert len(idx.tolist()) == len(set(idx.tolist()))


def test_random_coreset_seed_reproducible(embeddings):
    idx1 = random_coreset(embeddings, K, seed=7)
    idx2 = random_coreset(embeddings, K, seed=7)
    assert (idx1 == idx2).all()


# ---------------------------------------------------------------------------
# CoresetSelector
# ---------------------------------------------------------------------------


def test_coreset_selector_select_returns_correct_count(embeddings):
    cfg = CoresetConfig(n_select=K, method="greedy")
    selector = CoresetSelector(cfg)
    idx = selector.select(embeddings)
    assert idx.shape == (K,)


def test_coreset_selector_get_subset_returns_correct_length(embeddings):
    cfg = CoresetConfig(n_select=K, method="random", seed=0)
    selector = CoresetSelector(cfg)
    data = list(range(N))
    subset = selector.get_subset(data, embeddings)
    assert len(subset) == K


def test_coreset_selector_kcenter_method(embeddings):
    cfg = CoresetConfig(n_select=K, method="kcenter")
    selector = CoresetSelector(cfg)
    idx = selector.select(embeddings)
    assert idx.shape == (K,)
    assert (idx >= 0).all() and (idx < N).all()


def test_coreset_selector_invalid_method_raises(embeddings):
    cfg = CoresetConfig(n_select=K, method="invalid")
    selector = CoresetSelector(cfg)
    with pytest.raises(ValueError, match="Unknown coreset method"):
        selector.select(embeddings)
