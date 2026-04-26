"""Tests for src/training/hard_negative_mining.py."""

from __future__ import annotations

import torch

from src.training.hard_negative_mining import (
    HardNegativeConfig,
    HardNegativeMiner,
    infonce_loss,
    mine_hard_negatives,
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    triplet_loss,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

B, D = 8, 16
N_CLASSES = 4  # 2 samples per class


def make_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (embeddings, labels) with B=8, D=16, 4 classes of 2 samples each."""
    torch.manual_seed(seed)
    embeddings = torch.randn(B, D)
    # Labels: [0, 0, 1, 1, 2, 2, 3, 3]
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    return embeddings, labels


# ---------------------------------------------------------------------------
# 1. HardNegativeConfig defaults
# ---------------------------------------------------------------------------


def test_hard_negative_config_defaults():
    cfg = HardNegativeConfig()
    assert cfg.mining_strategy == "semi-hard"
    assert cfg.margin == 0.5
    assert cfg.distance_metric == "cosine"
    assert cfg.n_hard_negatives == 1
    assert cfg.temperature == 0.07
    assert cfg.max_pairs == 1000


# ---------------------------------------------------------------------------
# 2. pairwise_cosine_similarity shape
# ---------------------------------------------------------------------------


def test_pairwise_cosine_similarity_shape():
    M, N = 5, 7
    x = torch.randn(M, D)
    y = torch.randn(N, D)
    out = pairwise_cosine_similarity(x, y)
    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. pairwise_cosine_similarity self-similarity ≈ 1 on L2-normalised inputs
# ---------------------------------------------------------------------------


def test_pairwise_cosine_similarity_self():
    x = torch.randn(B, D)
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    sim = pairwise_cosine_similarity(x_norm, x_norm)
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones(B), atol=1e-5), f"Diagonal should be ≈ 1, got {diag}"


# ---------------------------------------------------------------------------
# 4. pairwise_euclidean_distance shape
# ---------------------------------------------------------------------------


def test_pairwise_euclidean_distance_shape():
    M, N = 6, 4
    x = torch.randn(M, D)
    y = torch.randn(N, D)
    out = pairwise_euclidean_distance(x, y)
    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. pairwise_euclidean_distance self-distance = 0 on diagonal
# ---------------------------------------------------------------------------


def test_pairwise_euclidean_distance_self_zero():
    x = torch.randn(B, D)
    dist = pairwise_euclidean_distance(x, x)
    diag = dist.diagonal()
    assert torch.allclose(diag, torch.zeros(B), atol=1e-5), f"Diagonal should be 0, got {diag}"


# ---------------------------------------------------------------------------
# 6. mine_hard_negatives strategy="hard" returns non-empty triplets
# ---------------------------------------------------------------------------


def test_mine_hard_negatives_hard_nonempty():
    embeddings, labels = make_batch()
    cfg = HardNegativeConfig(mining_strategy="hard")
    anchors, positives, negatives = mine_hard_negatives(embeddings, labels, cfg)
    assert anchors.shape[0] > 0, "Expected non-empty triplets for strategy='hard'"
    assert positives.shape[0] == anchors.shape[0]
    assert negatives.shape[0] == anchors.shape[0]


# ---------------------------------------------------------------------------
# 7. mine_hard_negatives strategy="semi-hard" returns valid shapes
# ---------------------------------------------------------------------------


def test_mine_hard_negatives_semi_hard_shapes():
    embeddings, labels = make_batch()
    cfg = HardNegativeConfig(mining_strategy="semi-hard")
    anchors, positives, negatives = mine_hard_negatives(embeddings, labels, cfg)
    N = anchors.shape[0]
    assert N > 0, "Expected at least one triplet"
    assert positives.shape == (N, D)
    assert negatives.shape == (N, D)


# ---------------------------------------------------------------------------
# 8. triplet_loss returns a scalar
# ---------------------------------------------------------------------------


def test_triplet_loss_scalar():
    embeddings, labels = make_batch()
    cfg = HardNegativeConfig(mining_strategy="hard")
    anchors, positives, negatives = mine_hard_negatives(embeddings, labels, cfg)
    loss = triplet_loss(anchors, positives, negatives)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 9. triplet_loss ≈ 0 when negatives are far, positives are close
# ---------------------------------------------------------------------------


def test_triplet_loss_zero_when_margin_satisfied():
    # Anchor and positives are nearly identical unit vectors (cosine dist ≈ 0).
    # Negatives are orthogonal to anchors (cosine dist = 1).
    # With margin=0.5: d(a,p) - d(a,n) + margin ≈ 0 - 1 + 0.5 = -0.5 → clamped to 0.
    torch.manual_seed(5)
    anchors = torch.randn(4, D)
    # Positives: almost identical direction
    positives = anchors + 1e-6 * torch.randn(4, D)
    # Negatives: build orthogonal complement for each anchor
    rand_vecs = torch.randn(4, D)
    a_norm = torch.nn.functional.normalize(anchors, p=2, dim=-1)
    proj = (rand_vecs * a_norm).sum(dim=-1, keepdim=True) * a_norm
    negatives = rand_vecs - proj  # orthogonal to anchor

    loss = triplet_loss(anchors, positives, negatives, margin=0.5, distance_metric="cosine")
    # d(a,p) ≈ 0, d(a,n) ≈ 1, margin = 0.5 → loss ≈ max(0 - 1 + 0.5, 0) = 0
    assert loss.item() < 1e-4, f"Expected ≈ 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 10. triplet_loss > 0 when negatives are closer than positives
# ---------------------------------------------------------------------------


def test_triplet_loss_positive_when_violation():
    # Anchor: random unit vectors
    torch.manual_seed(42)
    anchors = torch.randn(4, D)
    # Negatives: almost identical to anchors (very close in cosine space)
    negatives = anchors + 1e-4 * torch.randn(4, D)
    # Positives: orthogonal to anchors (far in cosine space)
    positives = torch.randn(4, D)

    loss = triplet_loss(anchors, positives, negatives, margin=0.5, distance_metric="cosine")
    assert loss.item() > 0.0, f"Expected loss > 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 11. infonce_loss returns a scalar
# ---------------------------------------------------------------------------


def test_infonce_loss_scalar():
    torch.manual_seed(7)
    K = 3
    query = torch.randn(B, D)
    positives = torch.randn(B, D)
    negatives = torch.randn(B, K, D)
    loss = infonce_loss(query, positives, negatives)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 12. infonce_loss decreases when positives are more similar
# ---------------------------------------------------------------------------


def test_infonce_loss_decreases_with_similar_positives():
    torch.manual_seed(11)
    K = 4
    query = torch.randn(B, D)
    negatives = torch.randn(B, K, D)

    # Case 1: positives are random (lower similarity)
    positives_random = torch.randn(B, D)
    loss_random = infonce_loss(query, positives_random, negatives).item()

    # Case 2: positives are nearly identical to queries (high similarity)
    positives_close = query + 1e-3 * torch.randn(B, D)
    loss_close = infonce_loss(query, positives_close, negatives).item()

    assert loss_close < loss_random, (
        f"Expected lower InfoNCE when positives are more similar "
        f"({loss_close:.4f} should be < {loss_random:.4f})"
    )


# ---------------------------------------------------------------------------
# 13. HardNegativeMiner.mine_and_compute_loss returns correct keys
# ---------------------------------------------------------------------------


def test_miner_mine_and_compute_loss_keys():
    embeddings, labels = make_batch()
    cfg = HardNegativeConfig()
    miner = HardNegativeMiner(cfg)
    loss, info = miner.mine_and_compute_loss(embeddings, labels)
    assert "n_triplets" in info, "Missing key 'n_triplets'"
    assert "mean_pos_dist" in info, "Missing key 'mean_pos_dist'"
    assert "mean_neg_dist" in info, "Missing key 'mean_neg_dist'"


# ---------------------------------------------------------------------------
# 14. HardNegativeMiner.mine_and_compute_loss n_triplets > 0 with multiple classes
# ---------------------------------------------------------------------------


def test_miner_mine_and_compute_loss_n_triplets():
    embeddings, labels = make_batch()
    cfg = HardNegativeConfig()
    miner = HardNegativeMiner(cfg)
    loss, info = miner.mine_and_compute_loss(embeddings, labels)
    assert info["n_triplets"] > 0, (
        f"Expected n_triplets > 0 for batch with {N_CLASSES} classes, got {info['n_triplets']}"
    )
    # Loss should be a valid scalar tensor
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
