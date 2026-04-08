"""Tests for src/training/hard_negative.py — Hard Negative Mining."""

from __future__ import annotations

import pytest
import torch

from src.training.hard_negative import (
    HardNegativeMiner,
    PreferenceHardNegative,
    TripletLossWithMining,
)

# ---------------------------------------------------------------------------
# Shared helpers / constants
# ---------------------------------------------------------------------------

torch.manual_seed(42)

D = 16   # embedding dimension
N = 12   # total embeddings per batch


def _make_embeddings(n: int = N, d: int = D, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _make_labels(n: int = N, n_classes: int = 3) -> torch.Tensor:
    """Balanced labels: each class has n // n_classes samples."""
    return torch.arange(n) % n_classes


# ---------------------------------------------------------------------------
# 1. test_compute_cosine_distances_range
# ---------------------------------------------------------------------------

def test_compute_cosine_distances_range():
    """Cosine distances must lie in [0, 2]."""
    miner = HardNegativeMiner(distance_metric='cosine')
    emb = _make_embeddings()
    dist = miner.compute_distances(emb)

    assert dist.shape == (N, N)
    assert (dist >= 0).all(), "Cosine distances must be >= 0"
    assert (dist <= 2 + 1e-5).all(), "Cosine distances must be <= 2"


# ---------------------------------------------------------------------------
# 2. test_compute_euclidean_distances_positive
# ---------------------------------------------------------------------------

def test_compute_euclidean_distances_positive():
    """Euclidean distances must be >= 0."""
    miner = HardNegativeMiner(distance_metric='euclidean')
    emb = _make_embeddings()
    dist = miner.compute_distances(emb)

    assert dist.shape == (N, N)
    assert (dist >= 0).all(), "Euclidean distances must be non-negative"
    # Diagonal should be (approximately) zero — allow for float32 round-trip error
    assert dist.diagonal().abs().max().item() < 1e-2, "Self-distance must be near zero"


# ---------------------------------------------------------------------------
# 3. test_mine_hardest_returns_negatives
# ---------------------------------------------------------------------------

def test_mine_hardest_returns_negatives():
    """Hardest-mined indices must have a different class than their anchors."""
    miner = HardNegativeMiner(mining_strategy='hardest', distance_metric='cosine')
    emb = _make_embeddings()
    labels = _make_labels()
    dist = miner.compute_distances(emb)

    B = 4
    anchor_idx = torch.arange(B)
    # Positives: same class as anchor (wrap within class)
    positive_idx = (anchor_idx + 3) % N  # offset to a different idx

    neg_idx = miner.mine_hardest(anchor_idx, positive_idx, dist, labels)

    assert neg_idx.shape == (B,)
    for i in range(B):
        a = anchor_idx[i].item()
        n = neg_idx[i].item()
        assert labels[a] != labels[n], (
            f"Anchor {a} (label {labels[a]}) and negative {n} (label {labels[n]}) "
            f"must have different labels"
        )


# ---------------------------------------------------------------------------
# 4. test_mine_hardest_is_closest
# ---------------------------------------------------------------------------

def test_mine_hardest_is_closest():
    """Hardest negative must be the closest (smallest distance) among all valid negatives."""
    miner = HardNegativeMiner(mining_strategy='hardest', distance_metric='euclidean')
    emb = _make_embeddings()
    labels = _make_labels()
    dist = miner.compute_distances(emb)

    anchor_idx = torch.tensor([0, 3, 6])
    positive_idx = torch.tensor([1, 4, 7])  # same-class offsets

    neg_idx = miner.mine_hardest(anchor_idx, positive_idx, dist, labels)

    for i, a in enumerate(anchor_idx.tolist()):
        anchor_label = labels[a]
        n = neg_idx[i].item()

        # Distance of the returned negative
        d_returned = dist[a, n].item()

        # All valid negatives
        for j in range(N):
            if j == a:
                continue
            if labels[j] == anchor_label:
                continue
            # The returned negative must be at least as close as every other valid negative
            assert d_returned <= dist[a, j].item() + 1e-5, (
                f"Anchor {a}: returned negative {n} (d={d_returned:.4f}) is not "
                f"the closest; negative {j} has distance {dist[a, j].item():.4f}"
            )


# ---------------------------------------------------------------------------
# 5. test_mine_semi_hard_within_margin
# ---------------------------------------------------------------------------

def test_mine_semi_hard_within_margin():
    """
    When a semi-hard negative exists it must satisfy:
      d(a, n) > d(a, p)   and   d(a, n) < d(a, p) + margin
    """
    margin = 0.5
    miner = HardNegativeMiner(
        margin=margin,
        mining_strategy='semi_hard',
        distance_metric='cosine',
    )
    emb = _make_embeddings(n=20, seed=7)
    labels = _make_labels(n=20, n_classes=4)
    dist = miner.compute_distances(emb)

    # Build pairs with known positives (same class)
    anchor_idx_list, positive_idx_list = [], []
    for i in range(20):
        same = (labels == labels[i]).nonzero(as_tuple=False).squeeze(1)
        others = same[same != i]
        if others.numel() > 0:
            anchor_idx_list.append(i)
            positive_idx_list.append(others[0].item())

    anchor_idx = torch.tensor(anchor_idx_list)
    positive_idx = torch.tensor(positive_idx_list)

    neg_idx = miner.mine_semi_hard(anchor_idx, positive_idx, dist, labels)

    # For each pair where a semi-hard negative *does* exist, verify the constraint
    for i in range(len(anchor_idx_list)):
        a = anchor_idx[i].item()
        p = positive_idx[i].item()
        n = neg_idx[i].item()
        anchor_label = labels[a]
        d_ap = dist[a, p].item()

        # Check there exists at least one semi-hard candidate
        neg_mask = (labels != anchor_label)
        neg_mask_no_self = neg_mask.clone()
        neg_mask_no_self[a] = False
        d_row = dist[a]
        semi_exists = (
            neg_mask_no_self & (d_row > d_ap) & (d_row < d_ap + margin)
        ).any()

        if semi_exists:
            d_an = dist[a, n].item()
            assert d_an > d_ap - 1e-5, (
                f"Semi-hard negative must be farther than positive: "
                f"d(a,n)={d_an:.4f} <= d(a,p)={d_ap:.4f}"
            )
            assert d_an < d_ap + margin + 1e-5, (
                f"Semi-hard negative must be within margin: "
                f"d(a,n)={d_an:.4f} >= d(a,p)+margin={d_ap + margin:.4f}"
            )


# ---------------------------------------------------------------------------
# 6. test_mine_semi_hard_fallback
# ---------------------------------------------------------------------------

def test_mine_semi_hard_fallback():
    """
    When no semi-hard negative exists, mine_semi_hard must return the hardest
    negative (same result as mine_hardest for that anchor).
    """
    miner = HardNegativeMiner(
        margin=0.0,  # zero margin → semi-hard zone is empty
        mining_strategy='semi_hard',
        distance_metric='cosine',
    )
    emb = _make_embeddings()
    labels = _make_labels()
    dist = miner.compute_distances(emb)

    anchor_idx = torch.tensor([0, 3])
    positive_idx = torch.tensor([1, 4])

    neg_semi = miner.mine_semi_hard(anchor_idx, positive_idx, dist, labels)
    neg_hard = miner.mine_hardest(anchor_idx, positive_idx, dist, labels)

    # With margin=0, no sample can satisfy d(a,n) > d(a,p) AND < d(a,p)+0,
    # so fallback to hardest must match
    assert torch.equal(neg_semi, neg_hard), (
        f"Expected fallback to hardest, got semi={neg_semi} vs hard={neg_hard}"
    )


# ---------------------------------------------------------------------------
# 7. test_mine_distance_weighted_valid_indices
# ---------------------------------------------------------------------------

def test_mine_distance_weighted_valid_indices():
    """Distance-weighted sampling must return valid in-range indices of a different class."""
    miner = HardNegativeMiner(
        mining_strategy='distance_weighted',
        distance_metric='cosine',
    )
    emb = _make_embeddings()
    labels = _make_labels()
    dist = miner.compute_distances(emb)

    anchor_idx = torch.arange(N)
    positive_idx = (anchor_idx + 3) % N  # dummy positives (unused by this strategy)

    neg_idx = miner.mine_distance_weighted(anchor_idx, dist, labels)

    assert neg_idx.shape == (N,)
    for i in range(N):
        a = anchor_idx[i].item()
        n = neg_idx[i].item()
        assert 0 <= n < N, f"Index {n} out of range [0, {N})"
        assert n != a, f"Negative must not be the anchor itself"
        assert labels[n] != labels[a], (
            f"Anchor {a} (label {labels[a].item()}) and negative {n} "
            f"(label {labels[n].item()}) must have different labels"
        )


# ---------------------------------------------------------------------------
# 8. test_triplet_loss_scalar
# ---------------------------------------------------------------------------

def test_triplet_loss_scalar():
    """TripletLossWithMining.forward() must return a 0-d scalar tensor."""
    loss_fn = TripletLossWithMining(margin=0.2)
    emb = _make_embeddings()
    labels = _make_labels()

    loss = loss_fn(emb, labels)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 9. test_triplet_loss_positive
# ---------------------------------------------------------------------------

def test_triplet_loss_positive():
    """Triplet loss must be >= 0 (relu ensures non-negativity)."""
    loss_fn = TripletLossWithMining(margin=0.2)
    emb = _make_embeddings()
    labels = _make_labels()

    loss = loss_fn(emb, labels)
    assert loss.item() >= 0, f"Triplet loss must be non-negative, got {loss.item()}"


# ---------------------------------------------------------------------------
# 10. test_triplet_loss_zero_perfect
# ---------------------------------------------------------------------------

def test_triplet_loss_zero_perfect():
    """
    When the anchor equals the positive and the negative is far away,
    d(a,p)=0 and d(a,n) is large, so loss = max(0, 0 - large + margin) ≈ 0.
    """
    D_small = 8
    n_classes = 2
    # Each class: two identical vectors (so anchor==positive in cosine space)
    base_pos = torch.zeros(D_small)
    base_pos[0] = 1.0
    base_neg = torch.zeros(D_small)
    base_neg[-1] = 1.0  # orthogonal → cosine distance = 1.0

    # 4 samples: 2 from class 0 (identical to base_pos), 2 from class 1 (identical to base_neg)
    emb = torch.stack([base_pos, base_pos, base_neg, base_neg])
    labels = torch.tensor([0, 0, 1, 1])

    miner = HardNegativeMiner(margin=0.2, mining_strategy='hardest', distance_metric='cosine')
    loss_fn = TripletLossWithMining(margin=0.2, miner=miner)
    loss = loss_fn(emb, labels)

    # d(a,p) = 0 (identical), d(a,n) = 1.0 (orthogonal cosine distance)
    # loss = max(0, 0 - 1.0 + 0.2) = max(0, -0.8) = 0
    assert loss.item() < 1e-4, f"Expected near-zero loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 11. test_preference_hard_negative_shape
# ---------------------------------------------------------------------------

def test_preference_hard_negative_shape():
    """find_hard_rejected must return a (B, d) tensor."""
    B = 6
    phn = PreferenceHardNegative(similarity_threshold=0.5)
    chosen = _make_embeddings(n=B, d=D, seed=1)
    rejected = _make_embeddings(n=B, d=D, seed=2)

    hard_rej = phn.find_hard_rejected(chosen, rejected)
    assert hard_rej.shape == (B, D), (
        f"Expected shape ({B}, {D}), got {hard_rej.shape}"
    )


# ---------------------------------------------------------------------------
# 12. test_preference_hard_negative_indices
# ---------------------------------------------------------------------------

def test_preference_hard_negative_indices():
    """hard_negative_indices must return a valid (B,) int tensor in [0, B)."""
    B = 8
    phn = PreferenceHardNegative(similarity_threshold=0.5)
    chosen = _make_embeddings(n=B, d=D, seed=3)

    indices = phn.hard_negative_indices(chosen)

    assert indices.shape == (B,), f"Expected shape ({B},), got {indices.shape}"
    assert indices.dtype in (torch.int32, torch.int64), (
        f"Expected integer dtype, got {indices.dtype}"
    )
    assert (indices >= 0).all() and (indices < B).all(), (
        f"All indices must be in [0, {B}), got {indices}"
    )
