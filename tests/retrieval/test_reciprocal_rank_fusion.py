"""Unit tests for :mod:`src.retrieval.reciprocal_rank_fusion`."""

from __future__ import annotations

import math
import random
import time

import pytest

from src.retrieval.reciprocal_rank_fusion import (
    FUSION_REGISTRY,
    borda_count,
    comb_mnz,
    comb_sum,
    fuse,
    reciprocal_rank_fusion,
)


ATOL = 1e-6


# ---------------------------------------------------------------------------
# RRF: hand-computed
# ---------------------------------------------------------------------------


def test_rrf_two_list_formula_hand_computed():
    # list1: A, B, C   list2: B, A, D
    # RRF k=60: A = 1/61 + 1/62, B = 1/62 + 1/61, C = 1/63, D = 1/63
    rankings = [
        [("A", 0.9), ("B", 0.8), ("C", 0.7)],
        [("B", 0.95), ("A", 0.85), ("D", 0.5)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)

    expected_ab = 1 / 61 + 1 / 62
    expected_cd = 1 / 63

    # A and B tie; tie-break by doc_id ascending -> A before B.
    assert fused[0][0] == "A"
    assert fused[1][0] == "B"
    # C and D tie; C before D.
    assert fused[2][0] == "C"
    assert fused[3][0] == "D"

    assert math.isclose(fused[0][1], expected_ab, abs_tol=ATOL)
    assert math.isclose(fused[1][1], expected_ab, abs_tol=ATOL)
    assert math.isclose(fused[2][1], expected_cd, abs_tol=ATOL)
    assert math.isclose(fused[3][1], expected_cd, abs_tol=ATOL)


def test_rrf_three_list_formula_hand_computed():
    # list1: X,Y,Z   list2: Y,Z,X   list3: Z,X,Y  (every doc appears in all three)
    # With k=60:
    # X = 1/61 + 1/63 + 1/62
    # Y = 1/62 + 1/61 + 1/63
    # Z = 1/63 + 1/62 + 1/61
    # All equal -> tie-break by doc_id ascending: X, Y, Z
    rankings = [
        [("X", 1.0), ("Y", 0.5), ("Z", 0.1)],
        [("Y", 1.0), ("Z", 0.5), ("X", 0.1)],
        [("Z", 1.0), ("X", 0.5), ("Y", 0.1)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)
    expected = 1 / 61 + 1 / 62 + 1 / 63
    assert [d for d, _ in fused] == ["X", "Y", "Z"]
    for _, s in fused:
        assert math.isclose(s, expected, abs_tol=ATOL)


def test_rrf_weighted_formula():
    # Weights [2.0, 1.0]; k=60.
    # list1: A,B     list2: B,A
    # A = 2/61 + 1/62
    # B = 2/62 + 1/61
    # A > B because doubling the top-rank advantage of A dominates.
    rankings = [
        [("A", 0.0), ("B", 0.0)],
        [("B", 0.0), ("A", 0.0)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60, weights=[2.0, 1.0])
    exp_a = 2 / 61 + 1 / 62
    exp_b = 2 / 62 + 1 / 61
    assert fused[0][0] == "A"
    assert fused[1][0] == "B"
    assert math.isclose(fused[0][1], exp_a, abs_tol=ATOL)
    assert math.isclose(fused[1][1], exp_b, abs_tol=ATOL)


def test_rrf_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights length"):
        reciprocal_rank_fusion(
            [[("A", 0.0)], [("B", 0.0)]], weights=[1.0, 1.0, 1.0]
        )


def test_rrf_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion(
            [[("A", 0.0)], [("B", 0.0)]], weights=[1.0, -0.5]
        )


def test_rrf_bad_k_raises():
    with pytest.raises(ValueError, match="k must be a positive"):
        reciprocal_rank_fusion([[("A", 0.0)]], k=0)


# ---------------------------------------------------------------------------
# Borda
# ---------------------------------------------------------------------------


def test_borda_uniform_size_lists():
    # Two lists of length 3. Points = L - rank.
    # list1: A(2), B(1), C(0)
    # list2: B(2), C(1), A(0)
    # Totals: A=2, B=3, C=1
    rankings = [
        [("A", 9), ("B", 8), ("C", 7)],
        [("B", 9), ("C", 8), ("A", 7)],
    ]
    fused = borda_count(rankings)
    assert fused == [("B", 3.0), ("A", 2.0), ("C", 1.0)]


# ---------------------------------------------------------------------------
# CombSUM / CombMNZ
# ---------------------------------------------------------------------------


def test_comb_sum_minmax_normalization():
    # list1 scores {A:10, B:5, C:0} -> normalized {A:1, B:0.5, C:0}
    # list2 scores {B:4, D:2}       -> normalized {B:1, D:0}
    # CombSUM: A=1.0, B=1.5, C=0.0, D=0.0
    rankings = [
        [("A", 10.0), ("B", 5.0), ("C", 0.0)],
        [("B", 4.0), ("D", 2.0)],
    ]
    fused = comb_sum(rankings, normalize="minmax")
    d = dict(fused)
    assert math.isclose(d["A"], 1.0, abs_tol=ATOL)
    assert math.isclose(d["B"], 1.5, abs_tol=ATOL)
    assert math.isclose(d["C"], 0.0, abs_tol=ATOL)
    assert math.isclose(d["D"], 0.0, abs_tol=ATOL)
    # Top ranked is B.
    assert fused[0][0] == "B"


def test_comb_mnz_equals_combsum_times_match_count():
    rankings = [
        [("A", 10.0), ("B", 5.0), ("C", 0.0)],
        [("B", 4.0), ("D", 2.0)],
    ]
    cs = dict(comb_sum(rankings, normalize="minmax"))
    mnz = dict(comb_mnz(rankings, normalize="minmax"))
    # Match counts: A=1, B=2, C=1, D=1
    assert math.isclose(mnz["A"], cs["A"] * 1, abs_tol=ATOL)
    assert math.isclose(mnz["B"], cs["B"] * 2, abs_tol=ATOL)
    assert math.isclose(mnz["C"], cs["C"] * 1, abs_tol=ATOL)
    assert math.isclose(mnz["D"], cs["D"] * 1, abs_tol=ATOL)


def test_comb_sum_invalid_normalize_raises():
    with pytest.raises(ValueError, match="unknown normalize"):
        comb_sum([[("A", 1.0)]], normalize="zscore")


# ---------------------------------------------------------------------------
# fuse dispatcher
# ---------------------------------------------------------------------------


def test_fuse_dispatcher_routes_correctly():
    rankings = [
        [("A", 0.0), ("B", 0.0)],
        [("B", 0.0), ("A", 0.0)],
    ]
    assert fuse("rrf", rankings) == reciprocal_rank_fusion(rankings)
    assert fuse("borda", rankings) == borda_count(rankings)
    assert fuse("combsum", rankings, normalize="none") == comb_sum(
        rankings, normalize="none"
    )
    assert fuse("combmnz", rankings, normalize="none") == comb_mnz(
        rankings, normalize="none"
    )


def test_fuse_invalid_method_raises():
    with pytest.raises(ValueError, match="unknown fusion method"):
        fuse("condorcet", [[("A", 0.0)]])


def test_fusion_registry_contents():
    assert set(FUSION_REGISTRY) == {"rrf", "borda", "combsum", "combmnz"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_rankings_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert borda_count([]) == []
    assert comb_sum([]) == []
    assert comb_mnz([]) == []


def test_single_ranking_preserves_order_rrf():
    ranking = [("a", 9.0), ("b", 7.0), ("c", 5.0), ("d", 3.0)]
    fused = reciprocal_rank_fusion([ranking], k=60)
    assert [d for d, _ in fused] == ["a", "b", "c", "d"]
    # Scores are 1/61, 1/62, 1/63, 1/64.
    for i, (_, s) in enumerate(fused):
        assert math.isclose(s, 1 / (60 + i + 1), abs_tol=ATOL)


def test_duplicate_doc_ids_within_single_ranking_raises():
    with pytest.raises(ValueError, match="duplicate doc_id"):
        reciprocal_rank_fusion([[("A", 1.0), ("A", 0.5)]])


def test_top_n_truncates_correctly():
    rankings = [
        [("A", 0.0), ("B", 0.0), ("C", 0.0), ("D", 0.0), ("E", 0.0)],
        [("B", 0.0), ("A", 0.0), ("C", 0.0), ("E", 0.0), ("D", 0.0)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60, top_n=2)
    assert len(fused) == 2
    assert [d for d, _ in fused] == ["A", "B"]


def test_top_n_invalid_raises():
    with pytest.raises(ValueError, match="top_n"):
        reciprocal_rank_fusion([[("A", 0.0)]], top_n=0)


def test_determinism_across_runs():
    rng = random.Random(1234)
    rankings = []
    for _ in range(4):
        ids = [f"d{j:04d}" for j in range(50)]
        rng.shuffle(ids)
        rankings.append([(d, rng.random()) for d in ids])

    first = reciprocal_rank_fusion(rankings, k=60)
    for _ in range(5):
        again = reciprocal_rank_fusion(rankings, k=60)
        assert again == first


def test_tie_break_by_doc_id_ascending():
    # Two completely symmetric single-doc lists with distinct doc ids all at rank 1.
    rankings = [
        [("zebra", 1.0)],
        [("apple", 1.0)],
        [("mango", 1.0)],
    ]
    fused = reciprocal_rank_fusion(rankings, k=60)
    # All tie at 1/61. Alphabetical.
    assert [d for d, _ in fused] == ["apple", "mango", "zebra"]


def test_large_10_rankings_1000_docs_under_1_second():
    rng = random.Random(42)
    N = 1000
    rankings = []
    for _ in range(10):
        ids = [f"doc{j:05d}" for j in range(N)]
        rng.shuffle(ids)
        rankings.append([(d, rng.random()) for d in ids])

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(rankings, k=60)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"RRF fusion took {elapsed:.3f}s, expected <1s"
    assert len(fused) == N  # every doc appears in every list
