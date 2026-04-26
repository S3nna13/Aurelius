"""Tests for src/alignment/preference_aggregation.py"""

from __future__ import annotations

from src.alignment.preference_aggregation import (
    AggregationConfig,
    PreferenceAggregator,
    annotator_agreement,
    borda_scores,
    bradley_terry_scores,
    majority_vote_scores,
    parse_ranking,
    ranking_to_pairwise,
)

# ---------------------------------------------------------------------------
# parse_ranking
# ---------------------------------------------------------------------------


def test_parse_ranking_simple():
    result = parse_ranking("B>A>C=D")
    assert result == [["B"], ["A"], ["C", "D"]]


def test_parse_ranking_single():
    result = parse_ranking("A")
    assert result == [["A"]]


def test_parse_ranking_empty():
    result = parse_ranking("")
    assert result == []


# ---------------------------------------------------------------------------
# ranking_to_pairwise
# ---------------------------------------------------------------------------


def test_ranking_to_pairwise_count():
    # "B>A>C" → 3 pairs: (B,A), (B,C), (A,C)
    ranking = [["B"], ["A"], ["C"]]
    pairs = ranking_to_pairwise(ranking)
    assert len(pairs) == 3
    winners = {p[0] for p in pairs}
    losers = {p[1] for p in pairs}
    assert "B" in winners
    assert "C" in losers


def test_ranking_to_pairwise_no_ties():
    # "A=B>C" — A and B are tied; only pairs involving C should appear
    ranking = [["A", "B"], ["C"]]
    pairs = ranking_to_pairwise(ranking)
    # Should have (A,C) and (B,C) — no (A,B) or (B,A) pair
    pair_keys = {(p[0], p[1]) for p in pairs}
    assert ("A", "B") not in pair_keys
    assert ("B", "A") not in pair_keys
    assert ("A", "C") in pair_keys
    assert ("B", "C") in pair_keys


# ---------------------------------------------------------------------------
# borda_scores
# ---------------------------------------------------------------------------


def test_borda_scores_ordering():
    # B is always ranked first → should have highest Borda score
    rankings = [
        [["B"], ["A"], ["C"]],
        [["B"], ["C"], ["A"]],
        [["B"], ["A"], ["C"]],
    ]
    candidates = ["A", "B", "C"]
    scores = borda_scores(rankings, candidates)
    assert scores["B"] > scores["A"]
    assert scores["B"] > scores["C"]


def test_borda_scores_normalized():
    rankings = [
        [["A"], ["B"], ["C"]],
    ]
    candidates = ["A", "B", "C"]
    scores = borda_scores(rankings, candidates)
    assert max(scores.values()) <= 1.0


def test_borda_scores_tied_average():
    # A=B both ranked first, C ranked last — A and B should have equal scores
    rankings = [
        [["A", "B"], ["C"]],
        [["A", "B"], ["C"]],
    ]
    candidates = ["A", "B", "C"]
    scores = borda_scores(rankings, candidates)
    assert abs(scores["A"] - scores["B"]) < 1e-9


# ---------------------------------------------------------------------------
# bradley_terry_scores
# ---------------------------------------------------------------------------


def test_bradley_terry_ordering():
    # A beats B and C convincingly
    pairwise_wins = {
        ("A", "B"): 5,
        ("A", "C"): 5,
        ("B", "C"): 3,
    }
    candidates = ["A", "B", "C"]
    scores = bradley_terry_scores(pairwise_wins, candidates)
    assert scores["A"] > scores["B"]
    assert scores["A"] > scores["C"]


def test_bradley_terry_sums_to_one():
    pairwise_wins = {
        ("A", "B"): 3,
        ("B", "C"): 2,
        ("A", "C"): 4,
    }
    candidates = ["A", "B", "C"]
    scores = bradley_terry_scores(pairwise_wins, candidates)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# majority_vote_scores
# ---------------------------------------------------------------------------


def test_majority_vote_clear_winner():
    # All annotators rank A first → A's win rate against others should be 1.0
    rankings = [
        [["A"], ["B"], ["C"]],
        [["A"], ["B"], ["C"]],
        [["A"], ["C"], ["B"]],
    ]
    candidates = ["A", "B", "C"]
    scores = majority_vote_scores(rankings, candidates)
    assert abs(scores["A"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# PreferenceAggregator
# ---------------------------------------------------------------------------


def test_preference_aggregator_borda():
    cfg = AggregationConfig(min_annotators=2, tie_threshold=0.05)
    agg = PreferenceAggregator(cfg)
    candidates = ["A", "B", "C"]
    rankings = [
        [["A"], ["B"], ["C"]],
        [["A"], ["C"], ["B"]],
        [["A"], ["B"], ["C"]],
    ]
    scores = agg.aggregate(candidates, rankings)
    result = agg.get_best_pair(candidates, scores)
    assert result is not None
    best, worst = result
    assert isinstance(best, str)
    assert isinstance(worst, str)
    assert best != worst


def test_preference_aggregator_no_pair_tied():
    # All candidates have identical rankings → scores differ, but let's force
    # all scores to be equal by using a single tied ranking
    cfg = AggregationConfig(min_annotators=1, tie_threshold=0.5)
    agg = PreferenceAggregator(cfg)
    candidates = ["A", "B"]
    # With a big tie_threshold, small score gaps should return None
    scores = {"A": 0.50, "B": 0.52}  # gap = 0.02, below threshold of 0.5
    result = agg.get_best_pair(candidates, scores)
    assert result is None


def test_to_dpo_pairs_returns_list():
    cfg = AggregationConfig(min_annotators=2, tie_threshold=0.05)
    agg = PreferenceAggregator(cfg)
    responses = {"A": "Response A text", "B": "Response B text", "C": "Response C text"}
    rankings = [
        [["A"], ["B"], ["C"]],
        [["A"], ["B"], ["C"]],
        [["A"], ["C"], ["B"]],
    ]
    result = agg.to_dpo_pairs(responses, rankings)
    assert isinstance(result, list)
    if result:
        chosen, rejected = result[0]
        assert isinstance(chosen, str)
        assert isinstance(rejected, str)


# ---------------------------------------------------------------------------
# annotator_agreement
# ---------------------------------------------------------------------------


def test_annotator_agreement_perfect():
    # All annotators give the same ranking → W should be close to 1.0
    rankings = [
        [["A"], ["B"], ["C"]],
        [["A"], ["B"], ["C"]],
        [["A"], ["B"], ["C"]],
    ]
    candidates = ["A", "B", "C"]
    w = annotator_agreement(rankings, candidates)
    assert abs(w - 1.0) < 1e-6


def test_annotator_agreement_range():
    rankings = [
        [["A"], ["B"], ["C"]],
        [["C"], ["B"], ["A"]],
        [["B"], ["A"], ["C"]],
    ]
    candidates = ["A", "B", "C"]
    w = annotator_agreement(rankings, candidates)
    assert 0.0 <= w <= 1.0
