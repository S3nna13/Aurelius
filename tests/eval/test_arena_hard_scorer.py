"""Unit tests for the Arena-Hard scorer. Uses a fake ``judge_fn``; no network."""

from __future__ import annotations

import math

import pytest

from src.eval.arena_hard_scorer import (
    ArenaComparison,
    ArenaHardScorer,
    ArenaProblem,
)


# ---------------------------------------------------------------------------
# Helper fake judges
# ---------------------------------------------------------------------------


def _always(verdict: str):
    """Returns a judge_fn that always emits the given bracket verdict."""
    def fn(prompt: str) -> str:
        return f"explanation text. [[{verdict}]]"
    return fn


def _order_dependent():
    """First call says A wins; second (swapped) call also says A wins.

    Because model ordering is swapped between the two calls, 'A' in the two
    frames refers to different models -> disagreement -> tie.
    """
    state = {"n": 0}

    def fn(prompt: str) -> str:
        state["n"] += 1
        return "verdict [[A]]"

    return fn


def _dominant_judge(winning_model_marker: str):
    """Judge peeks at the answer text; any answer containing ``marker`` wins.

    The scorer formats prompts with the text "[The Start of Assistant A's
    Answer]\n<answer>\n[The End of Assistant A's Answer]" so we can recover
    which side contains the marker by looking at character positions.
    """

    def fn(prompt: str) -> str:
        a_start = prompt.index("[The Start of Assistant A's Answer]")
        a_end = prompt.index("[The End of Assistant A's Answer]")
        b_start = prompt.index("[The Start of Assistant B's Answer]")
        b_end = prompt.index("[The End of Assistant B's Answer]")
        a_text = prompt[a_start:a_end]
        b_text = prompt[b_start:b_end]
        a_has = winning_model_marker in a_text
        b_has = winning_model_marker in b_text
        if a_has and not b_has:
            return "reasoning [[A]]"
        if b_has and not a_has:
            return "reasoning [[B]]"
        return "reasoning [[C]]"

    return fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compare_returns_a_winner():
    scorer = ArenaHardScorer(judge_fn=_always("A"), swap_order=False)
    prob = ArenaProblem(prompt_id="p1", prompt="Solve it.")
    c = scorer.compare(prob, "ans1", "ans2", "m_alpha", "m_beta")
    assert isinstance(c, ArenaComparison)
    assert c.winner == "A"
    assert c.model_a == "m_alpha"
    assert c.model_b == "m_beta"
    assert c.prompt_id == "p1"


def test_compare_returns_b_winner():
    scorer = ArenaHardScorer(judge_fn=_always("B"), swap_order=False)
    prob = ArenaProblem(prompt_id="p1", prompt="Solve it.")
    c = scorer.compare(prob, "a", "b", "x", "y")
    assert c.winner == "B"


def test_swap_order_disagreement_is_tie():
    # Judge always emits [[A]]. With swap, the first and second call refer
    # to different underlying models -> disagreement -> tie.
    scorer = ArenaHardScorer(judge_fn=_order_dependent(), swap_order=True)
    prob = ArenaProblem(prompt_id="p1", prompt="Explain BT.")
    c = scorer.compare(prob, "resp_x", "resp_y", "x", "y")
    assert c.winner == "tie"


def test_swap_order_agreement_preserved():
    # Dominant judge: model named by marker always wins regardless of
    # position. Should resolve to a consistent non-tie verdict.
    judge = _dominant_judge("MARKER_WINNER")
    scorer = ArenaHardScorer(judge_fn=judge, swap_order=True)
    prob = ArenaProblem(prompt_id="p1", prompt="question")
    c = scorer.compare(
        prob, "this is MARKER_WINNER content", "loser content", "m1", "m2"
    )
    assert c.winner == "A"  # model_a (m1) had the marker


def test_run_round_robin_generates_all_pairs():
    scorer = ArenaHardScorer(judge_fn=_always("A"), swap_order=False)
    problems = [
        ArenaProblem(prompt_id="p1", prompt="q1"),
        ArenaProblem(prompt_id="p2", prompt="q2"),
    ]
    responses = {
        "m1": ["a1", "a2"],
        "m2": ["b1", "b2"],
        "m3": ["c1", "c2"],
    }
    comps = scorer.run_round_robin(problems, responses)
    # C(3,2) pairs * 2 problems = 6
    assert len(comps) == 6
    pair_set = {(c.model_a, c.model_b) for c in comps}
    assert ("m1", "m2") in pair_set
    assert ("m1", "m3") in pair_set
    assert ("m2", "m3") in pair_set


def test_fit_bradley_terry_dominant_model_highest():
    # Construct comparisons where m_strong beats everyone.
    names = ["m_strong", "m_mid", "m_weak"]
    comparisons = []
    for _ in range(10):
        comparisons.append(
            ArenaComparison("p", "m_strong", "m_mid", "A")
        )
        comparisons.append(
            ArenaComparison("p", "m_strong", "m_weak", "A")
        )
        comparisons.append(
            ArenaComparison("p", "m_mid", "m_weak", "A")
        )
    ratings = ArenaHardScorer.fit_bradley_terry(comparisons, names, n_iters=200)
    assert ratings["m_strong"] > ratings["m_mid"] > ratings["m_weak"]
    # Zero-mean normalized.
    assert math.isclose(sum(ratings.values()), 0.0, abs_tol=1e-9)


def test_bootstrap_ci_contains_mean():
    names = ["m_strong", "m_weak"]
    comparisons = [
        ArenaComparison("p", "m_strong", "m_weak", "A") for _ in range(20)
    ]
    ci = ArenaHardScorer.bootstrap_confidence_intervals(
        comparisons, names, n_bootstrap=50, ci=0.90, seed=0
    )
    for m in names:
        mean, lo, hi = ci[m]
        assert lo <= mean <= hi


def test_bootstrap_lo_le_mean_le_hi_all_models():
    # Explicit test-12 condition.
    names = ["a", "b", "c"]
    comparisons = []
    for _ in range(5):
        comparisons.append(ArenaComparison("p", "a", "b", "A"))
        comparisons.append(ArenaComparison("p", "b", "c", "A"))
        comparisons.append(ArenaComparison("p", "a", "c", "A"))
    ci = ArenaHardScorer.bootstrap_confidence_intervals(
        comparisons, names, n_bootstrap=30, ci=0.95, seed=1
    )
    for m in names:
        mean, lo, hi = ci[m]
        assert lo <= mean <= hi


def test_invalid_judge_output_yields_invalid():
    def bad_judge(prompt: str) -> str:
        return "I refuse to answer."
    scorer = ArenaHardScorer(judge_fn=bad_judge, swap_order=False)
    prob = ArenaProblem(prompt_id="p1", prompt="q")
    c = scorer.compare(prob, "a", "b", "x", "y")
    assert c.winner == "invalid"


def test_empty_comparisons_returns_zero_ratings():
    names = ["m1", "m2"]
    ratings = ArenaHardScorer.fit_bradley_terry([], names, n_iters=50)
    assert ratings == {"m1": 0.0, "m2": 0.0}


def test_single_model_has_rating_zero():
    # Even with comparisons mentioning other models, a single listed name
    # gets rating 0.0 by the zero-mean normalization convention.
    names = ["only"]
    ratings = ArenaHardScorer.fit_bradley_terry([], names, n_iters=10)
    assert ratings == {"only": 0.0}
    # With comparisons only involving 'only' that cannot happen because we
    # require distinct model names -> just confirm trivial normalization.
    ratings2 = ArenaHardScorer.fit_bradley_terry(
        [ArenaComparison("p", "only", "other", "A")], ["only"], n_iters=10
    )
    assert ratings2 == {"only": 0.0}


def test_determinism_under_fixed_seed():
    names = ["a", "b", "c"]
    comparisons = []
    for _ in range(8):
        comparisons.append(ArenaComparison("p", "a", "b", "A"))
        comparisons.append(ArenaComparison("p", "b", "c", "A"))
        comparisons.append(ArenaComparison("p", "c", "a", "B"))
    ci1 = ArenaHardScorer.bootstrap_confidence_intervals(
        comparisons, names, n_bootstrap=40, ci=0.9, seed=42
    )
    ci2 = ArenaHardScorer.bootstrap_confidence_intervals(
        comparisons, names, n_bootstrap=40, ci=0.9, seed=42
    )
    assert ci1 == ci2


def test_malformed_model_names_raises():
    with pytest.raises(ValueError):
        ArenaHardScorer.fit_bradley_terry([], ["a", "a"], n_iters=5)
    with pytest.raises(ValueError):
        ArenaHardScorer.fit_bradley_terry([], ["a", ""], n_iters=5)


def test_missing_response_raises():
    scorer = ArenaHardScorer(judge_fn=_always("A"), swap_order=False)
    problems = [
        ArenaProblem(prompt_id="p1", prompt="q1"),
        ArenaProblem(prompt_id="p2", prompt="q2"),
    ]
    responses = {"m1": ["a1"], "m2": ["b1", "b2"]}  # m1 too short
    with pytest.raises(ValueError):
        scorer.run_round_robin(problems, responses)


def test_judge_fn_exceptions_handled():
    def raising_judge(prompt: str) -> str:
        raise RuntimeError("kaboom")
    scorer = ArenaHardScorer(judge_fn=raising_judge, swap_order=False)
    prob = ArenaProblem(prompt_id="p1", prompt="q")
    c = scorer.compare(prob, "a", "b", "x", "y")
    # Exception text doesn't contain [[A]]/[[B]]/[[C]], so it's invalid.
    assert c.winner == "invalid"


def test_compare_same_model_name_raises():
    scorer = ArenaHardScorer(judge_fn=_always("A"), swap_order=False)
    prob = ArenaProblem(prompt_id="p1", prompt="q")
    with pytest.raises(ValueError):
        scorer.compare(prob, "a", "b", "same", "same")


def test_bootstrap_empty_comparisons():
    names = ["a", "b"]
    ci = ArenaHardScorer.bootstrap_confidence_intervals(
        [], names, n_bootstrap=5, ci=0.9, seed=0
    )
    for m in names:
        mean, lo, hi = ci[m]
        assert mean == 0.0 and lo == 0.0 and hi == 0.0


def test_tie_contributes_half():
    # Exactly balanced tie data should yield equal ratings across 2 models.
    names = ["a", "b"]
    comps = [ArenaComparison("p", "a", "b", "tie") for _ in range(10)]
    ratings = ArenaHardScorer.fit_bradley_terry(comps, names, n_iters=100)
    assert math.isclose(ratings["a"], 0.0, abs_tol=1e-6)
    assert math.isclose(ratings["b"], 0.0, abs_tol=1e-6)
