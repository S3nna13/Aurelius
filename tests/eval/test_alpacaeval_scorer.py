"""Unit tests for AlpacaEvalScorer with fake judge_fn."""

from __future__ import annotations

import pytest

from src.eval.alpacaeval_scorer import (
    AlpacaComparison,
    AlpacaEvalScorer,
    AlpacaProblem,
)

# ---------------------------------------------------------------------------
# Fake judges
# ---------------------------------------------------------------------------


def _judge_always(verdict: str):
    def fn(prompt: str) -> str:
        return f"Explanation. [[{verdict}]]"

    return fn


def _judge_prefers_a_block_marker(marker: str = "[WIN]"):
    """Judge that returns [[A]] if marker present in A's block, else [[B]]."""

    def fn(prompt: str) -> str:
        a_idx = prompt.find("The Start of Assistant A")
        b_idx = prompt.find("The Start of Assistant B")
        a_block = prompt[a_idx:b_idx]
        b_block = prompt[b_idx:]
        a_hit = marker in a_block
        b_hit = marker in b_block
        if a_hit and not b_hit:
            return "reasoning [[A]]"
        if b_hit and not a_hit:
            return "reasoning [[B]]"
        return "reasoning [[C]]"

    return fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compare_candidate_wins_on_A():
    scorer = AlpacaEvalScorer(_judge_always("A"), swap_order=False)
    p = AlpacaProblem("q", "ref answer")
    c = scorer.compare(p, "cand answer")
    assert c.winner == "candidate"
    assert c.candidate_length == len("cand answer")
    assert c.reference_length == len("ref answer")


def test_compare_reference_wins_on_B():
    scorer = AlpacaEvalScorer(_judge_always("B"), swap_order=False)
    p = AlpacaProblem("q", "ref")
    c = scorer.compare(p, "cand")
    assert c.winner == "reference"


def test_swap_order_detects_disagreement_as_tie():
    # With [WIN] marker in candidate, first call (A=cand) -> A, second call
    # (A=ref, B=cand) should -> B. Both agree that candidate wins.
    scorer = AlpacaEvalScorer(_judge_prefers_a_block_marker(), swap_order=True)
    p = AlpacaProblem("q", "plain ref")
    c = scorer.compare(p, "cand with [WIN] here")
    assert c.winner == "candidate"

    # Position-biased judge: always prefers whatever is in slot A.
    scorer_biased = AlpacaEvalScorer(_judge_always("A"), swap_order=True)
    c2 = scorer_biased.compare(p, "cand")
    # First call: A=cand -> candidate. Second call: A=ref -> reference.
    # Disagreement -> tie.
    assert c2.winner == "tie"


def test_swap_order_false_trusts_first_call():
    scorer = AlpacaEvalScorer(_judge_always("A"), swap_order=False)
    p = AlpacaProblem("q", "ref")
    c = scorer.compare(p, "cand")
    assert c.winner == "candidate"


def test_malformed_judge_output_is_invalid():
    def judge(prompt: str) -> str:
        return "I have no opinion."

    scorer = AlpacaEvalScorer(judge, swap_order=False)
    p = AlpacaProblem("q", "ref")
    c = scorer.compare(p, "cand")
    assert c.winner == "invalid"


def test_score_aggregates_winrate_correctly():
    # [WIN] in candidate -> candidate wins under both orders.
    scorer = AlpacaEvalScorer(_judge_prefers_a_block_marker(), swap_order=True)
    problems = [AlpacaProblem(f"q{i}", "neutral ref text") for i in range(4)]
    candidates = [
        "cand [WIN] a",
        "cand [WIN] b",
        "cand [WIN] c",
        "plain cand",  # no win marker on either side -> C -> tie
    ]
    result = scorer.score(problems, candidates)
    assert result["n_valid"] == 4
    assert result["n_total"] == 4
    assert result["win_rate"] == 0.75
    assert result["tie_rate"] == 0.25
    assert result["reference_rate"] == 0.0


def test_length_controlled_winrate_differs_from_raw():
    # Candidate much longer than reference; candidate always wins.
    scorer = AlpacaEvalScorer(_judge_always("A"), swap_order=False)
    problems = [AlpacaProblem("q", "x")]  # ref_len = 1
    candidates = ["y" * 100]  # cand_len = 100 -> ratio dev = 99
    result = scorer.score(problems, candidates)
    assert result["win_rate"] == 1.0
    # LC = 1.0 - 0.1 * 99 = negative -> clamped to 0.0
    assert result["length_controlled_winrate"] < result["win_rate"]
    assert result["length_controlled_winrate"] == 0.0


def test_empty_results_returns_zeros_not_nan():
    scorer = AlpacaEvalScorer(_judge_always("A"))
    result = scorer.score([], [])
    assert result == {
        "win_rate": 0.0,
        "tie_rate": 0.0,
        "reference_rate": 0.0,
        "length_controlled_winrate": 0.0,
        "n_valid": 0,
        "n_total": 0,
    }


def test_judge_fn_raising_is_invalid():
    def judge(prompt: str) -> str:
        raise RuntimeError("boom")

    scorer = AlpacaEvalScorer(judge, swap_order=False)
    p = AlpacaProblem("q", "ref")
    c = scorer.compare(p, "cand")
    assert c.winner == "invalid"


def test_determinism():
    scorer = AlpacaEvalScorer(_judge_prefers_a_block_marker(), swap_order=True)
    p = AlpacaProblem("q", "neutral ref")
    cand = "cand [WIN] body"
    results = [scorer.compare(p, cand) for _ in range(5)]
    winners = {r.winner for r in results}
    assert winners == {"candidate"}
    # Lengths must be identical across repeated calls.
    assert len({r.candidate_length for r in results}) == 1
    assert len({r.reference_length for r in results}) == 1


def test_rates_sum_to_one_on_valid_subset():
    # Mix of outcomes: 2 candidate wins, 1 reference win, 1 tie, 1 invalid.
    calls = {"i": 0}

    def judge(prompt: str) -> str:
        i = calls["i"]
        calls["i"] += 1
        # swap_order=False so one call per comparison.
        return [
            "explanation [[A]]",  # candidate
            "explanation [[A]]",  # candidate
            "explanation [[B]]",  # reference
            "explanation [[C]]",  # tie
            "totally unparseable",  # invalid
        ][i]

    scorer = AlpacaEvalScorer(judge, swap_order=False)
    problems = [AlpacaProblem(f"q{i}", "ref") for i in range(5)]
    cands = [f"c{i}" for i in range(5)]
    res = scorer.score(problems, cands)
    assert res["n_valid"] == 4
    assert res["n_total"] == 5
    total = res["win_rate"] + res["tie_rate"] + res["reference_rate"]
    assert abs(total - 1.0) < 1e-9


def test_length_recorded_correctly():
    scorer = AlpacaEvalScorer(_judge_always("C"), swap_order=False)
    p = AlpacaProblem("instr", "reference of twenty-three chars")
    cand = "cand of sixteen!!"
    c = scorer.compare(p, cand)
    assert c.candidate_length == len(cand)
    assert c.reference_length == len(p.reference_output)
    assert isinstance(c, AlpacaComparison)


def test_n_valid_excludes_invalid():
    calls = {"i": 0}

    def judge(prompt: str) -> str:
        i = calls["i"]
        calls["i"] += 1
        return ["[[A]]", "not a verdict at all", "[[B]]"][i]

    scorer = AlpacaEvalScorer(judge, swap_order=False)
    problems = [AlpacaProblem(f"q{i}", "ref") for i in range(3)]
    cands = ["a", "b", "c"]
    res = scorer.score(problems, cands)
    assert res["n_total"] == 3
    assert res["n_valid"] == 2
    # With 1 candidate win and 1 reference win out of 2 valid:
    assert res["win_rate"] == 0.5
    assert res["reference_rate"] == 0.5
    assert res["tie_rate"] == 0.0


def test_swap_one_invalid_one_valid_becomes_tie():
    calls = {"i": 0}

    def judge(prompt: str) -> str:
        i = calls["i"]
        calls["i"] += 1
        return ["[[A]]", "unparseable"][i]

    scorer = AlpacaEvalScorer(judge, swap_order=True)
    p = AlpacaProblem("q", "ref")
    c = scorer.compare(p, "cand")
    # First call valid (candidate), second invalid -> tie (safe default).
    assert c.winner == "tie"


def test_score_length_mismatch_raises():
    scorer = AlpacaEvalScorer(_judge_always("A"))
    with pytest.raises(ValueError):
        scorer.score([AlpacaProblem("q", "r")], [])


def test_non_callable_judge_raises():
    with pytest.raises(TypeError):
        AlpacaEvalScorer(judge_fn="not callable")  # type: ignore[arg-type]
