"""Unit tests for MT-Bench LLM-as-judge harness."""

from __future__ import annotations

import pytest

from src.eval.mtbench_judge import (
    MTBenchJudge,
    MTBenchQuestion,
    PairwiseResult,
    SingleAnswerScore,
)

# --------------------------------------------------------------------------- helpers


def _q(turns=None, ref=None, qid="q1", cat="writing"):
    return MTBenchQuestion(
        question_id=qid,
        category=cat,
        turns=turns or ["What is 2+2?"],
        reference=ref,
    )


def _const(reply: str):
    return lambda prompt: reply


# --------------------------------------------------------------------------- tests


def test_score_single_parses_bracket_rating():
    judge = MTBenchJudge(_const("Good answer. Rating: [[7]]"))
    r = judge.score_single(_q(), "4")
    assert isinstance(r, SingleAnswerScore)
    assert r.score == 7.0
    assert r.question_id == "q1"
    assert "Good answer" in r.judge_reasoning


def test_score_single_parses_rating_fallback_style():
    judge = MTBenchJudge(_const("The reasoning is solid.\nRating: 8"))
    r = judge.score_single(_q(), "some answer")
    assert r.score == 8.0


def test_score_single_malformed_output_returns_none():
    judge = MTBenchJudge(_const("I refuse to give a number."))
    r = judge.score_single(_q(), "x")
    assert r.score is None
    assert r.judge_output == "I refuse to give a number."


def test_score_single_zero_not_conflated_with_none():
    # 0 is out of [1, 10] so must reject -> None (not 0.0).
    judge_zero = MTBenchJudge(_const("Rating: [[0]]"))
    r0 = judge_zero.score_single(_q(), "x")
    assert r0.score is None

    # But a real valid 1.0 must survive and not be treated as falsy "missing".
    judge_one = MTBenchJudge(_const("Rating: [[1]]"))
    r1 = judge_one.score_single(_q(), "x")
    assert r1.score == 1.0
    assert r1.score is not None


def test_score_single_out_of_range_rejected():
    for bad in ("[[0]]", "[[11]]", "[[100]]", "[[-3]]"):
        judge = MTBenchJudge(_const(f"text {bad}"))
        assert judge.score_single(_q(), "x").score is None


def test_score_pairwise_A_B_tie_invalid():
    q = _q()
    for out, expected in [
        ("I prefer A. [[A]]", "A"),
        ("B wins clearly. [[B]]", "B"),
        ("Both equal. [[C]]", "tie"),
        ("tie verdict [[tie]]", "tie"),
        ("no verdict here", "invalid"),
        ("[[X]] weird", "invalid"),
    ]:
        judge = MTBenchJudge(_const(out))
        r = judge.score_pairwise(q, "ans A", "ans B")
        assert isinstance(r, PairwiseResult)
        assert r.winner == expected, (out, r.winner, expected)


def test_multi_turn_formats_all_turns_in_prompt():
    captured = {}

    def judge_fn(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Rating: [[6]]"

    q = _q(turns=["Turn one question.", "Turn two follow-up.", "Turn three."])
    judge = MTBenchJudge(judge_fn)
    judge.score_single(q, "answer blob")
    p = captured["prompt"]
    assert "Turn one question." in p
    assert "Turn two follow-up." in p
    assert "Turn three." in p
    assert "User Turn 1" in p and "User Turn 2" in p and "User Turn 3" in p


def test_reference_answer_included_when_provided():
    captured = {}

    def judge_fn(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Rating: [[5]]"

    q = _q(ref="The answer is 4.")
    judge = MTBenchJudge(judge_fn)
    judge.score_single(q, "4")
    assert "Reference Answer" in captured["prompt"]
    assert "The answer is 4." in captured["prompt"]

    # And omitted when reference is None.
    captured.clear()
    q2 = _q(ref=None)
    judge.score_single(q2, "4")
    assert "Reference Answer" not in captured["prompt"]


def test_aggregate_single_mean_median_valid():
    rs = [
        SingleAnswerScore("a", 8.0, "", ""),
        SingleAnswerScore("b", 6.0, "", ""),
        SingleAnswerScore("c", 10.0, "", ""),
        SingleAnswerScore("d", None, "", ""),  # invalid, excluded
    ]
    agg = MTBenchJudge.aggregate_single(rs)
    assert agg["n_valid"] == 3
    assert agg["n_total"] == 4
    assert agg["mean"] == pytest.approx(8.0)
    assert agg["median"] == pytest.approx(8.0)


def test_aggregate_pairwise_rates_sum_to_one():
    rs = [
        PairwiseResult("a", "A", ""),
        PairwiseResult("b", "A", ""),
        PairwiseResult("c", "B", ""),
        PairwiseResult("d", "tie", ""),
        PairwiseResult("e", "invalid", ""),  # excluded from rates
    ]
    agg = MTBenchJudge.aggregate_pairwise(rs)
    assert agg["n_valid"] == 4
    assert agg["n_total"] == 5
    total = agg["win_rate_a"] + agg["win_rate_b"] + agg["tie_rate"]
    assert total == pytest.approx(1.0)
    assert agg["win_rate_a"] == pytest.approx(0.5)
    assert agg["win_rate_b"] == pytest.approx(0.25)
    assert agg["tie_rate"] == pytest.approx(0.25)


def test_aggregate_empty_returns_zeros():
    a = MTBenchJudge.aggregate_single([])
    assert a == {"mean": 0.0, "median": 0.0, "n_valid": 0, "n_total": 0}
    b = MTBenchJudge.aggregate_pairwise([])
    assert b["win_rate_a"] == 0.0
    assert b["win_rate_b"] == 0.0
    assert b["tie_rate"] == 0.0
    assert b["n_valid"] == 0


def test_aggregate_all_invalid_returns_zeros():
    rs_s = [SingleAnswerScore("a", None, "", ""), SingleAnswerScore("b", None, "", "")]
    a = MTBenchJudge.aggregate_single(rs_s)
    assert a["n_valid"] == 0 and a["mean"] == 0.0

    rs_p = [PairwiseResult("a", "invalid", ""), PairwiseResult("b", "invalid", "")]
    b = MTBenchJudge.aggregate_pairwise(rs_p)
    assert b["n_valid"] == 0 and b["win_rate_a"] == 0.0


def test_determinism_same_judge_same_result():
    judge = MTBenchJudge(_const("Some reasoning. [[7]]"))
    q = _q()
    r1 = judge.score_single(q, "ans")
    r2 = judge.score_single(q, "ans")
    assert r1.score == r2.score
    assert r1.judge_output == r2.judge_output
    assert r1.judge_reasoning == r2.judge_reasoning

    judge_p = MTBenchJudge(_const("A is better. [[A]]"))
    p1 = judge_p.score_pairwise(q, "a", "b")
    p2 = judge_p.score_pairwise(q, "a", "b")
    assert p1.winner == p2.winner == "A"


def test_judge_fn_raising_is_caught_single():
    def boom(prompt):
        raise RuntimeError("judge died")

    judge = MTBenchJudge(boom)
    r = judge.score_single(_q(), "x")
    assert r.score is None
    assert "RuntimeError" in r.judge_reasoning


def test_judge_fn_raising_is_caught_pairwise():
    def boom(prompt):
        raise ValueError("nope")

    judge = MTBenchJudge(boom)
    r = judge.score_pairwise(_q(), "a", "b")
    assert r.winner == "invalid"
    assert "ValueError" in r.judge_output


def test_non_callable_judge_rejected():
    with pytest.raises(TypeError):
        MTBenchJudge("not a function")  # type: ignore[arg-type]


def test_fractional_scores_accepted():
    judge = MTBenchJudge(_const("Reasoning [[7.5]]"))
    r = judge.score_single(_q(), "x")
    assert r.score == pytest.approx(7.5)
