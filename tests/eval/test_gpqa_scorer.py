"""Unit tests for GPQA scoring harness."""

from __future__ import annotations

import pytest

from src.eval.gpqa_scorer import (
    GPQAProblem,
    GPQAResult,
    GPQAScorer,
    format_prompt,
    parse_answer_letter,
)


def _mkprob(
    qid: str = "q1",
    correct_index: int = 0,
    domain: str = "biology",
    difficulty: str = "hard",
) -> GPQAProblem:
    return GPQAProblem(
        question_id=qid,
        question="What?",
        choices=["alpha", "beta", "gamma", "delta"],
        correct_index=correct_index,
        domain=domain,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# parse_answer_letter
# ---------------------------------------------------------------------------


def test_parse_double_bracket_A():
    assert parse_answer_letter("[[A]]") == "A"


def test_parse_answer_colon_B():
    assert parse_answer_letter("I think Answer: B because reasons") == "B"


def test_parse_paren_C():
    assert parse_answer_letter("The best choice is (C).") == "C"


def test_parse_final_answer_is_lowercase_d():
    assert parse_answer_letter("After careful thought the final answer is d") == "D"


def test_parse_nothing_returns_none():
    assert parse_answer_letter("no valid letter here 12345") is None
    assert parse_answer_letter("") is None


def test_parse_case_insensitive_double_bracket():
    assert parse_answer_letter("[[b]]") == "B"


# ---------------------------------------------------------------------------
# score_one
# ---------------------------------------------------------------------------


def test_score_one_correct_letter():
    p = _mkprob(correct_index=0)  # gold A
    s = GPQAScorer()
    res = s.score_one(p, "Answer: A")
    assert isinstance(res, GPQAResult)
    assert res.predicted_letter == "A"
    assert res.correct is True


def test_score_one_wrong_letter():
    p = _mkprob(correct_index=2)  # gold C
    s = GPQAScorer()
    res = s.score_one(p, "[[A]]")
    assert res.predicted_letter == "A"
    assert res.correct is False


def test_score_one_unparseable_is_incorrect():
    p = _mkprob(correct_index=0)
    s = GPQAScorer()
    res = s.score_one(p, "zzzzz")
    assert res.predicted_letter is None
    assert res.correct is False


# ---------------------------------------------------------------------------
# score (aggregation)
# ---------------------------------------------------------------------------


def test_score_overall_accuracy():
    probs = [
        _mkprob("q1", 0, "biology"),
        _mkprob("q2", 1, "biology"),
        _mkprob("q3", 2, "chemistry"),
        _mkprob("q4", 3, "physics"),
    ]
    responses = ["Answer: A", "Answer: B", "Answer: A", "Answer: D"]
    s = GPQAScorer()
    agg = s.score(probs, responses)
    assert agg["n_total"] == 4
    assert agg["n_valid"] == 4
    # 3 out of 4 correct
    assert agg["overall_accuracy"] == pytest.approx(0.75)


def test_score_per_domain():
    probs = [
        _mkprob("q1", 0, "biology"),
        _mkprob("q2", 1, "biology"),
        _mkprob("q3", 2, "chemistry"),
    ]
    responses = ["Answer: A", "Answer: A", "Answer: C"]
    agg = GPQAScorer().score(probs, responses)
    pd = agg["per_domain"]
    assert set(pd) == {"biology", "chemistry"}
    assert pd["biology"]["n"] == 2
    assert pd["biology"]["correct"] == 1
    assert pd["biology"]["accuracy"] == pytest.approx(0.5)
    assert pd["chemistry"]["accuracy"] == pytest.approx(1.0)


def test_score_per_difficulty():
    probs = [
        _mkprob("q1", 0, "biology", difficulty="hard"),
        _mkprob("q2", 1, "biology", difficulty="easy"),
    ]
    responses = ["Answer: A", "Answer: A"]
    agg = GPQAScorer().score(probs, responses)
    pdiff = agg["per_difficulty"]
    assert pdiff["hard"]["accuracy"] == pytest.approx(1.0)
    assert pdiff["easy"]["accuracy"] == pytest.approx(0.0)


def test_score_empty_returns_zeros():
    agg = GPQAScorer().score([], [])
    assert agg["overall_accuracy"] == 0.0
    assert agg["n_valid"] == 0
    assert agg["n_total"] == 0
    assert agg["per_domain"] == {}
    assert agg["per_difficulty"] == {}


def test_score_length_mismatch_raises():
    with pytest.raises(ValueError):
        GPQAScorer().score([_mkprob()], ["A", "B"])


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_calls_generate_fn_for_each_problem():
    calls = []

    def gen(prompt: str) -> str:
        calls.append(prompt)
        return "Answer: A"

    probs = [_mkprob("q1", 0), _mkprob("q2", 0), _mkprob("q3", 1)]
    s = GPQAScorer(generate_fn=gen)
    results = s.run(probs)
    assert len(calls) == 3
    assert len(results) == 3
    # first two correct (gold A), third wrong (gold B, predicted A)
    assert [r.correct for r in results] == [True, True, False]


def test_run_requires_generate_fn():
    with pytest.raises(ValueError):
        GPQAScorer().run([_mkprob()])


def test_determinism():
    probs = [_mkprob("q1", 0), _mkprob("q2", 2, "chemistry")]
    responses = ["[[A]]", "[[A]]"]
    a = GPQAScorer().score(probs, responses)
    b = GPQAScorer().score(probs, responses)
    assert a == b


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_prompt_contains_all_four_choices():
    p = GPQAProblem(
        question_id="q",
        question="Which planet is largest?",
        choices=["Earth", "Mars", "Jupiter", "Saturn"],
        correct_index=2,
    )
    prompt = format_prompt(p)
    assert "Which planet is largest?" in prompt
    assert "A) Earth" in prompt
    assert "B) Mars" in prompt
    assert "C) Jupiter" in prompt
    assert "D) Saturn" in prompt
    assert prompt.rstrip().endswith("Answer (just the letter):")


def test_problem_validates_choice_count():
    with pytest.raises(ValueError):
        GPQAProblem(
            question_id="x",
            question="q",
            choices=["a", "b", "c"],
            correct_index=0,
        )


def test_problem_validates_correct_index():
    with pytest.raises(ValueError):
        GPQAProblem(
            question_id="x",
            question="q",
            choices=["a", "b", "c", "d"],
            correct_index=7,
        )
