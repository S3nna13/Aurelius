"""Unit tests for src/eval/mmlu_scorer.py."""

from __future__ import annotations

import pytest

from src.eval.mmlu_scorer import (
    CANONICAL_EXEMPLARS,
    MMLUProblem,
    MMLUResult,
    MMLUScorer,
    format_prompt,
    parse_answer_letter,
)


def _mk(qid: str = "q0", subject: str = "history", correct: int = 0) -> MMLUProblem:
    return MMLUProblem(
        question_id=qid,
        subject=subject,
        question=f"Question {qid}?",
        choices=["alpha", "beta", "gamma", "delta"],
        correct_index=correct,
    )


# -- parse_answer_letter ----------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ("(A)", "A"),
        ("Answer: B", "B"),
        ("[[C]]", "C"),
        ("D.", "D"),
        ("the answer is d", "D"),
        ("Final answer: (C)", "C"),
        ("A", "A"),
        ("The correct choice is (b).", "B"),
    ],
)
def test_parse_answer_letter_various(response, expected):
    assert parse_answer_letter(response) == expected


def test_parse_answer_letter_garbage_returns_none():
    assert parse_answer_letter("hello world no letters here") is None
    assert parse_answer_letter("") is None
    assert parse_answer_letter("Z is my answer") is None


def test_parse_answer_letter_non_string_raises():
    with pytest.raises(TypeError):
        parse_answer_letter(42)  # type: ignore[arg-type]


# -- score_one --------------------------------------------------------------


def test_score_one_correct():
    scorer = MMLUScorer(n_shots=0)
    prob = _mk(correct=1)
    res = scorer.score_one(prob, "Answer: B")
    assert isinstance(res, MMLUResult)
    assert res.correct is True
    assert res.predicted_letter == "B"
    assert res.subject == "history"


def test_score_one_wrong():
    scorer = MMLUScorer(n_shots=0)
    prob = _mk(correct=0)  # gold is A
    res = scorer.score_one(prob, "Answer: C")
    assert res.correct is False
    assert res.predicted_letter == "C"


def test_score_one_unparseable_is_wrong():
    scorer = MMLUScorer(n_shots=0)
    prob = _mk(correct=0)
    res = scorer.score_one(prob, "no letters here")
    assert res.predicted_letter is None
    assert res.correct is False


# -- aggregate score --------------------------------------------------------


def test_score_overall_accuracy_and_n_valid():
    scorer = MMLUScorer(n_shots=0)
    probs = [_mk("q0", "math", 0), _mk("q1", "math", 1), _mk("q2", "math", 2)]
    responses = ["Answer: A", "Answer: C", "garbage"]
    out = scorer.score(probs, responses)
    # q0 correct, q1 wrong, q2 invalid/wrong -> 1/3
    assert out["overall_accuracy"] == pytest.approx(1 / 3)
    assert out["n_valid"] == 2
    assert out["n_total"] == 3


def test_score_per_subject_breakdown():
    scorer = MMLUScorer(n_shots=0)
    probs = [
        _mk("q0", "math", 0),
        _mk("q1", "math", 1),
        _mk("q2", "history", 2),
    ]
    responses = ["Answer: A", "Answer: B", "Answer: A"]
    out = scorer.score(probs, responses)
    per = out["per_subject"]
    assert per["math"]["n"] == 2
    assert per["math"]["correct"] == 2
    assert per["math"]["accuracy"] == pytest.approx(1.0)
    assert per["history"]["n"] == 1
    assert per["history"]["correct"] == 0
    assert per["history"]["accuracy"] == pytest.approx(0.0)


def test_score_empty_returns_zeros():
    scorer = MMLUScorer(n_shots=0)
    out = scorer.score([], [])
    assert out["overall_accuracy"] == 0.0
    assert out["n_valid"] == 0
    assert out["n_total"] == 0
    assert out["per_subject"] == {}


def test_score_mismatched_lengths_raises():
    scorer = MMLUScorer(n_shots=0)
    with pytest.raises(ValueError):
        scorer.score([_mk()], ["a", "b"])


# -- prompt formatting ------------------------------------------------------


def test_format_prompt_zero_shot_contains_question_and_choices():
    prob = _mk()
    out = format_prompt(prob, few_shot_examples=None, cot=False)
    assert "Question q0?" in out
    assert "A) alpha" in out
    assert "B) beta" in out
    assert "C) gamma" in out
    assert "D) delta" in out
    assert "Answer:" in out
    # No exemplar answers appear in zero-shot rendering
    assert out.count("Answer: ") == 0  # "Answer:\n" / "Answer:" but no "Answer: X"


def test_format_prompt_five_shot_includes_five_exemplars():
    prob = _mk()
    out = format_prompt(prob, few_shot_examples=CANONICAL_EXEMPLARS[:5], cot=False)
    # Each exemplar contributes one "Answer: X" line (X in A..D)
    answer_lines = [line for line in out.splitlines() if line.startswith("Answer: ")]
    assert len(answer_lines) == 5
    # All exemplar questions appear
    for ex in CANONICAL_EXEMPLARS[:5]:
        assert ex.question in out


def test_cot_flag_adds_step_by_step_instruction():
    prob = _mk()
    out_no = format_prompt(prob, few_shot_examples=None, cot=False)
    out_cot = format_prompt(prob, few_shot_examples=None, cot=True)
    assert "Let's think step by step" not in out_no
    assert "Let's think step by step" in out_cot
    assert "Final answer" in out_cot


def test_scorer_format_prompt_uses_n_shots():
    scorer = MMLUScorer(n_shots=3)
    out = scorer.format_prompt(_mk())
    answer_lines = [line for line in out.splitlines() if line.startswith("Answer: ")]
    assert len(answer_lines) == 3


# -- run --------------------------------------------------------------------


def test_run_calls_generate_fn_per_problem():
    calls = []

    def gen(prompt: str) -> str:
        calls.append(prompt)
        return "Answer: A"

    scorer = MMLUScorer(generate_fn=gen, n_shots=0)
    probs = [_mk("q0", "math", 0), _mk("q1", "math", 0), _mk("q2", "math", 0)]
    results = scorer.run(probs)
    assert len(calls) == 3
    assert len(results) == 3
    assert all(r.correct for r in results)


def test_run_without_generate_fn_raises():
    scorer = MMLUScorer(generate_fn=None, n_shots=0)
    with pytest.raises(ValueError):
        scorer.run([_mk()])


def test_run_few_shot_pool_shorter_than_n_shots_uses_all_available():
    captured = []

    def gen(prompt: str) -> str:
        captured.append(prompt)
        return "Answer: A"

    scorer = MMLUScorer(generate_fn=gen, n_shots=5)
    pool = CANONICAL_EXEMPLARS[:2]  # only 2 exemplars available
    scorer.run([_mk()], few_shot_pool=pool)
    # Should use 2 exemplars (all available), not error.
    prompt = captured[0]
    answer_lines = [line for line in prompt.splitlines() if line.startswith("Answer: ")]
    assert len(answer_lines) == 2


# -- determinism ------------------------------------------------------------


def test_determinism_same_inputs_same_outputs():
    scorer = MMLUScorer(n_shots=5, cot=True)
    prob = _mk()
    p1 = scorer.format_prompt(prob)
    p2 = scorer.format_prompt(prob)
    assert p1 == p2

    probs = [_mk("q0", "math", 0), _mk("q1", "bio", 2)]
    responses = ["(A)", "(C)"]
    a = scorer.score(probs, responses)
    b = scorer.score(probs, responses)
    assert a == b


# -- MMLUProblem validation -------------------------------------------------


def test_invalid_correct_index_raises():
    with pytest.raises(ValueError):
        MMLUProblem(
            question_id="q",
            subject="math",
            question="?",
            choices=["a", "b", "c", "d"],
            correct_index=4,
        )
    with pytest.raises(ValueError):
        MMLUProblem(
            question_id="q",
            subject="math",
            question="?",
            choices=["a", "b", "c", "d"],
            correct_index=-1,
        )


def test_invalid_choice_count_raises():
    with pytest.raises(ValueError):
        MMLUProblem(
            question_id="q",
            subject="math",
            question="?",
            choices=["a", "b", "c"],
            correct_index=0,
        )


def test_invalid_n_shots_raises():
    with pytest.raises(ValueError):
        MMLUScorer(n_shots=-1)
