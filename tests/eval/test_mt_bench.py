"""Tests for the MT-Bench evaluation harness."""

from __future__ import annotations

import pytest

from src.eval.mt_bench import (
    JudgeModel,
    MTBenchCategory,
    MTBenchEvaluator,
    MTBenchQuestion,
    MTBenchResult,
    build_judge_prompt,
    extract_score_from_text,
    get_sample_questions,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_generate_fn(response: str = "Score: 8"):
    """Return a deterministic generate function."""

    def generate_fn(prompt: str) -> str:
        return response

    return generate_fn


def make_judge(response: str = "Score: 8") -> JudgeModel:
    return JudgeModel(generate_fn=make_generate_fn(response))


def make_evaluator(
    model_response: str = "This is a great answer.", judge_response: str = "Score: 8"
):
    """Build an evaluator where model returns model_response and judge returns judge_response."""
    judge = make_judge(judge_response)
    # model generate_fn always returns model_response; judge uses its own generate_fn internally
    return MTBenchEvaluator(judge=judge, generate_fn=make_generate_fn(model_response))


# ---------------------------------------------------------------------------
# 1. extract_score_from_text — "Score: 7" -> 7.0
# ---------------------------------------------------------------------------


def test_extract_score_from_score_prefix():
    assert extract_score_from_text("Score: 7") == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 2. extract_score_from_text — "Rating: 9" -> 9.0
# ---------------------------------------------------------------------------


def test_extract_score_from_rating_prefix():
    assert extract_score_from_text("Rating: 9") == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# 3. extract_score_from_text — "8/10" -> 8.0
# ---------------------------------------------------------------------------


def test_extract_score_from_slash_ten():
    assert extract_score_from_text("The model scored 8/10 overall.") == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 4. extract_score_from_text — garbage returns None
# ---------------------------------------------------------------------------


def test_extract_score_returns_none_on_garbage():
    assert extract_score_from_text("no numeric score here at all!") is None


# ---------------------------------------------------------------------------
# 5. build_judge_prompt — contains question and response text
# ---------------------------------------------------------------------------


def test_build_judge_prompt_contains_question_and_response():
    q = "What is 2 + 2?"
    r = "The answer is 4."
    prompt = build_judge_prompt(q, r, category="math")
    assert q in prompt
    assert r in prompt


# ---------------------------------------------------------------------------
# 6. build_judge_prompt — instructs "Score: X" output
# ---------------------------------------------------------------------------


def test_build_judge_prompt_instructs_score_output():
    prompt = build_judge_prompt("q", "r")
    assert "Score:" in prompt


# ---------------------------------------------------------------------------
# 7. JudgeModel.score_response — returns float in [1, 10]
# ---------------------------------------------------------------------------


def test_judge_model_score_response_returns_float_in_range():
    judge = make_judge("Score: 6")
    score = judge.score_response("What is AI?", "AI is intelligence.", "general")
    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# 8. JudgeModel with mock returning "Score: 8" -> 8.0
# ---------------------------------------------------------------------------


def test_judge_model_score_response_parses_correctly():
    judge = make_judge("Score: 8")
    score = judge.score_response("question", "response", "general")
    assert score == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 9. score_multi_turn length matches input
# ---------------------------------------------------------------------------


def test_score_multi_turn_length_matches_input():
    judge = make_judge("Score: 7")
    questions = ["Q1", "Q2", "Q3"]
    responses = ["R1", "R2", "R3"]
    scores = judge.score_multi_turn(questions, responses, "reasoning")
    assert len(scores) == len(questions)


# ---------------------------------------------------------------------------
# 10. MTBenchQuestion dataclass creation
# ---------------------------------------------------------------------------


def test_mtbench_question_dataclass_creation():
    q = MTBenchQuestion(
        question_id=42,
        category=MTBenchCategory.MATH,
        turns=["What is pi?"],
        reference_answer="3.14159...",
    )
    assert q.question_id == 42
    assert q.category == MTBenchCategory.MATH
    assert len(q.turns) == 1
    assert q.reference_answer == "3.14159..."


# ---------------------------------------------------------------------------
# 11. MTBenchResult mean_score computed correctly
# ---------------------------------------------------------------------------


def test_mtbench_result_mean_score_computed():
    result = MTBenchResult(
        question_id=1,
        category="math",
        scores=[6.0, 8.0, 10.0],
        judge_outputs=["Score: 6", "Score: 8", "Score: 10"],
    )
    assert result.mean_score == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 12. MTBenchEvaluator.evaluate_question returns MTBenchResult
# ---------------------------------------------------------------------------


def test_evaluate_question_returns_mtbench_result():
    evaluator = make_evaluator(judge_response="Score: 7")
    question = MTBenchQuestion(
        question_id=1,
        category=MTBenchCategory.WRITING,
        turns=["Write a haiku about rain."],
    )
    result = evaluator.evaluate_question(question)
    assert isinstance(result, MTBenchResult)
    assert result.question_id == 1


# ---------------------------------------------------------------------------
# 13. evaluate_all length matches input questions
# ---------------------------------------------------------------------------


def test_evaluate_all_length_matches_input():
    evaluator = make_evaluator(judge_response="Score: 5")
    questions = [MTBenchQuestion(question_id=i, category="math", turns=[f"Q{i}"]) for i in range(4)]
    results = evaluator.evaluate_all(questions)
    assert len(results) == len(questions)


# ---------------------------------------------------------------------------
# 14. compute_summary keys (overall_score, per_category, n_questions)
# ---------------------------------------------------------------------------


def test_compute_summary_has_required_keys():
    evaluator = make_evaluator(judge_response="Score: 8")
    questions = [
        MTBenchQuestion(question_id=1, category="math", turns=["Q1"]),
        MTBenchQuestion(question_id=2, category="coding", turns=["Q2"]),
    ]
    results = evaluator.evaluate_all(questions)
    summary = evaluator.compute_summary(results)
    assert "overall_score" in summary
    assert "per_category" in summary
    assert "n_questions" in summary


# ---------------------------------------------------------------------------
# 15. compute_summary overall_score in [0, 10]
# ---------------------------------------------------------------------------


def test_compute_summary_overall_score_in_range():
    evaluator = make_evaluator(judge_response="Score: 9")
    questions = [MTBenchQuestion(question_id=1, category="stem", turns=["Q"])]
    results = evaluator.evaluate_all(questions)
    summary = evaluator.compute_summary(results)
    assert 0.0 <= summary["overall_score"] <= 10.0


# ---------------------------------------------------------------------------
# 16. compute_summary n_questions matches
# ---------------------------------------------------------------------------


def test_compute_summary_n_questions_matches():
    evaluator = make_evaluator(judge_response="Score: 6")
    questions = [
        MTBenchQuestion(question_id=i, category="reasoning", turns=[f"Q{i}"]) for i in range(5)
    ]
    results = evaluator.evaluate_all(questions)
    summary = evaluator.compute_summary(results)
    assert summary["n_questions"] == 5


# ---------------------------------------------------------------------------
# 17. get_sample_questions returns list of MTBenchQuestion
# ---------------------------------------------------------------------------


def test_get_sample_questions_returns_list_of_mtbench_questions():
    questions = get_sample_questions()
    assert isinstance(questions, list)
    assert len(questions) >= 3
    for q in questions:
        assert isinstance(q, MTBenchQuestion)


# ---------------------------------------------------------------------------
# 18. per_category contains correct categories from evaluated questions
# ---------------------------------------------------------------------------


def test_compute_summary_per_category_contains_correct_categories():
    evaluator = make_evaluator(judge_response="Score: 7")
    questions = [
        MTBenchQuestion(question_id=1, category="math", turns=["Q1"]),
        MTBenchQuestion(question_id=2, category="coding", turns=["Q2"]),
        MTBenchQuestion(question_id=3, category="math", turns=["Q3"]),
    ]
    results = evaluator.evaluate_all(questions)
    summary = evaluator.compute_summary(results)
    assert "math" in summary["per_category"]
    assert "coding" in summary["per_category"]


# ---------------------------------------------------------------------------
# 19. score default 5.0 when no score parseable
# ---------------------------------------------------------------------------


def test_score_default_5_when_no_score_parseable():
    judge = make_judge("No score here, just chatter.")
    score = judge.score_response("question", "response", "general")
    assert score == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 20. MTBenchResult scores list length matches turns
# ---------------------------------------------------------------------------


def test_mtbench_result_scores_length_matches_turns():
    evaluator = make_evaluator(judge_response="Score: 8")
    question = MTBenchQuestion(
        question_id=99,
        category=MTBenchCategory.STEM,
        turns=["Turn 1", "Turn 2"],
    )
    result = evaluator.evaluate_question(question)
    assert len(result.scores) == len(question.turns)
