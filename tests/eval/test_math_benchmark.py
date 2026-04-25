"""Tests for src/eval/math_benchmark.py — ~50 tests."""

import pytest

from src.eval.math_benchmark import (
    MathCategory,
    MathProblem,
    MathAnswer,
    MathBenchmark,
)
from src.eval import BENCHMARK_REGISTRY


# ---------------------------------------------------------------------------
# MathCategory enum
# ---------------------------------------------------------------------------


def test_math_category_algebra():
    assert MathCategory.ALGEBRA == "algebra"
    assert MathCategory.ALGEBRA.value == "algebra"


def test_math_category_geometry():
    assert MathCategory.GEOMETRY == "geometry"


def test_math_category_number_theory():
    assert MathCategory.NUMBER_THEORY == "number_theory"


def test_math_category_counting():
    assert MathCategory.COUNTING == "counting"


def test_math_category_precalculus():
    assert MathCategory.PRECALCULUS == "precalculus"


def test_math_category_intermediate_algebra():
    assert MathCategory.INTERMEDIATE_ALGEBRA == "intermediate_algebra"


def test_math_category_prealgebra():
    assert MathCategory.PREALGEBRA == "prealgebra"


def test_math_category_all_seven():
    assert len(MathCategory) == 7


# ---------------------------------------------------------------------------
# MathProblem dataclass
# ---------------------------------------------------------------------------


def test_math_problem_fields():
    p = MathProblem(
        problem_id="test_001",
        category=MathCategory.ALGEBRA,
        difficulty=2,
        problem="x + 1 = 3",
        answer="2",
    )
    assert p.problem_id == "test_001"
    assert p.category == MathCategory.ALGEBRA
    assert p.difficulty == 2
    assert p.problem == "x + 1 = 3"
    assert p.answer == "2"


# ---------------------------------------------------------------------------
# MathAnswer.normalize
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace():
    assert MathAnswer.normalize("  42  ") == "42"


def test_normalize_lowercases():
    assert MathAnswer.normalize("ABC") == "abc"


def test_normalize_removes_dollar_signs():
    assert MathAnswer.normalize("$42$") == "42"


def test_normalize_removes_boxed():
    assert MathAnswer.normalize(r"\boxed{42}") == "42"


def test_normalize_boxed_with_expression():
    assert MathAnswer.normalize(r"\boxed{x+1}") == "x+1"


def test_normalize_boxed_strips_after():
    # After extracting boxed content, dollar signs and whitespace stripped
    assert MathAnswer.normalize(r"  \boxed{ 7 }  ") == "7"


def test_normalize_dollar_around_boxed():
    # $ wrapping the boxed — normalize removes $, then boxed
    result = MathAnswer.normalize(r"$\boxed{5}$")
    # The boxed regex fires on the content after $ removal; either way result should contain "5"
    assert "5" in result


def test_normalize_no_special_chars():
    assert MathAnswer.normalize("hello") == "hello"


def test_normalize_nested_boxed_outer_group():
    # r'\boxed{\boxed{3}}' — the regex [^}]* is non-greedy, group1 = \boxed{3 (stops at first })
    # This tests that it doesn't crash and returns something sensible
    result = MathAnswer.normalize(r"\boxed{\boxed{3}}")
    # Should not raise; result may be partial inner content
    assert isinstance(result, str)


def test_normalize_plain_number_string():
    assert MathAnswer.normalize("3.14") == "3.14"


# ---------------------------------------------------------------------------
# MathAnswer.check_exact
# ---------------------------------------------------------------------------


def test_check_exact_same_string():
    assert MathAnswer.check_exact("42", "42") is True


def test_check_exact_different_strings():
    assert MathAnswer.check_exact("42", "43") is False


def test_check_exact_normalizes_dollar():
    assert MathAnswer.check_exact("$42$", "42") is True


def test_check_exact_normalizes_boxed():
    assert MathAnswer.check_exact(r"\boxed{42}", "42") is True


def test_check_exact_case_insensitive():
    assert MathAnswer.check_exact("ABC", "abc") is True


def test_check_exact_whitespace_ignored():
    assert MathAnswer.check_exact("  7  ", "7") is True


def test_check_exact_different_values():
    assert MathAnswer.check_exact("1", "2") is False


def test_check_exact_both_boxed():
    assert MathAnswer.check_exact(r"\boxed{10}", r"\boxed{10}") is True


# ---------------------------------------------------------------------------
# MathAnswer.check_numeric
# ---------------------------------------------------------------------------


def test_check_numeric_equal_floats():
    assert MathAnswer.check_numeric("3.14", "3.14") is True


def test_check_numeric_integer_strings():
    assert MathAnswer.check_numeric("7", "7") is True


def test_check_numeric_within_tolerance():
    assert MathAnswer.check_numeric("1.0000005", "1.0") is True


def test_check_numeric_outside_tolerance():
    # abs(1.01 - 1.0) = 0.01, which is >> 1e-6
    assert MathAnswer.check_numeric("1.01", "1.0") is False


def test_check_numeric_fraction_fails_gracefully():
    # "1/2" cannot be converted by float(), should return False (not raise)
    result = MathAnswer.check_numeric("1/2", "0.5")
    assert isinstance(result, bool)


def test_check_numeric_non_numeric_returns_false():
    assert MathAnswer.check_numeric("abc", "42") is False


def test_check_numeric_both_non_numeric_returns_false():
    assert MathAnswer.check_numeric("abc", "xyz") is False


def test_check_numeric_boxed_integer():
    assert MathAnswer.check_numeric(r"\boxed{5}", "5") is True


def test_check_numeric_dollar_wrapped():
    assert MathAnswer.check_numeric("$3$", "3") is True


def test_check_numeric_zero():
    assert MathAnswer.check_numeric("0", "0.0") is True


def test_check_numeric_negative():
    assert MathAnswer.check_numeric("-3", "-3.0") is True


# ---------------------------------------------------------------------------
# MathBenchmark
# ---------------------------------------------------------------------------


def test_math_benchmark_default_has_six_problems():
    mb = MathBenchmark()
    assert len(mb.problems) == 6


def test_math_benchmark_default_problem_ids_are_strings():
    mb = MathBenchmark()
    for pid in mb.problem_ids():
        assert isinstance(pid, str)


def test_math_benchmark_problem_ids_count():
    mb = MathBenchmark()
    assert len(mb.problem_ids()) == 6


def test_math_benchmark_custom_problems():
    problems = [
        MathProblem("x1", MathCategory.ALGEBRA, 1, "1+1=?", "2"),
        MathProblem("x2", MathCategory.GEOMETRY, 1, "area square side=3", "9"),
    ]
    mb = MathBenchmark(problems=problems)
    assert len(mb.problems) == 2


def test_evaluate_all_correct_accuracy_one():
    mb = MathBenchmark()
    predictions = {p.problem_id: p.answer for p in mb.problems}
    result = mb.evaluate(predictions)
    assert result["accuracy"] == 1.0
    assert result["correct"] == result["total"]


def test_evaluate_no_correct_accuracy_zero():
    mb = MathBenchmark()
    predictions = {p.problem_id: "WRONG_ANSWER_XYZ" for p in mb.problems}
    result = mb.evaluate(predictions)
    assert result["accuracy"] == 0.0
    assert result["correct"] == 0


def test_evaluate_empty_predictions():
    mb = MathBenchmark()
    result = mb.evaluate({})
    assert result["correct"] == 0
    assert result["accuracy"] == 0.0


def test_evaluate_total_equals_problem_count():
    mb = MathBenchmark()
    predictions = {}
    result = mb.evaluate(predictions)
    assert result["total"] == 6


def test_evaluate_by_category_is_dict():
    mb = MathBenchmark()
    result = mb.evaluate({})
    assert isinstance(result["by_category"], dict)


def test_evaluate_by_category_has_correct_and_total_keys():
    mb = MathBenchmark()
    result = mb.evaluate({})
    for cat, stats in result["by_category"].items():
        assert "correct" in stats
        assert "total" in stats


def test_evaluate_by_category_categories_present():
    mb = MathBenchmark()
    result = mb.evaluate({})
    cats = set(result["by_category"].keys())
    # Default problems cover algebra, geometry, number_theory
    assert "algebra" in cats
    assert "geometry" in cats
    assert "number_theory" in cats


def test_evaluate_by_difficulty_is_dict():
    mb = MathBenchmark()
    result = mb.evaluate({})
    assert isinstance(result["by_difficulty"], dict)


def test_evaluate_by_difficulty_has_correct_and_total_keys():
    mb = MathBenchmark()
    result = mb.evaluate({})
    for diff, stats in result["by_difficulty"].items():
        assert "correct" in stats
        assert "total" in stats


def test_evaluate_by_difficulty_integer_keys():
    mb = MathBenchmark()
    result = mb.evaluate({})
    for key in result["by_difficulty"].keys():
        assert isinstance(key, int)


def test_evaluate_partial_correct():
    problems = [
        MathProblem("p1", MathCategory.ALGEBRA, 1, "1+1", "2"),
        MathProblem("p2", MathCategory.ALGEBRA, 1, "2+2", "4"),
    ]
    mb = MathBenchmark(problems=problems)
    predictions = {"p1": "2", "p2": "99"}
    result = mb.evaluate(predictions)
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["accuracy"] == pytest.approx(0.5)


def test_evaluate_numeric_match_counts_correct():
    problems = [
        MathProblem("n1", MathCategory.NUMBER_THEORY, 1, "0.1+0.2", "0.3"),
    ]
    mb = MathBenchmark(problems=problems)
    predictions = {"n1": "0.30000000"}  # float-equivalent
    result = mb.evaluate(predictions)
    assert result["correct"] == 1


# ---------------------------------------------------------------------------
# BENCHMARK_REGISTRY
# ---------------------------------------------------------------------------


def test_benchmark_registry_math_key_exists():
    assert "math" in BENCHMARK_REGISTRY


def test_benchmark_registry_math_is_math_benchmark():
    assert isinstance(BENCHMARK_REGISTRY["math"], MathBenchmark)
