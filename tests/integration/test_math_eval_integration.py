"""
Integration tests for src/eval/math_eval.py

Verifies:
- evaluate() on a 5-problem mixed set (boxed, last-number, fractions)
- aime_score() on 3 integer problems
- Registry wiring: BENCHMARK_REGISTRY["math_eval"] == MathEval

Run with:
    .venv/bin/python3.14 -m pytest tests/integration/test_math_eval_integration.py -v
"""

import pytest

from src.eval.math_eval import MathEval

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator() -> MathEval:
    return MathEval()


# ---------------------------------------------------------------------------
# 5-problem mixed evaluation
# ---------------------------------------------------------------------------

# Problem set: mix of boxed, last-number, and fraction answers.
MIXED_PREDICTIONS = [
    # 1. Boxed integer — correct
    "After substitution we get \\boxed{15}.",
    # 2. Boxed fraction — correct (gt=0.25)
    "The probability is \\boxed{1/4}.",
    # 3. Last-number fallback (no boxed) — correct
    "Expanding the series gives us the value 7.",
    # 4. Boxed decimal — correct
    "We evaluate to \\boxed{3.14}.",
    # 5. Boxed integer — wrong (gt=100)
    "Clearly \\boxed{99} is the result.",
]

MIXED_GROUND_TRUTHS = ["15", "0.25", "7", "3.14", "100"]

MIXED_CATEGORIES = [
    "algebra",
    "probability",
    "number_theory",
    "calculus",
    "algebra",
]


def test_mixed_evaluate_accuracy(evaluator):
    """4 out of 5 correct → accuracy 0.8."""
    result = evaluator.evaluate(MIXED_PREDICTIONS, MIXED_GROUND_TRUTHS)
    assert result["n_total"] == 5
    assert result["n_correct"] == 4
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["n_extraction_failed"] == 0


def test_mixed_evaluate_by_category(evaluator):
    """Per-category breakdown should be correct."""
    result = evaluator.evaluate(MIXED_PREDICTIONS, MIXED_GROUND_TRUTHS, categories=MIXED_CATEGORIES)
    assert "by_category" in result
    by_cat = result["by_category"]

    # algebra: problem 1 (correct) + problem 5 (wrong) = 1/2
    assert by_cat["algebra"]["n"] == 2
    assert by_cat["algebra"]["accuracy"] == pytest.approx(0.5)

    # probability: 1/1 correct
    assert by_cat["probability"]["n"] == 1
    assert by_cat["probability"]["accuracy"] == pytest.approx(1.0)

    # number_theory: 1/1 correct
    assert by_cat["number_theory"]["n"] == 1
    assert by_cat["number_theory"]["accuracy"] == pytest.approx(1.0)

    # calculus: 1/1 correct
    assert by_cat["calculus"]["n"] == 1
    assert by_cat["calculus"]["accuracy"] == pytest.approx(1.0)


def test_mixed_evaluate_fraction_correct(evaluator):
    """Fraction answer '1/4' should match ground truth '0.25'."""
    result = evaluator.evaluate([MIXED_PREDICTIONS[1]], [MIXED_GROUND_TRUTHS[1]])
    assert result["n_correct"] == 1


# ---------------------------------------------------------------------------
# AIME-specific scoring
# ---------------------------------------------------------------------------

AIME_PREDICTIONS = [
    "By inspection, \\boxed{042}.",  # 42 → correct
    "We get \\boxed{314}.",  # 314 → correct
    "The answer must be \\boxed{500}.",  # 500 → correct
]

AIME_GROUND_TRUTHS = ["42", "314", "500"]


def test_aime_score_all_correct(evaluator):
    """All 3 AIME answers correct → score=3, accuracy=1.0."""
    result = evaluator.aime_score(AIME_PREDICTIONS, AIME_GROUND_TRUTHS)
    assert result["score"] == 3
    assert result["accuracy"] == pytest.approx(1.0)


def test_aime_score_partial(evaluator):
    """One wrong AIME answer → score=2, accuracy≈0.667."""
    preds = list(AIME_PREDICTIONS)
    preds[2] = "The answer is \\boxed{999}."  # wrong (gt=500)
    result = evaluator.aime_score(preds, AIME_GROUND_TRUTHS)
    assert result["score"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_wired():
    """BENCHMARK_REGISTRY['math_eval'] should be the MathEval class."""
    # Import through the package to trigger the registry wiring.
    from src.eval import BENCHMARK_REGISTRY  # type: ignore

    assert "math_eval" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["math_eval"] is MathEval
