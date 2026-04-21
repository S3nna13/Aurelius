"""
Unit tests for src/eval/math_eval.py

Run with:
    .venv/bin/python3.14 -m pytest tests/eval/test_math_eval.py -v
"""

import pytest
from src.eval.math_eval import MathEval, MathEvalConfig, MathAnswer


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """MathEvalConfig should have the correct default values."""
    cfg = MathEvalConfig()
    assert cfg.numeric_tolerance == pytest.approx(1e-6)
    assert cfg.extract_boxed is True
    assert cfg.extract_last_number is True
    assert cfg.normalize_fractions is True


# ---------------------------------------------------------------------------
# 2. test_extract_boxed
# ---------------------------------------------------------------------------

def test_extract_boxed():
    """\\boxed{42} should yield extracted='42', numeric=42.0."""
    ev = MathEval()
    ans = ev.extract_answer("We compute and find \\boxed{42}.")
    assert ans.extracted == "42"
    assert ans.numeric == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 3. test_extract_last_number
# ---------------------------------------------------------------------------

def test_extract_last_number():
    """When no boxed content, last number in text should be extracted."""
    ev = MathEval()
    ans = ev.extract_answer("The answer is 7.")
    assert ans.extracted == "7"
    assert ans.numeric == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 4. test_extract_fraction
# ---------------------------------------------------------------------------

def test_extract_fraction():
    """\\boxed{3/7} should yield numeric ≈ 0.42857, is_fraction=True."""
    ev = MathEval()
    ans = ev.extract_answer("Therefore the answer is \\boxed{3/7}.")
    assert ans.extracted == "3/7"
    assert ans.numeric == pytest.approx(3 / 7, abs=1e-4)
    assert ans.is_fraction is True


# ---------------------------------------------------------------------------
# 5. test_extract_none
# ---------------------------------------------------------------------------

def test_extract_none():
    """A response with no numbers and no boxed content → extracted=None."""
    ev = MathEval()
    ans = ev.extract_answer("I have no idea what the answer might be.")
    assert ans.numeric is None


# ---------------------------------------------------------------------------
# 6. test_normalize_strips_whitespace
# ---------------------------------------------------------------------------

def test_normalize_strips_whitespace():
    """normalize_answer should strip leading/trailing whitespace."""
    ev = MathEval()
    norm_str, num = ev.normalize_answer("  42  ")
    assert norm_str == "42"
    assert num == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# 7. test_is_correct_exact
# ---------------------------------------------------------------------------

def test_is_correct_exact():
    """Exact string match (after normalization) should return True."""
    ev = MathEval()
    assert ev.is_correct("42", "42") is True


# ---------------------------------------------------------------------------
# 8. test_is_correct_numeric
# ---------------------------------------------------------------------------

def test_is_correct_numeric():
    """'42.0' vs '42' should match within default tolerance."""
    ev = MathEval()
    assert ev.is_correct("42.0", "42") is True


# ---------------------------------------------------------------------------
# 9. test_is_correct_fraction
# ---------------------------------------------------------------------------

def test_is_correct_fraction():
    """'1/2' vs '0.5' should match within tolerance."""
    ev = MathEval()
    assert ev.is_correct("1/2", "0.5") is True


# ---------------------------------------------------------------------------
# 10. test_is_correct_wrong
# ---------------------------------------------------------------------------

def test_is_correct_wrong():
    """'41' vs '42' should not match."""
    ev = MathEval()
    assert ev.is_correct("41", "42") is False


# ---------------------------------------------------------------------------
# 11. test_evaluate_accuracy
# ---------------------------------------------------------------------------

def test_evaluate_accuracy():
    """All correct predictions should yield accuracy=1.0."""
    ev = MathEval()
    predictions = [
        "The answer is \\boxed{10}.",
        "So we get \\boxed{20}.",
        "Final answer: \\boxed{30}.",
    ]
    ground_truths = ["10", "20", "30"]
    result = ev.evaluate(predictions, ground_truths)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n_correct"] == 3
    assert result["n_total"] == 3
    assert result["n_extraction_failed"] == 0


# ---------------------------------------------------------------------------
# 12. test_evaluate_by_category
# ---------------------------------------------------------------------------

def test_evaluate_by_category():
    """evaluate() with categories should return per-category breakdown."""
    ev = MathEval()
    predictions = [
        "\\boxed{1}",   # algebra, correct
        "\\boxed{99}", # algebra, wrong (gt=2)
        "\\boxed{3}",  # number_theory, correct
    ]
    ground_truths = ["1", "2", "3"]
    categories = ["algebra", "algebra", "number_theory"]
    result = ev.evaluate(predictions, ground_truths, categories=categories)

    assert "by_category" in result
    by_cat = result["by_category"]

    assert by_cat["algebra"]["n"] == 2
    assert by_cat["algebra"]["accuracy"] == pytest.approx(0.5)
    assert by_cat["number_theory"]["n"] == 1
    assert by_cat["number_theory"]["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 13. test_evaluate_extraction_failed
# ---------------------------------------------------------------------------

def test_evaluate_extraction_failed():
    """Completely unparseable responses should be counted in n_extraction_failed."""
    cfg = MathEvalConfig(extract_boxed=False, extract_last_number=False)
    ev = MathEval(config=cfg)
    # With both extraction modes off, "I don't know" has no boxed and no number
    # extraction, and the raw text is returned as extracted (non-None).
    # To force extraction_failed, we need a response that gives extracted=None.
    # With extract_boxed=False and extract_last_number=False the fallback path
    # returns the raw stripped text, which won't be None for a non-empty string.
    # Use a truly empty or whitespace response.
    predictions = ["   "]   # whitespace only → extracted = None
    ground_truths = ["42"]
    result = ev.evaluate(predictions, ground_truths)
    assert result["n_extraction_failed"] == 1
    assert result["n_correct"] == 0


# ---------------------------------------------------------------------------
# 14. test_aime_score_integer
# ---------------------------------------------------------------------------

def test_aime_score_integer():
    """AIME: '042' in prediction text should match ground truth '42'."""
    ev = MathEval()
    predictions = [
        "The answer is \\boxed{042}.",
        "We compute \\boxed{100}.",
        "Thus \\boxed{999}.",
    ]
    ground_truths = ["42", "100", "999"]
    result = ev.aime_score(predictions, ground_truths)
    assert result["score"] == 3
    assert result["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. test_evaluate_empty
# ---------------------------------------------------------------------------

def test_evaluate_empty():
    """Empty prediction / ground-truth lists should not crash and return 0.0."""
    ev = MathEval()
    result = ev.evaluate([], [])
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["n_total"] == 0
    assert result["n_correct"] == 0
    assert result["n_extraction_failed"] == 0


# ---------------------------------------------------------------------------
# 16. test_extract_boxed_with_expression
# ---------------------------------------------------------------------------

def test_extract_boxed_with_expression():
    """\\boxed{} containing a simple expression like '-5' should extract correctly."""
    ev = MathEval()
    ans = ev.extract_answer("Therefore \\boxed{-5} is the root.")
    assert ans.extracted == "-5"
    assert ans.numeric == pytest.approx(-5.0)
