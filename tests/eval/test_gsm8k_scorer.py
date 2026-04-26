"""Unit tests for src/eval/gsm8k_scorer.py — 12 tests."""

from __future__ import annotations

import pytest

from src.eval.gsm8k_scorer import GSM8KScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def scorer() -> GSM8KScorer:
    return GSM8KScorer()


# ---------------------------------------------------------------------------
# extract_answer — separator present
# ---------------------------------------------------------------------------


def test_extract_answer_with_separator(scorer: GSM8KScorer):
    text = "Step 1: 3+4=7.\nStep 2: 7*2=14.\n#### 14"
    ans = scorer.extract_answer(text)
    assert ans.extracted_number == pytest.approx(14.0)


def test_extract_answer_steps_before_separator(scorer: GSM8KScorer):
    text = "First I add.\nThen multiply.\n#### 42"
    ans = scorer.extract_answer(text)
    assert len(ans.steps) >= 1  # at least one reasoning line


def test_extract_answer_with_dollar_sign(scorer: GSM8KScorer):
    text = "The total cost is #### $35"
    ans = scorer.extract_answer(text)
    assert ans.extracted_number == pytest.approx(35.0)


def test_extract_answer_with_comma_in_number(scorer: GSM8KScorer):
    text = "Answer: #### 1,234"
    ans = scorer.extract_answer(text)
    assert ans.extracted_number == pytest.approx(1234.0)


def test_extract_answer_negative(scorer: GSM8KScorer):
    text = "Net change: #### -5"
    ans = scorer.extract_answer(text)
    assert ans.extracted_number == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# extract_answer — no separator (fallback)
# ---------------------------------------------------------------------------


def test_extract_answer_fallback_no_separator(scorer: GSM8KScorer):
    text = "After calculation, the answer is 99."
    ans = scorer.extract_answer(text)
    assert ans.extracted_number == pytest.approx(99.0)


def test_extract_answer_no_number_returns_none(scorer: GSM8KScorer):
    text = "There is no numeric answer here."
    ans = scorer.extract_answer(text)
    assert ans.extracted_number is None


def test_extract_answer_raw_text_preserved(scorer: GSM8KScorer):
    text = "Result: #### 7"
    ans = scorer.extract_answer(text)
    assert ans.raw_text == text


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_exact_match(scorer: GSM8KScorer):
    assert scorer.score("#### 42", "#### 42") is True


def test_score_within_tolerance(scorer: GSM8KScorer):
    assert scorer.score("#### 42.005", "#### 42") is True


def test_score_outside_tolerance(scorer: GSM8KScorer):
    assert scorer.score("#### 43", "#### 42") is False


def test_score_missing_prediction_returns_false(scorer: GSM8KScorer):
    assert scorer.score("no number here", "#### 42") is False


# ---------------------------------------------------------------------------
# batch_score
# ---------------------------------------------------------------------------


def test_batch_score_all_correct(scorer: GSM8KScorer):
    preds = ["#### 1", "#### 2", "#### 3"]
    gts = ["#### 1", "#### 2", "#### 3"]
    result = scorer.batch_score(preds, gts)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n_correct"] == 3
    assert result["n_total"] == 3


def test_batch_score_none_correct(scorer: GSM8KScorer):
    preds = ["#### 99", "#### 88"]
    gts = ["#### 1", "#### 2"]
    result = scorer.batch_score(preds, gts)
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["n_correct"] == 0


def test_batch_score_partial(scorer: GSM8KScorer):
    preds = ["#### 10", "#### 99"]
    gts = ["#### 10", "#### 2"]
    result = scorer.batch_score(preds, gts)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["n_correct"] == 1


def test_batch_score_avg_steps_nonnegative(scorer: GSM8KScorer):
    preds = ["Step 1: add.\n#### 5"]
    gts = ["#### 5"]
    result = scorer.batch_score(preds, gts)
    assert result["avg_steps"] >= 0.0


def test_batch_score_empty_inputs(scorer: GSM8KScorer):
    result = scorer.batch_score([], [])
    assert result["n_total"] == 0
    assert result["accuracy"] == 0.0


def test_batch_score_length_mismatch_raises(scorer: GSM8KScorer):
    with pytest.raises(ValueError):
        scorer.batch_score(["#### 1"], ["#### 1", "#### 2"])


# ---------------------------------------------------------------------------
# BENCHMARK_REGISTRY
# ---------------------------------------------------------------------------


def test_benchmark_registry_gsm8k():
    from src.eval.gsm8k_scorer import BENCHMARK_REGISTRY

    assert "gsm8k" in BENCHMARK_REGISTRY
    assert isinstance(BENCHMARK_REGISTRY["gsm8k"], GSM8KScorer)
