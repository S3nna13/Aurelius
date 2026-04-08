"""Tests for model-written eval helpers."""

import pytest

from src.eval.model_written_evals import (
    average_rubric_score,
    parse_written_eval,
    weighted_verdict_score,
    written_eval_agreement,
)


def make_text(verdict: str = "pass", confidence: float = 0.8, rubric_score: float = 7.0) -> str:
    return (
        f"Verdict: {verdict}\n"
        "Rationale: concise and grounded\n"
        f"Confidence: {confidence}\n"
        f"Rubric Score: {rubric_score}\n"
    )


def test_parse_written_eval_extracts_fields():
    parsed = parse_written_eval(make_text())
    assert parsed.verdict == "pass"
    assert parsed.confidence == pytest.approx(0.8)


def test_parse_written_eval_rejects_missing_fields():
    with pytest.raises(ValueError):
        parse_written_eval("Verdict: pass\n")


def test_weighted_verdict_score_uses_positive_label():
    evals = [parse_written_eval(make_text("pass", 0.9)), parse_written_eval(make_text("fail", 0.4))]
    assert weighted_verdict_score(evals) == pytest.approx(0.45)


def test_average_rubric_score_means_values():
    evals = [parse_written_eval(make_text(rubric_score=6.0)), parse_written_eval(make_text(rubric_score=8.0))]
    assert average_rubric_score(evals) == pytest.approx(7.0)


def test_written_eval_agreement_tracks_majority():
    evals = [
        parse_written_eval(make_text("pass")),
        parse_written_eval(make_text("pass")),
        parse_written_eval(make_text("fail")),
    ]
    assert written_eval_agreement(evals) == pytest.approx(2.0 / 3.0)


def test_weighted_verdict_score_is_zero_for_empty_list():
    assert weighted_verdict_score([]) == pytest.approx(0.0)


def test_average_rubric_score_is_zero_for_empty_list():
    assert average_rubric_score([]) == pytest.approx(0.0)
