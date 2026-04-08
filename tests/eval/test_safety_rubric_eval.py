"""Tests for safety rubric dataset evaluation."""

import pytest

from src.alignment.safety_rubric import SafetyRubric
from src.eval.safety_rubric_eval import compare_safety_reports, evaluate_safety_texts


def make_rubric():
    return SafetyRubric(harmless_terms=("safe", "refuse"), harmful_terms=("harm",))


def test_evaluate_safety_texts_returns_report():
    report = evaluate_safety_texts(["safe", "harm"], make_rubric())
    assert report.mean_score == pytest.approx(0.0)


def test_evaluate_safety_texts_computes_pass_rate():
    report = evaluate_safety_texts(["safe", "safe", "harm"], make_rubric())
    assert report.pass_rate == pytest.approx(2.0 / 3.0)


def test_evaluate_safety_texts_handles_empty_input():
    report = evaluate_safety_texts([], make_rubric())
    assert report.pass_rate == pytest.approx(0.0)


def test_compare_safety_reports_prefers_higher_pass_rate():
    left = evaluate_safety_texts(["safe", "safe"], make_rubric())
    right = evaluate_safety_texts(["safe", "harm"], make_rubric())
    assert compare_safety_reports(left, right) == "left"


def test_compare_safety_reports_prefers_higher_mean_score_on_tie():
    left = evaluate_safety_texts(["safe"], make_rubric())
    right = evaluate_safety_texts(["refuse"], make_rubric())
    assert compare_safety_reports(left, right) == "tie"


def test_compare_safety_reports_returns_tie_when_equal():
    left = evaluate_safety_texts(["safe"], make_rubric())
    right = evaluate_safety_texts(["safe"], make_rubric())
    assert compare_safety_reports(left, right) == "tie"


def test_evaluate_safety_texts_tracks_extrema():
    report = evaluate_safety_texts(["safe", "harm"], make_rubric())
    assert report.max_score >= report.min_score
