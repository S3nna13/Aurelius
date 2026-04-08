"""Tests for evidence coverage curve helpers."""

import pytest

from src.eval.evidence_coverage_curve import area_under_coverage_curve, cumulative_recall, first_full_coverage_index


def test_cumulative_recall_builds_curve():
    assert cumulative_recall([True, False, True]) == [0.5, 0.5, 1.0]


def test_area_under_coverage_curve_averages_curve_values():
    area = area_under_coverage_curve([True, False, True])
    assert area == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_first_full_coverage_index_returns_first_complete_hit():
    assert first_full_coverage_index([True, False, True]) == 2


def test_first_full_coverage_index_none_when_no_relevant_items():
    assert first_full_coverage_index([False, False]) is None


def test_cumulative_recall_zero_curve_when_no_relevant_items():
    assert cumulative_recall([False, False]) == [0.0, 0.0]


def test_area_under_coverage_curve_zero_for_empty_input():
    assert area_under_coverage_curve([]) == pytest.approx(0.0)


def test_first_full_coverage_index_none_for_empty_input():
    assert first_full_coverage_index([]) is None

