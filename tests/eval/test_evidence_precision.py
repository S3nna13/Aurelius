"""Tests for evidence precision helpers."""

import pytest

from src.eval.evidence_precision import average_precision, precision_at_k, recall_at_k


def test_precision_at_k_computes_fraction():
    assert precision_at_k([True, False, True], 2) == pytest.approx(0.5)


def test_average_precision_rewards_early_hits():
    assert average_precision([True, False, True]) == pytest.approx((1.0 + 2 / 3) / 2)


def test_recall_at_k_uses_total_relevant_count():
    assert recall_at_k([True, False, True], 2) == pytest.approx(0.5)


def test_precision_at_k_handles_empty_list():
    assert precision_at_k([], 1) == pytest.approx(0.0)


def test_average_precision_zero_when_no_relevant_items():
    assert average_precision([False, False]) == pytest.approx(0.0)


def test_precision_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        precision_at_k([True], 0)


def test_recall_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        recall_at_k([True], 0)
