"""Tests for judge conflict helpers."""

import pytest

from src.eval.judge_conflict import conflict_entropy, conflict_rate, unanimous


def test_conflict_rate_zero_for_unanimous_verdicts():
    assert conflict_rate(["yes", "yes"]) == pytest.approx(0.0)


def test_conflict_rate_positive_for_mixed_verdicts():
    assert conflict_rate(["yes", "no", "yes"]) > 0.0


def test_unanimous_true_when_all_same():
    assert unanimous(["a", "a"])


def test_unanimous_false_when_not_all_same():
    assert not unanimous(["a", "b"])


def test_conflict_entropy_zero_for_unanimous():
    assert conflict_entropy(["a", "a"]) == pytest.approx(0.0)


def test_conflict_entropy_positive_for_mixed_verdicts():
    assert conflict_entropy(["a", "b"]) > 0.0


def test_conflict_rate_handles_empty_input():
    assert conflict_rate([]) == pytest.approx(0.0)

