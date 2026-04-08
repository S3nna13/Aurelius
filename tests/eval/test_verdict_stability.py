"""Tests for verdict stability helpers."""

import pytest

from src.eval.verdict_stability import verdict_flip_rate, verdict_mode, verdict_stability


def test_verdict_mode_returns_modal_label():
    assert verdict_mode(["yes", "no", "yes"]) == "yes"


def test_verdict_stability_fraction_matches_mode_frequency():
    assert verdict_stability(["yes", "no", "yes"]) == pytest.approx(2.0 / 3.0)


def test_verdict_flip_rate_counts_adjacent_changes():
    assert verdict_flip_rate(["a", "b", "b", "a"]) == pytest.approx(2.0 / 3.0)


def test_verdict_flip_rate_zero_for_short_sequences():
    assert verdict_flip_rate(["a"]) == pytest.approx(0.0)


def test_verdict_mode_rejects_empty_input():
    with pytest.raises(ValueError):
        verdict_mode([])


def test_verdict_stability_handles_empty_input():
    assert verdict_stability([]) == pytest.approx(0.0)


def test_verdict_flip_rate_zero_for_constant_sequence():
    assert verdict_flip_rate(["a", "a", "a"]) == pytest.approx(0.0)

