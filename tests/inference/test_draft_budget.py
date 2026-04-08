"""Tests for draft budget helpers."""

import pytest

from src.inference.draft_budget import (
    accepted_tokens_per_round,
    allocate_draft_steps,
    expected_speedup,
    remaining_budget,
)


def test_allocate_draft_steps_rounds_up():
    assert allocate_draft_steps(10, 3) == 4


def test_allocate_draft_steps_zero_budget():
    assert allocate_draft_steps(0, 3) == 0


def test_accepted_tokens_per_round_scales_with_acceptance():
    assert accepted_tokens_per_round(0.5, 4) == pytest.approx(2.0)


def test_expected_speedup_at_least_one():
    assert expected_speedup(0.0, 4) == pytest.approx(1.0)


def test_remaining_budget_clamps_at_zero():
    assert remaining_budget(5, 8) == 0


def test_accepted_tokens_per_round_rejects_bad_acceptance():
    with pytest.raises(ValueError):
        accepted_tokens_per_round(1.5, 4)


def test_allocate_draft_steps_rejects_bad_width():
    with pytest.raises(ValueError):
        allocate_draft_steps(5, 0)
