"""Tests for src/federation/privacy_budget_tracker.py."""

from __future__ import annotations

import pytest

from src.federation.privacy_budget_tracker import (
    BudgetRecord,
    PrivacyBudget,
    PrivacyBudgetTracker,
    PRIVACY_BUDGET_TRACKER_REGISTRY,
)


# ---------------------------------------------------------------------------
# PrivacyBudget frozen dataclass
# ---------------------------------------------------------------------------

class TestPrivacyBudget:
    def test_fields_stored(self):
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        assert budget.total_epsilon == 1.0
        assert budget.total_delta == 1e-5

    def test_frozen_epsilon(self):
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        with pytest.raises((AttributeError, TypeError)):
            budget.total_epsilon = 2.0  # type: ignore[misc]

    def test_frozen_delta(self):
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        with pytest.raises((AttributeError, TypeError)):
            budget.total_delta = 2e-5  # type: ignore[misc]

    def test_equality(self):
        b1 = PrivacyBudget(total_epsilon=2.0, total_delta=1e-6)
        b2 = PrivacyBudget(total_epsilon=2.0, total_delta=1e-6)
        assert b1 == b2


# ---------------------------------------------------------------------------
# BudgetRecord frozen dataclass
# ---------------------------------------------------------------------------

class TestBudgetRecord:
    def test_fields_stored(self):
        rec = BudgetRecord(
            round_idx=0,
            epsilon_used=0.1,
            delta_used=1e-6,
            cumulative_epsilon=0.1,
            cumulative_delta=1e-6,
        )
        assert rec.round_idx == 0
        assert rec.epsilon_used == 0.1
        assert rec.delta_used == 1e-6
        assert rec.cumulative_epsilon == 0.1
        assert rec.cumulative_delta == 1e-6

    def test_frozen_round_idx(self):
        rec = BudgetRecord(
            round_idx=1, epsilon_used=0.1, delta_used=1e-6,
            cumulative_epsilon=0.1, cumulative_delta=1e-6,
        )
        with pytest.raises((AttributeError, TypeError)):
            rec.round_idx = 99  # type: ignore[misc]

    def test_frozen_epsilon_used(self):
        rec = BudgetRecord(
            round_idx=0, epsilon_used=0.2, delta_used=1e-6,
            cumulative_epsilon=0.2, cumulative_delta=1e-6,
        )
        with pytest.raises((AttributeError, TypeError)):
            rec.epsilon_used = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PrivacyBudgetTracker — consume
# ---------------------------------------------------------------------------

class TestConsume:
    def test_consume_returns_budget_record(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        rec = tracker.consume(0, 0.1, 1e-5)
        assert isinstance(rec, BudgetRecord)

    def test_consume_round_idx_stored(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        rec = tracker.consume(3, 0.1, 1e-5)
        assert rec.round_idx == 3

    def test_consume_epsilon_used_stored(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        rec = tracker.consume(0, 0.25, 1e-6)
        assert abs(rec.epsilon_used - 0.25) < 1e-12

    def test_consume_cumulative_epsilon_grows(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(2.0, 1e-3))
        tracker.consume(0, 0.3, 1e-5)
        rec = tracker.consume(1, 0.4, 1e-5)
        assert abs(rec.cumulative_epsilon - 0.7) < 1e-12

    def test_consume_cumulative_delta_grows(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(2.0, 1e-3))
        tracker.consume(0, 0.1, 1e-5)
        rec = tracker.consume(1, 0.1, 2e-5)
        assert abs(rec.cumulative_delta - 3e-5) < 1e-15

    def test_consume_exceed_epsilon_raises_value_error(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(total_epsilon=0.5, total_delta=1.0))
        tracker.consume(0, 0.4, 0.0)
        with pytest.raises(ValueError):
            tracker.consume(1, 0.2, 0.0)  # 0.6 > 0.5

    def test_consume_exceed_delta_raises_value_error(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(total_epsilon=10.0, total_delta=1e-5))
        tracker.consume(0, 0.1, 8e-6)
        with pytest.raises(ValueError):
            tracker.consume(1, 0.1, 5e-6)  # 13e-6 > 1e-5

    def test_consume_exact_budget_no_error(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(total_epsilon=1.0, total_delta=1e-5))
        rec = tracker.consume(0, 1.0, 1e-5)
        assert abs(rec.cumulative_epsilon - 1.0) < 1e-12

    def test_consume_multiple_rounds_records(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(5.0, 1.0))
        for i in range(4):
            tracker.consume(i, 0.5, 0.1)
        assert len(tracker.history()) == 4


# ---------------------------------------------------------------------------
# remaining
# ---------------------------------------------------------------------------

class TestRemaining:
    def test_remaining_initial_is_total(self):
        budget = PrivacyBudget(1.0, 1e-5)
        tracker = PrivacyBudgetTracker(budget)
        rem_eps, rem_delta = tracker.remaining()
        assert abs(rem_eps - 1.0) < 1e-12
        assert abs(rem_delta - 1e-5) < 1e-15

    def test_remaining_decreases_after_consume(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        tracker.consume(0, 0.3, 1e-5)
        rem_eps, rem_delta = tracker.remaining()
        assert abs(rem_eps - 0.7) < 1e-12
        assert abs(rem_delta - 9e-5) < 1e-15

    def test_remaining_returns_tuple(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        result = tracker.remaining()
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# is_exhausted
# ---------------------------------------------------------------------------

class TestIsExhausted:
    def test_not_exhausted_initially(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-5))
        assert tracker.is_exhausted() is False

    def test_exhausted_when_epsilon_zero(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-3))
        tracker.consume(0, 1.0, 0.0)
        assert tracker.is_exhausted() is True

    def test_exhausted_when_delta_zero(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(10.0, 1e-5))
        tracker.consume(0, 0.1, 1e-5)
        assert tracker.is_exhausted() is True

    def test_not_exhausted_with_budget_remaining(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(2.0, 1e-3))
        tracker.consume(0, 0.5, 1e-5)
        assert tracker.is_exhausted() is False


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_empty_initially(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-5))
        assert tracker.history() == []

    def test_history_length_matches_rounds(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(5.0, 1.0))
        for i in range(3):
            tracker.consume(i, 0.1, 0.01)
        assert len(tracker.history()) == 3

    def test_history_returns_list(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1.0))
        assert isinstance(tracker.history(), list)

    def test_history_is_copy(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(5.0, 1.0))
        tracker.consume(0, 0.1, 0.1)
        hist1 = tracker.history()
        tracker.consume(1, 0.1, 0.1)
        hist2 = tracker.history()
        assert len(hist1) == 1
        assert len(hist2) == 2

    def test_history_records_are_budget_records(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(5.0, 1.0))
        tracker.consume(0, 0.1, 0.01)
        assert all(isinstance(r, BudgetRecord) for r in tracker.history())


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_has_all_keys(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-5))
        summary = tracker.summary()
        expected_keys = {
            "total_epsilon", "used_epsilon", "remaining_epsilon",
            "total_delta", "used_delta", "remaining_delta", "rounds",
        }
        assert expected_keys.issubset(summary.keys())

    def test_summary_total_epsilon(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(2.5, 1e-4))
        assert tracker.summary()["total_epsilon"] == 2.5

    def test_summary_used_epsilon_after_consume(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        tracker.consume(0, 0.3, 1e-5)
        assert abs(tracker.summary()["used_epsilon"] - 0.3) < 1e-12

    def test_summary_remaining_epsilon(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(1.0, 1e-4))
        tracker.consume(0, 0.4, 1e-5)
        assert abs(tracker.summary()["remaining_epsilon"] - 0.6) < 1e-12

    def test_summary_rounds_count(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(5.0, 1.0))
        for i in range(4):
            tracker.consume(i, 0.1, 0.01)
        assert tracker.summary()["rounds"] == 4

    def test_summary_used_delta(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(10.0, 1e-3))
        tracker.consume(0, 0.1, 2e-4)
        assert abs(tracker.summary()["used_delta"] - 2e-4) < 1e-15

    def test_summary_remaining_delta(self):
        tracker = PrivacyBudgetTracker(PrivacyBudget(10.0, 1e-3))
        tracker.consume(0, 0.1, 2e-4)
        assert abs(tracker.summary()["remaining_delta"] - 8e-4) < 1e-15


# ---------------------------------------------------------------------------
# PRIVACY_BUDGET_TRACKER_REGISTRY
# ---------------------------------------------------------------------------

class TestPrivacyBudgetTrackerRegistry:
    def test_registry_exists(self):
        assert PRIVACY_BUDGET_TRACKER_REGISTRY is not None

    def test_registry_has_default(self):
        assert "default" in PRIVACY_BUDGET_TRACKER_REGISTRY

    def test_registry_default_is_class(self):
        assert PRIVACY_BUDGET_TRACKER_REGISTRY["default"] is PrivacyBudgetTracker

    def test_registry_default_callable(self):
        cls = PRIVACY_BUDGET_TRACKER_REGISTRY["default"]
        instance = cls(PrivacyBudget(1.0, 1e-5))
        assert isinstance(instance, PrivacyBudgetTracker)
