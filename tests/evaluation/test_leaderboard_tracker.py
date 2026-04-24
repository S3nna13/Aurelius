"""Tests for src/evaluation/leaderboard_tracker.py — ≥28 test cases."""

from __future__ import annotations

import time
import pytest

from src.evaluation.leaderboard_tracker import (
    Leaderboard,
    LeaderboardEntry,
    LEADERBOARD_TRACKER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lb_higher():
    """Leaderboard where higher score is better (default)."""
    return Leaderboard(metric_name="accuracy", higher_is_better=True)


@pytest.fixture
def lb_lower():
    """Leaderboard where lower score is better (e.g., loss)."""
    return Leaderboard(metric_name="loss", higher_is_better=False)


# ---------------------------------------------------------------------------
# LeaderboardEntry frozen dataclass
# ---------------------------------------------------------------------------

class TestLeaderboardEntryFrozen:
    def test_is_frozen(self):
        entry = LeaderboardEntry(
            model_id="m1", metric_name="acc", score=0.9, timestamp=1.0
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.score = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        entry = LeaderboardEntry(
            model_id="m1",
            metric_name="acc",
            score=0.9,
            timestamp=1.5,
            metadata={"lr": 0.001},
        )
        assert entry.model_id == "m1"
        assert entry.metric_name == "acc"
        assert entry.score == 0.9
        assert entry.timestamp == 1.5
        assert entry.metadata == {"lr": 0.001}

    def test_default_metadata_is_empty_dict(self):
        entry = LeaderboardEntry(model_id="m1", metric_name="acc", score=0.9, timestamp=0.0)
        assert entry.metadata == {}


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_submit_returns_leaderboard_entry(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85)
        assert isinstance(entry, LeaderboardEntry)

    def test_submit_stores_model_id(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85)
        assert entry.model_id == "modelA"

    def test_submit_stores_score(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85)
        assert entry.score == 0.85

    def test_submit_stores_metric_name(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85)
        assert entry.metric_name == "accuracy"

    def test_submit_timestamp_is_positive(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85)
        assert entry.timestamp >= 0.0

    def test_submit_metadata_stored(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85, metadata={"tag": "v1"})
        assert entry.metadata == {"tag": "v1"}

    def test_submit_none_metadata_becomes_empty_dict(self, lb_higher):
        entry = lb_higher.submit("modelA", 0.85, metadata=None)
        assert entry.metadata == {}


# ---------------------------------------------------------------------------
# rankings — higher_is_better
# ---------------------------------------------------------------------------

class TestRankingsHigherIsBetter:
    def test_rankings_sorted_descending(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m2", 0.9)
        lb_higher.submit("m3", 0.8)
        ranked = lb_higher.rankings()
        scores = [e.score for e in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rankings_returns_list(self, lb_higher):
        lb_higher.submit("m1", 0.5)
        assert isinstance(lb_higher.rankings(), list)

    def test_rankings_empty_leaderboard(self, lb_higher):
        assert lb_higher.rankings() == []

    def test_rankings_tie_break_by_timestamp_asc(self, lb_higher):
        # Submit two models with equal scores; the one submitted first should rank higher
        e1 = lb_higher.submit("early", 0.8)
        e2 = lb_higher.submit("late", 0.8)
        # Guarantee e1 timestamp <= e2 timestamp
        ranked = lb_higher.rankings()
        assert ranked[0].model_id == "early"
        assert ranked[1].model_id == "late"


# ---------------------------------------------------------------------------
# rankings — lower_is_better
# ---------------------------------------------------------------------------

class TestRankingsLowerIsBetter:
    def test_rankings_sorted_ascending(self, lb_lower):
        lb_lower.submit("m1", 0.5)
        lb_lower.submit("m2", 0.1)
        lb_lower.submit("m3", 0.3)
        ranked = lb_lower.rankings()
        scores = [e.score for e in ranked]
        assert scores == sorted(scores)

    def test_best_is_lowest_score(self, lb_lower):
        lb_lower.submit("m1", 0.5)
        lb_lower.submit("m2", 0.1)
        assert lb_lower.best().score == 0.1


# ---------------------------------------------------------------------------
# best
# ---------------------------------------------------------------------------

class TestBest:
    def test_best_returns_none_when_empty(self, lb_higher):
        assert lb_higher.best() is None

    def test_best_is_highest_score(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m2", 0.95)
        lb_higher.submit("m3", 0.8)
        assert lb_higher.best().score == 0.95

    def test_best_is_leaderboard_entry(self, lb_higher):
        lb_higher.submit("m1", 0.9)
        assert isinstance(lb_higher.best(), LeaderboardEntry)


# ---------------------------------------------------------------------------
# rank_of
# ---------------------------------------------------------------------------

class TestRankOf:
    def test_rank_of_top_model_is_one(self, lb_higher):
        lb_higher.submit("m1", 0.9)
        lb_higher.submit("m2", 0.7)
        assert lb_higher.rank_of("m1") == 1

    def test_rank_of_second_model_is_two(self, lb_higher):
        lb_higher.submit("m1", 0.9)
        lb_higher.submit("m2", 0.7)
        assert lb_higher.rank_of("m2") == 2

    def test_rank_of_missing_model_is_none(self, lb_higher):
        lb_higher.submit("m1", 0.9)
        assert lb_higher.rank_of("unknown") is None

    def test_rank_of_empty_leaderboard_is_none(self, lb_higher):
        assert lb_higher.rank_of("m1") is None


# ---------------------------------------------------------------------------
# history_of
# ---------------------------------------------------------------------------

class TestHistoryOf:
    def test_history_of_single_submission(self, lb_higher):
        lb_higher.submit("m1", 0.8)
        history = lb_higher.history_of("m1")
        assert len(history) == 1

    def test_history_of_multiple_submissions(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m1", 0.8)
        lb_higher.submit("m1", 0.9)
        history = lb_higher.history_of("m1")
        assert len(history) == 3

    def test_history_of_sorted_by_timestamp(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m1", 0.9)
        history = lb_higher.history_of("m1")
        assert history[0].timestamp <= history[1].timestamp

    def test_history_of_unknown_model_is_empty(self, lb_higher):
        lb_higher.submit("m1", 0.8)
        assert lb_higher.history_of("unknown") == []

    def test_history_of_returns_only_requested_model(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m2", 0.8)
        history = lb_higher.history_of("m1")
        assert all(e.model_id == "m1" for e in history)


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_len_zero_when_empty(self, lb_higher):
        assert len(lb_higher) == 0

    def test_len_counts_unique_models(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m2", 0.8)
        assert len(lb_higher) == 2

    def test_len_multiple_submissions_same_model_count_once(self, lb_higher):
        lb_higher.submit("m1", 0.7)
        lb_higher.submit("m1", 0.9)
        assert len(lb_higher) == 1


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in LEADERBOARD_TRACKER_REGISTRY

    def test_registry_default_is_leaderboard_class(self):
        assert LEADERBOARD_TRACKER_REGISTRY["default"] is Leaderboard

    def test_registry_default_instantiable(self):
        cls = LEADERBOARD_TRACKER_REGISTRY["default"]
        instance = cls(metric_name="f1")
        assert isinstance(instance, Leaderboard)
