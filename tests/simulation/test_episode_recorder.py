"""Tests for src/simulation/episode_recorder.py — ~45 tests."""
from __future__ import annotations

import dataclasses

import pytest

from src.simulation.agent_harness import AgentHarness, Trajectory
from src.simulation.episode_recorder import EPISODE_RECORDER, Episode, EpisodeRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(success: bool = False, env_name: str = "gridworld") -> Trajectory:
    return Trajectory(env_name=env_name, steps=[], total_reward=0.0, success=success)


def _fresh_recorder() -> EpisodeRecorder:
    """Return a brand-new recorder with no episodes recorded."""
    return EpisodeRecorder()


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

class TestEpisode:
    def _make_episode(self, **kwargs):
        defaults = dict(
            env_name="gridworld",
            policy_name="random",
            trajectory_data={"env_name": "gridworld", "steps": [], "total_reward": 0.0, "success": False},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        defaults.update(kwargs)
        return Episode(**defaults)

    def test_auto_generates_id(self):
        ep = self._make_episode()
        assert isinstance(ep.id, str)
        assert len(ep.id) == 8

    def test_two_episodes_different_ids(self):
        ep1 = self._make_episode()
        ep2 = self._make_episode()
        assert ep1.id != ep2.id

    def test_env_name_stored(self):
        ep = self._make_episode(env_name="myenv")
        assert ep.env_name == "myenv"

    def test_policy_name_stored(self):
        ep = self._make_episode(policy_name="greedy")
        assert ep.policy_name == "greedy"

    def test_trajectory_data_stored(self):
        data = {"success": True}
        ep = self._make_episode(trajectory_data=data)
        assert ep.trajectory_data == data

    def test_timestamp_stored(self):
        ep = self._make_episode(timestamp="2026-01-01T00:00:00+00:00")
        assert "2026" in ep.timestamp

    def test_default_metadata_empty(self):
        ep = self._make_episode()
        assert ep.metadata == {}

    def test_metadata_not_shared(self):
        ep1 = self._make_episode()
        ep2 = self._make_episode()
        ep1.metadata["k"] = 1
        assert "k" not in ep2.metadata

    def test_custom_metadata(self):
        ep = self._make_episode()
        ep2 = Episode(
            env_name="g",
            policy_name="r",
            trajectory_data={},
            timestamp="t",
            metadata={"tag": "test"},
        )
        assert ep2.metadata["tag"] == "test"


# ---------------------------------------------------------------------------
# EpisodeRecorder.record
# ---------------------------------------------------------------------------

class TestEpisodeRecorderRecord:
    def test_record_returns_episode(self):
        rec = _fresh_recorder()
        traj = _make_trajectory()
        ep = rec.record("gridworld", "random", traj)
        assert isinstance(ep, Episode)

    def test_record_stores_env_name(self):
        rec = _fresh_recorder()
        ep = rec.record("gridworld", "random", _make_trajectory())
        assert ep.env_name == "gridworld"

    def test_record_stores_policy_name(self):
        rec = _fresh_recorder()
        ep = rec.record("gridworld", "greedy", _make_trajectory())
        assert ep.policy_name == "greedy"

    def test_record_stores_trajectory_as_dict(self):
        rec = _fresh_recorder()
        traj = _make_trajectory(success=True)
        ep = rec.record("gridworld", "random", traj)
        assert isinstance(ep.trajectory_data, dict)

    def test_record_trajectory_data_has_success(self):
        rec = _fresh_recorder()
        traj = _make_trajectory(success=True)
        ep = rec.record("gridworld", "random", traj)
        assert ep.trajectory_data["success"] is True

    def test_record_trajectory_data_equals_asdict(self):
        rec = _fresh_recorder()
        traj = _make_trajectory()
        ep = rec.record("gridworld", "random", traj)
        assert ep.trajectory_data == dataclasses.asdict(traj)

    def test_record_timestamp_is_string(self):
        rec = _fresh_recorder()
        ep = rec.record("gridworld", "random", _make_trajectory())
        assert isinstance(ep.timestamp, str)

    def test_record_stores_metadata(self):
        rec = _fresh_recorder()
        ep = rec.record("gridworld", "random", _make_trajectory(), metadata={"run": 1})
        assert ep.metadata["run"] == 1

    def test_record_increments_count(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory())
        rec.record("gridworld", "random", _make_trajectory())
        assert len(rec._episodes) == 2


# ---------------------------------------------------------------------------
# EpisodeRecorder.get
# ---------------------------------------------------------------------------

class TestEpisodeRecorderGet:
    def test_get_returns_episode_for_valid_id(self):
        rec = _fresh_recorder()
        ep = rec.record("gridworld", "random", _make_trajectory())
        assert rec.get(ep.id) is ep

    def test_get_returns_none_for_unknown(self):
        rec = _fresh_recorder()
        assert rec.get("nonexistent") is None

    def test_get_multiple_episodes(self):
        rec = _fresh_recorder()
        ep1 = rec.record("gridworld", "random", _make_trajectory())
        ep2 = rec.record("gridworld", "greedy", _make_trajectory())
        assert rec.get(ep1.id) is ep1
        assert rec.get(ep2.id) is ep2


# ---------------------------------------------------------------------------
# EpisodeRecorder.query
# ---------------------------------------------------------------------------

class TestEpisodeRecorderQuery:
    def test_query_all_no_filters(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory())
        rec.record("maze", "greedy", _make_trajectory())
        assert len(rec.query()) == 2

    def test_query_by_env_name(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory())
        rec.record("maze", "greedy", _make_trajectory())
        result = rec.query(env_name="gridworld")
        assert len(result) == 1
        assert result[0].env_name == "gridworld"

    def test_query_env_name_no_match(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory())
        assert rec.query(env_name="unknown") == []

    def test_query_success_only(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory(success=False))
        rec.record("gridworld", "greedy", _make_trajectory(success=True))
        result = rec.query(success_only=True)
        assert len(result) == 1
        assert result[0].trajectory_data["success"] is True

    def test_query_success_only_none(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory(success=False))
        assert rec.query(success_only=True) == []

    def test_query_env_and_success(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory(success=True))
        rec.record("maze", "random", _make_trajectory(success=True))
        rec.record("gridworld", "greedy", _make_trajectory(success=False))
        result = rec.query(env_name="gridworld", success_only=True)
        assert len(result) == 1
        assert result[0].env_name == "gridworld"


# ---------------------------------------------------------------------------
# EpisodeRecorder.stats
# ---------------------------------------------------------------------------

class TestEpisodeRecorderStats:
    def test_stats_has_total(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory())
        s = rec.stats()
        assert "total" in s

    def test_stats_total_correct(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "random", _make_trajectory())
        rec.record("gridworld", "random", _make_trajectory())
        assert rec.stats()["total"] == 2

    def test_stats_has_by_env(self):
        rec = _fresh_recorder()
        s = rec.stats()
        assert "by_env" in s

    def test_stats_by_env_counts(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "r", _make_trajectory())
        rec.record("maze", "r", _make_trajectory())
        rec.record("gridworld", "r", _make_trajectory())
        assert rec.stats()["by_env"]["gridworld"] == 2
        assert rec.stats()["by_env"]["maze"] == 1

    def test_stats_has_success_rate(self):
        rec = _fresh_recorder()
        s = rec.stats()
        assert "success_rate" in s

    def test_stats_success_rate_correct(self):
        rec = _fresh_recorder()
        rec.record("gridworld", "r", _make_trajectory(success=True))
        rec.record("gridworld", "r", _make_trajectory(success=False))
        assert rec.stats()["success_rate"] == pytest.approx(0.5)

    def test_stats_empty_recorder(self):
        rec = _fresh_recorder()
        s = rec.stats()
        assert s["total"] == 0
        assert s["success_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EpisodeRecorder.export_ids
# ---------------------------------------------------------------------------

class TestEpisodeRecorderExportIds:
    def test_export_ids_returns_list(self):
        rec = _fresh_recorder()
        assert isinstance(rec.export_ids(), list)

    def test_export_ids_empty_recorder(self):
        rec = _fresh_recorder()
        assert rec.export_ids() == []

    def test_export_ids_contains_recorded_ids(self):
        rec = _fresh_recorder()
        ep1 = rec.record("gridworld", "r", _make_trajectory())
        ep2 = rec.record("gridworld", "r", _make_trajectory())
        ids = rec.export_ids()
        assert ep1.id in ids
        assert ep2.id in ids

    def test_export_ids_length_matches_total(self):
        rec = _fresh_recorder()
        for _ in range(3):
            rec.record("gridworld", "r", _make_trajectory())
        assert len(rec.export_ids()) == 3


# ---------------------------------------------------------------------------
# max_episodes eviction
# ---------------------------------------------------------------------------

class TestMaxEpisodesEviction:
    def test_evicts_oldest_when_over_capacity(self):
        rec = EpisodeRecorder(max_episodes=3)
        ep1 = rec.record("gridworld", "r", _make_trajectory())
        ep2 = rec.record("gridworld", "r", _make_trajectory())
        ep3 = rec.record("gridworld", "r", _make_trajectory())
        ep4 = rec.record("gridworld", "r", _make_trajectory())  # should evict ep1
        assert rec.get(ep1.id) is None

    def test_does_not_exceed_max_capacity(self):
        rec = EpisodeRecorder(max_episodes=3)
        for _ in range(5):
            rec.record("gridworld", "r", _make_trajectory())
        assert len(rec._episodes) == 3

    def test_latest_episodes_retained(self):
        rec = EpisodeRecorder(max_episodes=2)
        rec.record("gridworld", "r", _make_trajectory())
        ep3 = rec.record("gridworld", "r", _make_trajectory())
        ep4 = rec.record("gridworld", "r", _make_trajectory())
        assert rec.get(ep3.id) is ep3
        assert rec.get(ep4.id) is ep4


# ---------------------------------------------------------------------------
# EPISODE_RECORDER singleton
# ---------------------------------------------------------------------------

class TestEpisodeRecorderSingleton:
    def test_exists(self):
        assert EPISODE_RECORDER is not None

    def test_is_instance(self):
        assert isinstance(EPISODE_RECORDER, EpisodeRecorder)
