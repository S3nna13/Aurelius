"""
Integration test for CurriculumRLSampler.

Verifies:
  - Register 10 tasks with varying difficulties.
  - Drive some tasks to "easy" (high accuracy) and some to "hard" (low accuracy).
  - sample(50, rng_seed=...) returns only registered IDs.
  - statistics() zones are consistent with manual expectation.
  - TRAINING_REGISTRY["curriculum_rl"] == CurriculumRLSampler.
"""

import pytest

from src.training import TRAINING_REGISTRY
from src.training.curriculum_rl import CurriculumRLConfig, CurriculumRLSampler

# ---------------------------------------------------------------------------
# Fixture: fully-wired sampler with 10 tasks
# ---------------------------------------------------------------------------


@pytest.fixture
def wired_sampler():
    """
    10 tasks:
      - task_0 .. task_3  → driven to easy (very high accuracy)
      - task_4 .. task_5  → driven to hard (very low accuracy)
      - task_6 .. task_9  → left in learning zone (medium accuracy)
    """
    cfg = CurriculumRLConfig(
        easy_threshold=0.85,
        hard_threshold=0.15,
        exploration_prob=0.1,
        temperature=1.0,
        min_attempts_before_skip=5,
    )
    sampler = CurriculumRLSampler(cfg)

    # Varying static difficulties
    difficulties = [i / 9.0 for i in range(10)]  # 0.0 .. 1.0
    for i in range(10):
        sampler.register_task(f"task_{i}", difficulty=difficulties[i])

    # Drive task_0..task_3 to easy
    for i in range(4):
        for _ in range(150):
            sampler.update(f"task_{i}", is_correct=True)

    # Drive task_4..task_5 to hard
    for i in range(4, 6):
        for _ in range(150):
            sampler.update(f"task_{i}", is_correct=False)

    # task_6..task_9 stay near initial accuracy (0.5) — just a few balanced updates
    for i in range(6, 10):
        for _ in range(30):
            sampler.update(f"task_{i}", is_correct=True)
            sampler.update(f"task_{i}", is_correct=False)

    return sampler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_integration_sampled_ids_are_registered(wired_sampler):
    """All 50 sampled task IDs must be among the 10 registered tasks."""
    registered = {f"task_{i}" for i in range(10)}
    results = wired_sampler.sample(50, rng_seed=99)
    assert len(results) == 50
    for tid in results:
        assert tid in registered, f"Unexpected task_id in sample: {tid!r}"


def test_integration_statistics_zones(wired_sampler):
    """Statistics must reflect the 4 easy, 2 hard, 4 zone tasks."""
    stats = wired_sampler.statistics()

    assert stats["n_tasks"] == 10
    assert stats["n_easy"] == 4, f"Expected 4 easy tasks, got {stats['n_easy']}"
    assert stats["n_hard"] == 2, f"Expected 2 hard tasks, got {stats['n_hard']}"
    assert stats["n_in_zone"] == 4, f"Expected 4 tasks in learning zone, got {stats['n_in_zone']}"
    assert 0.0 <= stats["mean_accuracy"] <= 1.0


def test_integration_task_summaries_consistent(wired_sampler):
    """task_summary in_learning_zone must agree with in_learning_zone()."""
    for i in range(10):
        tid = f"task_{i}"
        summary = wired_sampler.task_summary(tid)
        assert summary["in_learning_zone"] == wired_sampler.in_learning_zone(tid)
        assert 0.0 <= summary["accuracy"] <= 1.0
        assert summary["attempts"] > 0 or i >= 6  # all tasks were updated
        assert 0.0 <= summary["difficulty"] <= 1.0


def test_integration_sample_reproducible(wired_sampler):
    """Same rng_seed must produce identical sequences."""
    result_a = wired_sampler.sample(50, rng_seed=123)
    result_b = wired_sampler.sample(50, rng_seed=123)
    assert result_a == result_b


def test_integration_easy_tasks_not_in_zone(wired_sampler):
    """task_0..task_3 should be marked easy (above easy_threshold)."""
    cfg = wired_sampler.config
    for i in range(4):
        tid = f"task_{i}"
        summary = wired_sampler.task_summary(tid)
        assert summary["accuracy"] > cfg.easy_threshold, (
            f"{tid}: expected easy accuracy, got {summary['accuracy']:.4f}"
        )
        assert summary["in_learning_zone"] is False


def test_integration_hard_tasks_not_in_zone(wired_sampler):
    """task_4..task_5 should be marked hard (below hard_threshold)."""
    cfg = wired_sampler.config
    for i in range(4, 6):
        tid = f"task_{i}"
        summary = wired_sampler.task_summary(tid)
        assert summary["accuracy"] < cfg.hard_threshold, (
            f"{tid}: expected hard accuracy, got {summary['accuracy']:.4f}"
        )
        assert summary["in_learning_zone"] is False


def test_integration_zone_tasks_in_zone(wired_sampler):
    """task_6..task_9 should remain in the learning zone."""
    for i in range(6, 10):
        tid = f"task_{i}"
        assert wired_sampler.in_learning_zone(tid), (
            f"{tid} should be in learning zone; summary={wired_sampler.task_summary(tid)}"
        )


def test_integration_registry_wired():
    """TRAINING_REGISTRY['curriculum_rl'] must be CurriculumRLSampler."""
    assert "curriculum_rl" in TRAINING_REGISTRY
    assert TRAINING_REGISTRY["curriculum_rl"] is CurriculumRLSampler


def test_integration_registry_instantiable():
    """The registry entry can be instantiated with default config."""
    cls = TRAINING_REGISTRY["curriculum_rl"]
    sampler = cls()
    sampler.register_task("t0", difficulty=0.5)
    sampler.update("t0", is_correct=True)
    result = sampler.sample(3, rng_seed=0)
    assert len(result) == 3
    assert all(r == "t0" for r in result)
