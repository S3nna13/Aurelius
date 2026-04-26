"""
Unit tests for src/training/curriculum_rl.py

Tests:
  1.  test_config_defaults
  2.  test_register_task
  3.  test_register_duplicate_raises
  4.  test_update_correct
  5.  test_update_incorrect
  6.  test_update_unknown_raises
  7.  test_in_learning_zone_new_task
  8.  test_in_learning_zone_easy
  9.  test_in_learning_zone_hard
  10. test_in_learning_zone_medium
  11. test_sample_count
  12. test_sample_valid_ids
  13. test_sample_prefers_zone
  14. test_statistics_keys
  15. test_statistics_n_in_zone
"""

import pytest

from src.training.curriculum_rl import (
    CurriculumRLConfig,
    CurriculumRLSampler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sampler(**cfg_kwargs) -> CurriculumRLSampler:
    """Return a sampler with the given config overrides."""
    cfg = CurriculumRLConfig(**cfg_kwargs)
    return CurriculumRLSampler(cfg)


def _register_n(sampler: CurriculumRLSampler, n: int, base_difficulty: float = 0.5):
    """Register *n* tasks named task_0 … task_{n-1}."""
    for i in range(n):
        sampler.register_task(f"task_{i}", difficulty=base_difficulty)


def _drive_to_accuracy(
    sampler: CurriculumRLSampler, task_id: str, target_accuracy: float, n_steps: int = 200
) -> None:
    """Feed the sampler enough updates so recent_accuracy converges near target_accuracy."""
    for _ in range(n_steps):
        is_correct = target_accuracy >= 0.5  # deterministic shortcut
        sampler.update(task_id, is_correct=is_correct)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CurriculumRLConfig()
    assert cfg.easy_threshold == 0.85
    assert cfg.hard_threshold == 0.15
    assert cfg.exploration_prob == 0.1
    assert cfg.temperature == 1.0
    assert cfg.min_attempts_before_skip == 5


# ---------------------------------------------------------------------------
# 2. Register task
# ---------------------------------------------------------------------------


def test_register_task():
    sampler = _make_sampler()
    sampler.register_task("task_a", difficulty=0.3)
    stats = sampler.statistics()
    assert stats["n_tasks"] == 1


# ---------------------------------------------------------------------------
# 3. Register duplicate raises ValueError
# ---------------------------------------------------------------------------


def test_register_duplicate_raises():
    sampler = _make_sampler()
    sampler.register_task("task_a", difficulty=0.3)
    with pytest.raises(ValueError, match="already registered"):
        sampler.register_task("task_a", difficulty=0.7)


# ---------------------------------------------------------------------------
# 4. Update correct increases accuracy
# ---------------------------------------------------------------------------


def test_update_correct():
    sampler = _make_sampler()
    sampler.register_task("task_a", difficulty=0.5)
    initial = sampler.task_summary("task_a")["accuracy"]
    sampler.update("task_a", is_correct=True)
    after = sampler.task_summary("task_a")["accuracy"]
    assert after > initial


# ---------------------------------------------------------------------------
# 5. Update incorrect decreases accuracy
# ---------------------------------------------------------------------------


def test_update_incorrect():
    sampler = _make_sampler()
    sampler.register_task("task_a", difficulty=0.5)
    initial = sampler.task_summary("task_a")["accuracy"]
    sampler.update("task_a", is_correct=False)
    after = sampler.task_summary("task_a")["accuracy"]
    assert after < initial


# ---------------------------------------------------------------------------
# 6. Update unknown task raises KeyError
# ---------------------------------------------------------------------------


def test_update_unknown_raises():
    sampler = _make_sampler()
    with pytest.raises(KeyError, match="not registered"):
        sampler.update("nonexistent", is_correct=True)


# ---------------------------------------------------------------------------
# 7. In learning zone — new task (not enough attempts)
# ---------------------------------------------------------------------------


def test_in_learning_zone_new_task():
    sampler = _make_sampler(min_attempts_before_skip=5)
    sampler.register_task("task_a", difficulty=0.5)
    # 0 attempts < 5 → always in zone
    assert sampler.in_learning_zone("task_a") is True
    # Even after 4 attempts still in zone
    for _ in range(4):
        sampler.update("task_a", is_correct=True)
    assert sampler.in_learning_zone("task_a") is True


# ---------------------------------------------------------------------------
# 8. Not in learning zone — task is "easy"
# ---------------------------------------------------------------------------


def test_in_learning_zone_easy():
    sampler = _make_sampler(
        easy_threshold=0.85,
        hard_threshold=0.15,
        min_attempts_before_skip=5,
    )
    sampler.register_task("task_a", difficulty=0.5)
    # Drive accuracy very high (all correct, many steps)
    for _ in range(100):
        sampler.update("task_a", is_correct=True)
    summary = sampler.task_summary("task_a")
    assert summary["accuracy"] > 0.85, "Expected high accuracy after many correct updates"
    assert sampler.in_learning_zone("task_a") is False


# ---------------------------------------------------------------------------
# 9. Not in learning zone — task is "hard"
# ---------------------------------------------------------------------------


def test_in_learning_zone_hard():
    sampler = _make_sampler(
        easy_threshold=0.85,
        hard_threshold=0.15,
        min_attempts_before_skip=5,
    )
    sampler.register_task("task_a", difficulty=0.5)
    # Drive accuracy very low (all incorrect, many steps)
    for _ in range(100):
        sampler.update("task_a", is_correct=False)
    summary = sampler.task_summary("task_a")
    assert summary["accuracy"] < 0.15, "Expected low accuracy after many incorrect updates"
    assert sampler.in_learning_zone("task_a") is False


# ---------------------------------------------------------------------------
# 10. In learning zone — accuracy in medium range
# ---------------------------------------------------------------------------


def test_in_learning_zone_medium():
    sampler = _make_sampler(
        easy_threshold=0.85,
        hard_threshold=0.15,
        min_attempts_before_skip=5,
    )
    sampler.register_task("task_a", difficulty=0.5)
    # Alternating correct/incorrect: EMA stays near 0.5
    for _ in range(50):
        sampler.update("task_a", is_correct=True)
        sampler.update("task_a", is_correct=False)
    summary = sampler.task_summary("task_a")
    assert 0.15 <= summary["accuracy"] <= 0.85
    assert sampler.in_learning_zone("task_a") is True


# ---------------------------------------------------------------------------
# 11. sample(n) returns exactly n items
# ---------------------------------------------------------------------------


def test_sample_count():
    sampler = _make_sampler()
    _register_n(sampler, 5)
    result = sampler.sample(n=5, rng_seed=0)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# 12. Sampled IDs are all registered
# ---------------------------------------------------------------------------


def test_sample_valid_ids():
    sampler = _make_sampler()
    _register_n(sampler, 8)
    registered = {f"task_{i}" for i in range(8)}
    result = sampler.sample(n=20, rng_seed=42)
    for tid in result:
        assert tid in registered, f"Unexpected task_id: {tid}"


# ---------------------------------------------------------------------------
# 13. Sampler prefers learning-zone tasks (≥90% of draws from zone)
# ---------------------------------------------------------------------------


def test_sample_prefers_zone():
    # exploration_prob=0 so every draw is from learning zone
    sampler = _make_sampler(
        exploration_prob=0.0,
        easy_threshold=0.85,
        hard_threshold=0.15,
        min_attempts_before_skip=5,
    )
    # Register 10 zone tasks and 5 easy tasks
    for i in range(10):
        sampler.register_task(f"zone_{i}", difficulty=0.5)
    for i in range(5):
        tid = f"easy_{i}"
        sampler.register_task(tid, difficulty=0.5)
        for _ in range(100):
            sampler.update(tid, is_correct=True)

    zone_ids = {f"zone_{i}" for i in range(10)}
    n_samples = 200
    results = sampler.sample(n=n_samples, rng_seed=7)
    zone_hits = sum(1 for r in results if r in zone_ids)
    # With exploration_prob=0, all draws should be from zone tasks
    assert zone_hits == n_samples, f"Expected {n_samples} zone hits, got {zone_hits}"


# ---------------------------------------------------------------------------
# 14. statistics() returns correct keys
# ---------------------------------------------------------------------------


def test_statistics_keys():
    sampler = _make_sampler()
    _register_n(sampler, 3)
    stats = sampler.statistics()
    expected_keys = {"n_tasks", "n_in_zone", "mean_accuracy", "n_easy", "n_hard"}
    assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 15. statistics() n_in_zone counts correctly
# ---------------------------------------------------------------------------


def test_statistics_n_in_zone():
    sampler = _make_sampler(
        easy_threshold=0.85,
        hard_threshold=0.15,
        min_attempts_before_skip=5,
    )
    # 3 zone tasks (medium accuracy / few attempts)
    for i in range(3):
        sampler.register_task(f"zone_{i}", difficulty=0.5)

    # 2 easy tasks
    for i in range(2):
        tid = f"easy_{i}"
        sampler.register_task(tid, difficulty=0.5)
        for _ in range(100):
            sampler.update(tid, is_correct=True)

    # 1 hard task
    sampler.register_task("hard_0", difficulty=0.5)
    for _ in range(100):
        sampler.update("hard_0", is_correct=False)

    stats = sampler.statistics()
    assert stats["n_tasks"] == 6
    # zone tasks (few attempts) + nothing else qualifies
    assert stats["n_in_zone"] == 3
    assert stats["n_easy"] == 2
    assert stats["n_hard"] == 1
