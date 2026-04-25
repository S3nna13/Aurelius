"""Tests for src/simulation/curriculum_env.py — 10+ tests."""
import pytest
from src.simulation.curriculum_env import CurriculumLevel, CurriculumEnv, _default_grid_factory
from src.simulation.environment import EnvAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _levels(n=3, threshold=0.8):
    return [
        CurriculumLevel(
            level_id=i,
            config={"width": 3 + i, "height": 3 + i},
            success_threshold=threshold,
        )
        for i in range(n)
    ]


def _mark_successes(env: CurriculumEnv, n: int):
    for _ in range(n):
        env.record_episode_result(True)


def _mark_failures(env: CurriculumEnv, n: int):
    for _ in range(n):
        env.record_episode_result(False)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

def test_initial_level_is_zero():
    env = CurriculumEnv(_levels())
    assert env.current_level.level_id == 0


def test_raises_on_empty_levels():
    with pytest.raises(ValueError):
        CurriculumEnv([])


# ---------------------------------------------------------------------------
# 2. Success rate
# ---------------------------------------------------------------------------

def test_success_rate_starts_zero():
    env = CurriculumEnv(_levels())
    assert env.success_rate() == pytest.approx(0.0)


def test_success_rate_all_success():
    env = CurriculumEnv(_levels())
    _mark_successes(env, 10)
    assert env.success_rate() == pytest.approx(1.0)


def test_success_rate_mixed():
    env = CurriculumEnv(_levels(), window=10)
    _mark_successes(env, 8)
    _mark_failures(env, 2)
    assert env.success_rate() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 3. Advancement
# ---------------------------------------------------------------------------

def test_no_advance_below_threshold():
    env = CurriculumEnv(_levels(threshold=0.8), window=10)
    _mark_successes(env, 7)
    _mark_failures(env, 3)  # 70% < 80%
    advanced = env.try_advance()
    assert advanced is False
    assert env.current_level.level_id == 0


def test_advance_at_threshold():
    env = CurriculumEnv(_levels(threshold=0.8), window=10)
    _mark_successes(env, 8)
    _mark_failures(env, 2)  # 80% == threshold
    advanced = env.try_advance()
    assert advanced is True
    assert env.current_level.level_id == 1


def test_advance_clears_history():
    env = CurriculumEnv(_levels(threshold=0.8), window=10)
    _mark_successes(env, 10)
    env.try_advance()
    assert env.success_rate() == pytest.approx(0.0)


def test_no_advance_on_final_level():
    env = CurriculumEnv(_levels(n=1))
    _mark_successes(env, 20)
    advanced = env.try_advance()
    assert advanced is False
    assert env.is_final_level is True


def test_advance_increments_level_sequentially():
    env = CurriculumEnv(_levels(n=3, threshold=0.8), window=10)
    for expected_id in [0, 1, 2]:
        assert env.current_level.level_id == expected_id
        if not env.is_final_level:
            _mark_successes(env, 10)
            env.try_advance()


# ---------------------------------------------------------------------------
# 4. Env delegation
# ---------------------------------------------------------------------------

def test_reset_delegates_to_inner():
    env = CurriculumEnv(_levels())
    state = env.reset()
    assert "x" in state.obs and "y" in state.obs


def test_step_delegates_to_inner():
    env = CurriculumEnv(_levels())
    state = env.reset()
    action = EnvAction(action_id="right")
    step = env.step(state, action)
    assert step.next_state is not None


def test_valid_actions_delegates():
    env = CurriculumEnv(_levels())
    state = env.reset()
    actions = env.valid_actions(state)
    assert len(actions) == 4


# ---------------------------------------------------------------------------
# 5. Default factory
# ---------------------------------------------------------------------------

def test_default_grid_factory_creates_env():
    inner = _default_grid_factory({"width": 4, "height": 4})
    state = inner.reset()
    assert state.obs["x"] == 0 and state.obs["y"] == 0
