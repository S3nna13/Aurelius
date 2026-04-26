"""Tests for environment_v2 — extensible RL environment."""

from __future__ import annotations

from src.simulation.environment_v2 import ActionSpace, EnvV2, ObservationSpace, StepResult


class TestActionSpace:
    def test_discrete_actions(self):
        a = ActionSpace(discrete_n=4)
        assert a.sample() in range(4)
        assert a.contains(2)
        assert not a.contains(10)


class TestObservationSpace:
    def test_box_space(self):
        o = ObservationSpace(low=0.0, high=1.0, shape=(4,))
        assert o.contains([0.5, 0.5, 0.5, 0.5])
        assert not o.contains([5.0, 5.0, 5.0, 5.0])


class TestStepResult:
    def test_done_flag(self):
        r = StepResult(obs=[0.0], reward=1.0, done=False, info={})
        assert not r.done


class TestEnvV2:
    def test_reset_returns_observation(self):
        env = EnvV2()
        obs = env.reset()
        assert len(obs) == 4

    def test_step_returns_result(self):
        env = EnvV2()
        env.reset()
        result = env.step(0)
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, float)

    def test_random_steps_dont_crash(self):
        env = EnvV2()
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            result = env.step(action)
            if result.done:
                env.reset()

    def test_action_space_defined(self):
        env = EnvV2()
        assert env.action_space.discrete_n == 4

    def test_observation_space_defined(self):
        env = EnvV2()
        assert len(env.observation_space.shape) >= 1
