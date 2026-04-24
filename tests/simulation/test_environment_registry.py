"""Tests for src/simulation/environment_registry.py."""
from __future__ import annotations

import math

import pytest

from src.simulation.environment_registry import (
    DEFAULT_REGISTRY,
    ENVIRONMENT_REGISTRY_REGISTRY,
    EnvironmentRegistry,
    EnvSpec,
)


def _spec(env_id: str, tags: list[str] | None = None) -> EnvSpec:
    return EnvSpec(
        env_id=env_id,
        observation_space={"type": "Box", "shape": [2]},
        action_space={"type": "Discrete", "n": 2},
        tags=tags or [],
    )


def _factory(env_id: str):
    def make(**kwargs):
        return {"env_id": env_id, "kwargs": kwargs}

    return make


def test_envspec_is_frozen():
    s = _spec("A")
    with pytest.raises(Exception):
        s.env_id = "B"  # type: ignore[misc]


def test_envspec_defaults():
    s = _spec("A")
    assert s.max_episode_steps == 200
    assert s.reward_range == (-math.inf, math.inf)
    assert s.tags == []


def test_envspec_custom_fields():
    s = EnvSpec(
        env_id="X",
        observation_space={},
        action_space={},
        max_episode_steps=10,
        reward_range=(0.0, 1.0),
        tags=["t1"],
    )
    assert s.max_episode_steps == 10
    assert s.reward_range == (0.0, 1.0)
    assert s.tags == ["t1"]


def test_register_and_get_spec():
    r = EnvironmentRegistry()
    s = _spec("A")
    r.register(s, _factory("A"))
    assert r.get_spec("A") is s


def test_get_spec_unknown_returns_none():
    r = EnvironmentRegistry()
    assert r.get_spec("missing") is None


def test_make_calls_factory():
    r = EnvironmentRegistry()
    r.register(_spec("A"), _factory("A"))
    env = r.make("A", seed=42)
    assert env["env_id"] == "A"
    assert env["kwargs"] == {"seed": 42}


def test_make_unknown_raises_keyerror():
    r = EnvironmentRegistry()
    with pytest.raises(KeyError):
        r.make("does-not-exist")


def test_list_envs_sorted():
    r = EnvironmentRegistry()
    r.register(_spec("B"), _factory("B"))
    r.register(_spec("A"), _factory("A"))
    r.register(_spec("C"), _factory("C"))
    ids = [s.env_id for s in r.list_envs()]
    assert ids == ["A", "B", "C"]


def test_list_envs_empty():
    r = EnvironmentRegistry()
    assert r.list_envs() == []


def test_filter_by_tag_returns_matching():
    r = EnvironmentRegistry()
    r.register(_spec("A", ["x"]), _factory("A"))
    r.register(_spec("B", ["y"]), _factory("B"))
    r.register(_spec("C", ["x", "y"]), _factory("C"))
    xs = [s.env_id for s in r.filter_by_tag("x")]
    assert xs == ["A", "C"]


def test_filter_by_tag_no_match():
    r = EnvironmentRegistry()
    r.register(_spec("A", ["x"]), _factory("A"))
    assert r.filter_by_tag("z") == []


def test_filter_by_tag_empty_registry():
    r = EnvironmentRegistry()
    assert r.filter_by_tag("anything") == []


def test_default_registry_has_three_envs():
    assert len(DEFAULT_REGISTRY.list_envs()) >= 3


def test_default_registry_has_cartpole():
    assert DEFAULT_REGISTRY.get_spec("CartPole-v1") is not None


def test_default_registry_has_mountaincar():
    assert DEFAULT_REGISTRY.get_spec("MountainCar-v0") is not None


def test_default_registry_has_pendulum():
    assert DEFAULT_REGISTRY.get_spec("Pendulum-v1") is not None


def test_default_registry_cartpole_discrete():
    spec = DEFAULT_REGISTRY.get_spec("CartPole-v1")
    assert spec is not None
    assert spec.action_space["type"] == "Discrete"
    assert spec.action_space["n"] == 2


def test_default_registry_make_cartpole():
    env = DEFAULT_REGISTRY.make("CartPole-v1")
    obs = env.reset()
    assert obs["env_id"] == "CartPole-v1"


def test_default_registry_make_mountaincar_step():
    env = DEFAULT_REGISTRY.make("MountainCar-v0")
    env.reset()
    obs, r, done, info = env.step(1)
    assert obs["step"] == 1
    assert done is False
    assert info["action"] == 1


def test_default_registry_render():
    env = DEFAULT_REGISTRY.make("Pendulum-v1")
    env.reset()
    assert "Pendulum-v1" in env.render()


def test_default_registry_filter_continuous():
    continuous = DEFAULT_REGISTRY.filter_by_tag("continuous")
    assert any(s.env_id == "Pendulum-v1" for s in continuous)


def test_default_registry_filter_discrete():
    discrete = DEFAULT_REGISTRY.filter_by_tag("discrete")
    ids = {s.env_id for s in discrete}
    assert "CartPole-v1" in ids
    assert "MountainCar-v0" in ids


def test_registry_registry_has_default():
    assert "default" in ENVIRONMENT_REGISTRY_REGISTRY
    assert ENVIRONMENT_REGISTRY_REGISTRY["default"] is DEFAULT_REGISTRY


def test_register_overwrites():
    r = EnvironmentRegistry()
    r.register(_spec("A", ["old"]), _factory("A1"))
    r.register(_spec("A", ["new"]), _factory("A2"))
    spec = r.get_spec("A")
    assert spec is not None
    assert spec.tags == ["new"]


def test_make_passes_kwargs():
    r = EnvironmentRegistry()
    r.register(_spec("A"), _factory("A"))
    env = r.make("A", a=1, b=2)
    assert env["kwargs"] == {"a": 1, "b": 2}


def test_envspec_tags_default_independent():
    s1 = _spec("A")
    s2 = _spec("B")
    # frozen dataclass, but default_factory should give fresh list
    assert s1.tags is not s2.tags


def test_default_registry_cartpole_max_steps():
    spec = DEFAULT_REGISTRY.get_spec("CartPole-v1")
    assert spec is not None
    assert spec.max_episode_steps == 500


def test_default_registry_pendulum_continuous_action():
    spec = DEFAULT_REGISTRY.get_spec("Pendulum-v1")
    assert spec is not None
    assert spec.action_space["type"] == "Box"
