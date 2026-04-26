"""Registry for simulation environments."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    observation_space: dict
    action_space: dict
    max_episode_steps: int = 200
    reward_range: tuple[float, float] = (-math.inf, math.inf)
    tags: list[str] = field(default_factory=list)


EnvFactory = Callable[..., Any]


class _MockEnv:
    """Lightweight mock environment used by the default registry."""

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        self.env_id = env_id
        self.kwargs = kwargs
        self._steps = 0

    def reset(self) -> dict:
        self._steps = 0
        return {"env_id": self.env_id, "step": 0}

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        self._steps += 1
        return (
            {"env_id": self.env_id, "step": self._steps},
            0.0,
            False,
            {"action": action},
        )

    def render(self) -> str:
        return f"<MockEnv {self.env_id} step={self._steps}>"


def _mock_factory(env_id: str) -> EnvFactory:
    def factory(**kwargs: Any) -> _MockEnv:
        return _MockEnv(env_id, **kwargs)

    return factory


class EnvironmentRegistry:
    """Registry keyed by env_id holding EnvSpec + factory."""

    def __init__(self) -> None:
        self._specs: dict[str, EnvSpec] = {}
        self._factories: dict[str, EnvFactory] = {}

    def register(self, spec: EnvSpec, factory: EnvFactory) -> None:
        self._specs[spec.env_id] = spec
        self._factories[spec.env_id] = factory

    def make(self, env_id: str, **kwargs: Any) -> Any:
        if env_id not in self._factories:
            raise KeyError(f"Unknown env_id: {env_id}")
        return self._factories[env_id](**kwargs)

    def list_envs(self) -> list[EnvSpec]:
        return [self._specs[k] for k in sorted(self._specs)]

    def filter_by_tag(self, tag: str) -> list[EnvSpec]:
        return [s for s in self.list_envs() if tag in s.tags]

    def get_spec(self, env_id: str) -> EnvSpec | None:
        return self._specs.get(env_id)


DEFAULT_REGISTRY = EnvironmentRegistry()

DEFAULT_REGISTRY.register(
    EnvSpec(
        env_id="CartPole-v1",
        observation_space={
            "type": "Box",
            "shape": [4],
            "low": -math.inf,
            "high": math.inf,
        },
        action_space={"type": "Discrete", "n": 2},
        max_episode_steps=500,
        reward_range=(0.0, 1.0),
        tags=["classic_control", "discrete"],
    ),
    _mock_factory("CartPole-v1"),
)

DEFAULT_REGISTRY.register(
    EnvSpec(
        env_id="MountainCar-v0",
        observation_space={
            "type": "Box",
            "shape": [2],
            "low": -1.2,
            "high": 0.6,
        },
        action_space={"type": "Discrete", "n": 3},
        max_episode_steps=200,
        reward_range=(-1.0, 0.0),
        tags=["classic_control", "discrete"],
    ),
    _mock_factory("MountainCar-v0"),
)

DEFAULT_REGISTRY.register(
    EnvSpec(
        env_id="Pendulum-v1",
        observation_space={
            "type": "Box",
            "shape": [3],
            "low": -8.0,
            "high": 8.0,
        },
        action_space={"type": "Box", "shape": [1], "low": -2.0, "high": 2.0},
        max_episode_steps=200,
        reward_range=(-16.2736044, 0.0),
        tags=["classic_control", "continuous"],
    ),
    _mock_factory("Pendulum-v1"),
)


ENVIRONMENT_REGISTRY_REGISTRY = {"default": DEFAULT_REGISTRY}
