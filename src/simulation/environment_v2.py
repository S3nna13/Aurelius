from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ActionSpace:
    discrete_n: int = 0

    def sample(self) -> int:
        return random.randint(0, max(0, self.discrete_n - 1))

    def contains(self, action: int) -> bool:
        return 0 <= action < self.discrete_n


@dataclass
class ObservationSpace:
    low: float = 0.0
    high: float = 1.0
    shape: tuple[int, ...] = (4,)

    def contains(self, obs: Sequence[float]) -> bool:
        return all(self.low <= o <= self.high for o in obs)


@dataclass
class StepResult:
    obs: list[float]
    reward: float
    done: bool
    info: dict[str, Any]


class EnvV2:
    def __init__(self) -> None:
        self.action_space = ActionSpace(discrete_n=4)
        self.observation_space = ObservationSpace(low=-1.0, high=1.0, shape=(4,))
        self._state: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._step_count = 0

    def reset(self) -> list[float]:
        self._state = [random.uniform(-0.1, 0.1) for _ in range(4)]
        self._step_count = 0
        return self._state

    def step(self, action: int) -> StepResult:
        self._step_count += 1
        if action == 0:
            self._state[0] += 0.1
        elif action == 1:
            self._state[0] -= 0.1
        elif action == 2:
            self._state[1] += 0.1
        elif action == 3:
            self._state[1] -= 0.1
        reward = -abs(self._state[0]) - abs(self._state[1])
        done = self._step_count >= 50
        return StepResult(obs=list(self._state), reward=reward, done=done, info={})


ENV_V2 = EnvV2()
