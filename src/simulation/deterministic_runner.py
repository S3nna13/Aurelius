"""Deterministic simulation runner with fixed seed for reproducibility."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SimStep:
    state: dict[str, Any]
    action: str
    reward: float
    done: bool


@dataclass
class DeterministicRunner:
    """Run simulations with determinism via fixed seed."""

    seed: int = 42
    _rng: random.Random = field(default_factory=lambda: random.Random(42))

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed if self.seed is not None else 42)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed if seed is not None else self.seed)

    def run(self, env_fn: Callable[[], Any], steps: int = 100) -> list[SimStep]:
        env = env_fn()
        state = env
        history = []
        for _ in range(steps):
            action = self._rng.choice(["left", "right", "up", "down", "stay"])
            reward = self._rng.random()
            done = self._rng.random() < 0.1
            history.append(SimStep(state={"pos": 0}, action=action, reward=reward, done=done))
            if done:
                break
        return history


DETERMINISTIC_RUNNER = DeterministicRunner()