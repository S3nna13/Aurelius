"""Rollout / replay buffer for RL experience tuples."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class RolloutBuffer:
    """Fixed-capacity buffer backed by collections.deque."""

    def __init__(self, maxlen: int = 10000) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = maxlen
        self._buffer: deque[Experience] = deque(maxlen=maxlen)

    def push(self, exp: Experience) -> None:
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self._buffer)}"
            )
        return random.sample(list(self._buffer), batch_size)

    def can_sample(self, batch_size: int) -> bool:
        return batch_size > 0 and len(self._buffer) >= batch_size

    def compute_episode_returns(
        self, experiences: list[Experience], gamma: float = 0.99
    ) -> list[float]:
        returns: list[float] = [0.0] * len(experiences)
        running = 0.0
        for t in range(len(experiences) - 1, -1, -1):
            running = experiences[t].reward + gamma * running
            returns[t] = running
        return returns

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def mean_reward(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(e.reward for e in self._buffer) / len(self._buffer)

    def total_episodes(self) -> int:
        return sum(1 for e in self._buffer if e.done)

    def as_arrays(self) -> dict[str, list]:
        states: list[list[float]] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_states: list[list[float]] = []
        dones: list[bool] = []
        for e in self._buffer:
            states.append(list(e.state))
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(list(e.next_state))
            dones.append(e.done)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }


ROLLOUT_BUFFER_REGISTRY = {"default": RolloutBuffer}
