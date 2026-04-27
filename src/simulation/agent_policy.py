"""Epsilon-greedy policy with discounted returns and TD error helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    clip_ratio: float = 0.2


@dataclass(frozen=True)
class PolicyStats:
    episode: int
    total_reward: float
    steps: int
    epsilon: float
    loss: float = 0.0


class EpsilonGreedyPolicy:
    """Discrete epsilon-greedy policy operating on plain Python lists."""

    def __init__(self, n_actions: int, config: PolicyConfig) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self.config = config
        self.epsilon = config.epsilon_start

    def select_action(self, q_values: list[float]) -> int:
        if len(q_values) != self.n_actions:
            raise ValueError(f"q_values length {len(q_values)} != n_actions {self.n_actions}")
        if random.uniform(0.0, 1.0) < self.epsilon:  # noqa: S311
            return random.randint(0, self.n_actions - 1)  # noqa: S311
        # argmax
        best_idx = 0
        best_val = q_values[0]
        for i in range(1, self.n_actions):
            if q_values[i] > best_val:
                best_val = q_values[i]
                best_idx = i
        return best_idx

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return self.epsilon

    def compute_returns(self, rewards: list[float], gamma: float | None = None) -> list[float]:
        g = self.config.gamma if gamma is None else gamma
        returns: list[float] = [0.0] * len(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + g * running
            returns[t] = running
        return returns

    def td_error(self, reward: float, next_q: float, current_q: float, done: bool) -> float:
        return reward + self.config.gamma * next_q * (0.0 if done else 1.0) - current_q

    def update_stats(self, episode: int, total_reward: float, steps: int) -> PolicyStats:
        return PolicyStats(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            epsilon=self.epsilon,
        )


AGENT_POLICY_REGISTRY = {"default": EpsilonGreedyPolicy}
