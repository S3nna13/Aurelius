"""Monte Carlo simulation for RL value estimation."""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MCConfig:
    num_episodes: int = 1000
    gamma: float = 0.99
    seed: int = 42


@dataclass(frozen=True)
class MCEpisode:
    episode_id: int
    rewards: list[float]
    returns: list[float]
    total_return: float


class MonteCarloSimulator:
    """Monte Carlo simulator for RL value estimation."""

    def __init__(self, config: MCConfig | None = None) -> None:
        self.config = config if config is not None else MCConfig()
        self._rng = random.Random(self.config.seed)

    def compute_returns(self, rewards: list[float], gamma: float | None = None) -> list[float]:
        """Compute discounted returns G_t = r_t + gamma * G_{t+1}, working backwards."""
        if gamma is None:
            gamma = self.config.gamma
        n = len(rewards)
        if n == 0:
            return []
        returns = [0.0] * n
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    def run_episode(
        self,
        reward_fn: Callable[[int], float],
        max_steps: int = 200,
    ) -> MCEpisode:
        """Run a single episode using reward_fn(step) for each step."""
        rewards: list[float] = []
        for step in range(max_steps):
            r = reward_fn(step)
            rewards.append(r)
        returns = self.compute_returns(rewards)
        total_return = returns[0] if returns else 0.0
        episode_id = self._rng.randint(0, 2**31 - 1)
        return MCEpisode(
            episode_id=episode_id,
            rewards=rewards,
            returns=returns,
            total_return=total_return,
        )

    def estimate_value(self, episodes: list[MCEpisode]) -> float:
        """Return the mean total_return across episodes."""
        if not episodes:
            return 0.0
        return statistics.mean(ep.total_return for ep in episodes)

    def value_variance(self, episodes: list[MCEpisode]) -> float:
        """Return population stdev of total_return; 0.0 if fewer than 2 episodes."""
        if len(episodes) < 2:
            return 0.0
        return statistics.pstdev(ep.total_return for ep in episodes)


MONTE_CARLO_REGISTRY: dict[str, type] = {"default": MonteCarloSimulator}

REGISTRY = MONTE_CARLO_REGISTRY
