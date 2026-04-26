"""Experience replay buffer for RLHF and online PPO training.

Implements:
- ReplayBuffer: circular buffer with optional prioritized sampling (Schaul et al. 2015)
- OnlinePPOBuffer: rollout buffer with GAE advantage computation (Schulman et al. 2017)
- RewardNormalizer: online Welford variance for reward normalization

References:
    Ziegler et al. 2019 (RLHF), Schulman et al. 2017 (PPO), Schaul et al. 2015 (PER)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class Experience:
    prompt_ids: list[int]  # tokenized prompt
    completion_ids: list[int]  # tokenized completion
    reward: float  # scalar reward signal
    log_prob: float  # log probability under policy at collection time
    step: int  # training step when collected
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Circular replay buffer with optional priority
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Circular experience replay buffer with optional prioritized sampling.

    Args:
        capacity: Maximum number of experiences stored.
        prioritized: If True, sample proportional to |reward|^alpha.
        alpha: Priority exponent (default 0.6).
        beta: Importance-sampling correction exponent (default 0.4).
    """

    def __init__(
        self,
        capacity: int = 10_000,
        prioritized: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta

        self._buffer: list[Experience] = [None] * capacity  # type: ignore[list-item]
        self._priorities: list[float] = [0.0] * capacity
        self._write_idx: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------
    def add(self, experience: Experience) -> None:
        """Add an experience. Priority = |reward|^alpha + 1e-6."""
        priority = abs(experience.reward) ** self.alpha + 1e-6
        idx = self._write_idx % self.capacity
        self._buffer[idx] = experience
        self._priorities[idx] = priority
        self._write_idx += 1
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> tuple[list[Experience], list[float]]:
        """Sample *batch_size* experiences.

        Returns:
            (experiences, importance_weights)
            - Prioritized: weights w_i = (1/N * 1/P(i))^beta, normalized by max(w).
            - Uniform: all weights 1.0.
        """
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")
        n = self._size
        batch_size = min(batch_size, n)

        if self.prioritized:
            priorities = torch.tensor(self._priorities[:n], dtype=torch.float64)
            # torch.multinomial requires float32
            probs = (priorities / priorities.sum()).float()
            indices = torch.multinomial(probs, num_samples=batch_size, replacement=True).tolist()

            # Importance-sampling weights
            p_i = probs[indices]
            weights = (1.0 / (n * p_i)) ** self.beta
            weights = (weights / weights.max()).tolist()
        else:
            indices = random.choices(range(n), k=batch_size)
            weights = [1.0] * batch_size

        experiences = [self._buffer[i] for i in indices]
        return experiences, weights

    # ------------------------------------------------------------------
    def update_priorities(self, indices: list[int], new_rewards: list[float]) -> None:
        """Update priorities for given indices after new reward estimates."""
        for idx, reward in zip(indices, new_rewards):
            if 0 <= idx < self._size:
                self._priorities[idx] = abs(reward) ** self.alpha + 1e-6

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return buffer statistics."""
        if self._size == 0:
            return {
                "size": 0,
                "mean_reward": float("nan"),
                "max_reward": float("nan"),
                "min_reward": float("nan"),
                "capacity": self.capacity,
            }
        rewards = [self._buffer[i].reward for i in range(self._size)]
        return {
            "size": self._size,
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "capacity": self.capacity,
        }


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------


class OnlinePPOBuffer:
    """PPO-style rollout buffer.

    Stores full episodes then clears after each policy update.
    Computes GAE advantages from rewards and value estimates.

    Args:
        gamma: Discount factor (default 1.0 for RLHF bandit setting).
        lam: GAE lambda (default 0.95).
    """

    def __init__(self, gamma: float = 1.0, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam
        self.experiences: list[Experience] = []
        self.values: list[float] = []

    # ------------------------------------------------------------------
    def add(self, experience: Experience, value: float) -> None:
        """Add an experience together with its baseline value estimate."""
        self.experiences.append(experience)
        self.values.append(value)

    # ------------------------------------------------------------------
    def compute_advantages(self) -> list[float]:
        """Compute GAE advantages.

        δ_t = r_t + γ*V_{t+1} - V_t
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        For the typical RLHF bandit setting (γ=1, single reward at end):
        A_t = reward - value_t
        """
        n = len(self.experiences)
        if n == 0:
            return []

        advantages = [0.0] * n
        gae = 0.0
        # Bootstrap value after last step is 0 (episode ends)
        next_value = 0.0

        for t in reversed(range(n)):
            reward = self.experiences[t].reward
            value = self.values[t]
            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            next_value = value

        return advantages

    # ------------------------------------------------------------------
    def get_batch(self) -> tuple[list[Experience], list[float]]:
        """Return (experiences, advantages). Does NOT clear the buffer."""
        return self.experiences, self.compute_advantages()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear the buffer after a policy update."""
        self.experiences = []
        self.values = []


# ---------------------------------------------------------------------------
# Online reward normalizer (Welford's algorithm)
# ---------------------------------------------------------------------------


class RewardNormalizer:
    """Online running statistics for reward normalization.

    Uses Welford's algorithm to maintain a running mean and variance.
    Normalized reward: r_norm = (r - mean) / (std + eps)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # sum of squared deviations from current mean

    # ------------------------------------------------------------------
    def update(self, reward: float) -> None:
        """Update running statistics with a new reward (Welford's online algorithm)."""
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        delta2 = reward - self.mean
        self.M2 += delta * delta2

    # ------------------------------------------------------------------
    def normalize(self, reward: float) -> float:
        """Return normalized reward. If n < 2, return reward unchanged."""
        if self.n < 2:
            return reward
        return (reward - self.mean) / (self.std + self.eps)

    # ------------------------------------------------------------------
    @property
    def std(self) -> float:
        """Return running standard deviation (0.0 if n < 2)."""
        if self.n < 2:
            return 0.0
        variance = self.M2 / (self.n - 1)  # unbiased Bessel-corrected variance
        return math.sqrt(max(variance, 0.0))
