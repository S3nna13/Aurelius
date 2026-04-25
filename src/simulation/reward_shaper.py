"""Reward shaping: dense, sparse, potential-based, curiosity, and clip strategies."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class RewardShapeType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    POTENTIAL_BASED = "potential_based"
    CURIOSITY = "curiosity"
    CLIP = "clip"


@dataclass
class RewardShaperConfig:
    shape_type: RewardShapeType = RewardShapeType.DENSE
    clip_range: tuple[float, float] = (-10.0, 10.0)
    gamma: float = 0.99
    curiosity_beta: float = 0.01


@runtime_checkable
class PotentialFunction(Protocol):
    """Protocol for potential functions used in potential-based shaping."""

    def __call__(self, state: dict) -> float:
        ...


class _DefaultPotential:
    """Default zero potential function."""

    def __call__(self, state: dict) -> float:
        return 0.0


class RewardShaper:
    """Shapes raw rewards according to the configured strategy.

    Strategies
    ----------
    DENSE           : pass-through with clip
    SPARSE          : zero except on done; clip applied
    POTENTIAL_BASED : r + gamma * phi(s') - phi(s)  (Ng et al. 1999)
    CURIOSITY       : bonus for novel states (count-based)
    CLIP            : clamp to clip_range only
    """

    def __init__(
        self,
        config: RewardShaperConfig | None = None,
        potential_fn: PotentialFunction | None = None,
    ) -> None:
        self.config = config or RewardShaperConfig()
        self.potential_fn: PotentialFunction = potential_fn or _DefaultPotential()
        self._visit_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clip(self, r: float) -> float:
        lo, hi = self.config.clip_range
        return max(lo, min(hi, r))

    def _state_key(self, state: dict) -> str:
        """Convert a state dict to a stable string key."""
        try:
            return str(sorted((k, str(v)) for k, v in state.items()))
        except TypeError:
            return str(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shape(
        self,
        raw_reward: float,
        state: dict,
        next_state: dict,
        done: bool,
    ) -> float:
        """Return the shaped reward for transition (state → next_state)."""
        t = self.config.shape_type

        if t == RewardShapeType.DENSE:
            return self._clip(raw_reward)

        elif t == RewardShapeType.SPARSE:
            r = raw_reward if done else 0.0
            return self._clip(r)

        elif t == RewardShapeType.POTENTIAL_BASED:
            phi_s = self.potential_fn(state)
            phi_s_prime = self.potential_fn(next_state)
            shaped = raw_reward + self.config.gamma * phi_s_prime - phi_s
            return self._clip(shaped)

        elif t == RewardShapeType.CURIOSITY:
            key = self._state_key(next_state)
            self._visit_counts[key] += 1
            bonus = self.config.curiosity_beta / (self._visit_counts[key] ** 0.5)
            return self._clip(raw_reward + bonus)

        elif t == RewardShapeType.CLIP:
            return self._clip(raw_reward)

        else:
            return raw_reward

    def reset_visit_counts(self) -> None:
        """Clear curiosity visit counters (call between experiments)."""
        self._visit_counts.clear()
