from __future__ import annotations

import torch


def shaped_reward(raw: float, prev: float, scale: float = 1.0) -> float:
    return (raw - prev) * scale


def potential_based_reward(
    raw: float, new_potential: float, old_potential: float, gamma: float = 0.99
) -> float:
    return raw + gamma * new_potential - old_potential


class RewardShaperV2:
    def __init__(self, gamma: float = 0.99, potential_scale: float = 1.0) -> None:
        self.gamma = gamma
        self.potential_scale = potential_scale
        self._last_potential: float | None = None

    def shape(self, raw: float, state: torch.Tensor) -> float:
        potential = float(torch.norm(state).item()) * self.potential_scale
        if self._last_potential is None:
            self._last_potential = potential
            return raw
        shaped = potential_based_reward(raw, potential, self._last_potential, self.gamma)
        self._last_potential = potential
        return shaped

    def reset(self) -> None:
        self._last_potential = None


REWARD_SHAPER_V2 = RewardShaperV2()
