"""Multi-agent environment: base class and cooperative grid implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentAction:
    agent_id: str
    action: int
    metadata: dict = field(default_factory=dict)


@dataclass
class MultiAgentStep:
    observations: dict[str, Any]
    rewards: dict[str, float]
    dones: dict[str, bool]
    info: dict


class MultiAgentEnv:
    """Base environment for N cooperative/competitive agents.

    Subclasses must implement ``_build_obs``, ``_apply_actions``,
    ``observation_space``, and ``action_space``.
    """

    n_agents: int = 0
    observation_space: dict = {}
    action_space: dict = {}

    def reset(self) -> dict[str, Any]:
        """Return per-agent observation dict."""
        raise NotImplementedError

    def step(self, actions: list[AgentAction]) -> MultiAgentStep:
        """Apply actions and return (obs, rewards, dones, info)."""
        raise NotImplementedError

    def render(self) -> str:
        """Return a text representation of the current state."""
        return f"<{self.__class__.__name__}>"


# ---------------------------------------------------------------------------
# SimpleCooperativeEnv — 2-agent grid with shared reward on joint goal arrival
# ---------------------------------------------------------------------------

# Action integers
_UP = 0
_DOWN = 1
_LEFT = 2
_RIGHT = 3
_STAY = 4

_DELTAS = {_UP: (-1, 0), _DOWN: (1, 0), _LEFT: (0, -1), _RIGHT: (0, 1), _STAY: (0, 0)}


class SimpleCooperativeEnv(MultiAgentEnv):
    """Two-agent cooperative grid world.

    Both agents share a reward of +1.0 when they have *both* reached the
    goal cell in the same step.  Each step costs -0.01 per agent.

    Grid is ``grid_size x grid_size``.  Agent IDs are ``"agent_0"`` and
    ``"agent_1"``.
    """

    n_agents: int = 2

    def __init__(self, grid_size: int = 5, goal: tuple[int, int] = (4, 4)) -> None:
        self.grid_size = grid_size
        self.goal = goal
        self.agent_ids = ["agent_0", "agent_1"]
        self.observation_space = {
            aid: {"row": grid_size, "col": grid_size} for aid in self.agent_ids
        }
        self.action_space = {
            aid: {"n": 5}
            for aid in self.agent_ids  # UP DOWN LEFT RIGHT STAY
        }
        self._positions: dict[str, list[int]] = {}
        self._step_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        self._positions = {
            "agent_0": [0, 0],
            "agent_1": [0, 1],
        }
        self._step_count = 0
        return self._build_obs()

    def step(self, actions: list[AgentAction]) -> MultiAgentStep:
        action_map = {a.agent_id: a.action for a in actions}

        for aid in self.agent_ids:
            act = action_map.get(aid, _STAY)
            dr, dc = _DELTAS.get(act, (0, 0))
            r, c = self._positions[aid]
            r = max(0, min(self.grid_size - 1, r + dr))
            c = max(0, min(self.grid_size - 1, c + dc))
            self._positions[aid] = [r, c]

        self._step_count += 1

        both_at_goal = all(tuple(self._positions[aid]) == self.goal for aid in self.agent_ids)
        rewards: dict[str, float] = {}
        dones: dict[str, bool] = {}
        for aid in self.agent_ids:
            rewards[aid] = 1.0 if both_at_goal else -0.01
            dones[aid] = both_at_goal

        info = {"step": self._step_count, "both_at_goal": both_at_goal}
        return MultiAgentStep(
            observations=self._build_obs(),
            rewards=rewards,
            dones=dones,
            info=info,
        )

    def render(self) -> str:
        rows = []
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                cell = "."
                if (r, c) == self.goal:
                    cell = "G"
                for aid in self.agent_ids:
                    if self._positions[aid] == [r, c]:
                        cell = aid[-1]  # "0" or "1"
                row += cell
            rows.append(row)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict[str, Any]:
        return {
            aid: {
                "position": list(self._positions[aid]),
                "goal": list(self.goal),
                "other_positions": {
                    other: list(self._positions[other]) for other in self.agent_ids if other != aid
                },
            }
            for aid in self.agent_ids
        }
