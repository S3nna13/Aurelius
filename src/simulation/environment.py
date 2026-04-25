"""Environment abstraction: state, action, step interface, GridWorld implementation."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class EnvState:
    obs: dict
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)
    state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class EnvAction:
    action_id: str
    params: dict = field(default_factory=dict)


@dataclass
class EnvStep:
    prev_state: EnvState
    action: EnvAction
    next_state: EnvState
    step_number: int


class Environment:
    """Abstract base environment."""

    name: str = "base"

    def reset(self) -> EnvState:
        raise NotImplementedError

    def step(self, state: EnvState, action: EnvAction) -> EnvStep:
        raise NotImplementedError

    def valid_actions(self, state: EnvState) -> list[EnvAction]:
        raise NotImplementedError


class GridWorldEnv(Environment):
    """Simple 2-D grid world where an agent navigates to a goal cell."""

    name = "gridworld"

    def __init__(self, width: int = 5, height: int = 5, goal: tuple = (4, 4)) -> None:
        self.width = width
        self.height = height
        self.goal = goal

    def reset(self) -> EnvState:
        return EnvState(obs={"x": 0, "y": 0, "goal": self.goal})

    def step(self, state: EnvState, action: EnvAction) -> EnvStep:
        x = state.obs["x"]
        y = state.obs["y"]

        dx, dy = 0, 0
        if action.action_id == "right":
            dx = 1
        elif action.action_id == "left":
            dx = -1
        elif action.action_id == "up":
            dy = 1
        elif action.action_id == "down":
            dy = -1

        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))

        reached_goal = (new_x, new_y) == tuple(self.goal)
        reward = 1.0 if reached_goal else -0.01
        done = reached_goal

        next_state = EnvState(
            obs={"x": new_x, "y": new_y, "goal": self.goal},
            reward=reward,
            done=done,
        )
        step_number = getattr(state, "_step_number", 0) + 1
        return EnvStep(
            prev_state=state,
            action=action,
            next_state=next_state,
            step_number=step_number,
        )

    def valid_actions(self, state: EnvState) -> list[EnvAction]:
        return [
            EnvAction(action_id="up"),
            EnvAction(action_id="down"),
            EnvAction(action_id="left"),
            EnvAction(action_id="right"),
        ]


ENV_REGISTRY: dict[str, type] = {"gridworld": GridWorldEnv}
