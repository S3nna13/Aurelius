"""Agent harness: run agent policy in environment, collect trajectories."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from .environment import EnvAction, Environment, EnvState, EnvStep, GridWorldEnv

random.seed(42)

PolicyFn = Callable[[EnvState, list[EnvAction]], EnvAction]


@dataclass
class Trajectory:
    env_name: str
    steps: list[EnvStep]
    total_reward: float
    success: bool


class AgentHarness:
    """Run agent policies in an environment and collect trajectories."""

    def __init__(self, env: Environment | None = None) -> None:
        self.env = env if env is not None else GridWorldEnv()

    def run_episode(self, policy: PolicyFn, max_steps: int = 100) -> Trajectory:
        state = self.env.reset()
        steps: list[EnvStep] = []
        total_reward = 0.0

        for _ in range(max_steps):
            actions = self.env.valid_actions(state)
            action = policy(state, actions)
            env_step = self.env.step(state, action)
            # Attach step number properly
            env_step = EnvStep(
                prev_state=env_step.prev_state,
                action=env_step.action,
                next_state=env_step.next_state,
                step_number=len(steps) + 1,
            )
            steps.append(env_step)
            total_reward += env_step.next_state.reward
            state = env_step.next_state
            if state.done:
                break

        return Trajectory(
            env_name=self.env.name,
            steps=steps,
            total_reward=total_reward,
            success=state.done,
        )

    @staticmethod
    def random_policy(state: EnvState, valid_actions: list[EnvAction]) -> EnvAction:
        return random.choice(valid_actions)  # noqa: S311

    @staticmethod
    def greedy_policy(goal: tuple) -> PolicyFn:
        """Return a policy that greedily minimises Manhattan distance to goal."""

        def _policy(state: EnvState, valid_actions: list[EnvAction]) -> EnvAction:
            x, y = state.obs["x"], state.obs["y"]
            gx, gy = goal

            _DELTAS = {
                "up": (0, 1),
                "down": (0, -1),
                "left": (-1, 0),
                "right": (1, 0),
            }

            best_action = valid_actions[0]
            best_dist = math.inf
            for action in valid_actions:
                dx, dy = _DELTAS.get(action.action_id, (0, 0))
                dist = abs(x + dx - gx) + abs(y + dy - gy)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action
            return best_action

        return _policy

    def run_n_episodes(self, policy: PolicyFn, n: int, max_steps: int = 100) -> list[Trajectory]:
        return [self.run_episode(policy, max_steps=max_steps) for _ in range(n)]

    @staticmethod
    def success_rate(trajectories: list[Trajectory]) -> float:
        if not trajectories:
            return 0.0
        return sum(1 for t in trajectories if t.success) / len(trajectories)


AGENT_HARNESS = AgentHarness()
