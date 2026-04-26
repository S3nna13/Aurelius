"""Curriculum environment: wraps an inner environment and advances difficulty on mastery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .environment import EnvAction, Environment, EnvState, EnvStep, GridWorldEnv


@dataclass
class CurriculumLevel:
    level_id: int
    config: dict
    success_threshold: float = 0.8


class CurriculumEnv:
    """Wraps an :class:`Environment` and advances difficulty when the agent masters each level.

    The rolling success rate is tracked over the last ``window`` episodes.
    When ``try_advance()`` detects success_rate >= current level's threshold,
    the level is incremented and the inner environment is rebuilt with the
    new level config.

    Parameters
    ----------
    levels:
        Ordered list of curriculum levels from easiest to hardest.
    env_factory:
        Callable that takes a level ``config: dict`` and returns a fresh
        :class:`Environment` instance.
    window:
        Size of the rolling window used to compute success rate.
    """

    def __init__(
        self,
        levels: list[CurriculumLevel],
        env_factory=None,
        window: int = 20,
    ) -> None:
        if not levels:
            raise ValueError("CurriculumEnv requires at least one level.")
        self.levels = levels
        self._level_index = 0
        self.window = window
        self._history: deque[bool] = deque(maxlen=window)
        self._env_factory = env_factory or _default_grid_factory
        self._inner: Environment = self._env_factory(self.current_level.config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> CurriculumLevel:
        return self.levels[self._level_index]

    @property
    def is_final_level(self) -> bool:
        return self._level_index >= len(self.levels) - 1

    # ------------------------------------------------------------------
    # Episode result tracking
    # ------------------------------------------------------------------

    def record_episode_result(self, success: bool) -> None:
        """Record whether the most recent episode was a success."""
        self._history.append(success)

    def success_rate(self) -> float:
        """Rolling success rate over the last ``window`` episodes."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def try_advance(self) -> bool:
        """Advance to the next level if success rate meets the threshold.

        Returns ``True`` if the level was advanced, ``False`` otherwise.
        """
        if self.is_final_level:
            return False
        if self.success_rate() >= self.current_level.success_threshold:
            self._level_index += 1
            self._history.clear()
            self._inner = self._env_factory(self.current_level.config)
            return True
        return False

    # ------------------------------------------------------------------
    # Environment delegation
    # ------------------------------------------------------------------

    def reset(self) -> EnvState:
        return self._inner.reset()

    def step(self, state: EnvState, action: EnvAction) -> EnvStep:
        return self._inner.step(state, action)

    def valid_actions(self, state: EnvState) -> list[EnvAction]:
        return self._inner.valid_actions(state)


# ---------------------------------------------------------------------------
# Default factory — creates a GridWorldEnv sized by the level config
# ---------------------------------------------------------------------------


def _default_grid_factory(config: dict) -> Environment:
    width = config.get("width", 5)
    height = config.get("height", 5)
    goal = tuple(config.get("goal", (width - 1, height - 1)))
    return GridWorldEnv(width=width, height=height, goal=goal)
