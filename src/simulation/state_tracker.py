"""Tracks environment state transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class StateTransition:
    from_state: str
    action: str
    to_state: str
    reward: float
    timestamp: float


class StateTracker:
    """Tracks state transitions for an RL environment."""

    def __init__(self, initial_state: str = "start") -> None:
        self._current_state = initial_state
        self._history: list[StateTransition] = []

    @property
    def current_state(self) -> str:
        return self._current_state

    def transition(
        self,
        action: str,
        next_state: str,
        reward: float = 0.0,
    ) -> StateTransition:
        """Record a transition and update the current state."""
        t = StateTransition(
            from_state=self._current_state,
            action=action,
            to_state=next_state,
            reward=reward,
            timestamp=time.monotonic(),
        )
        self._history.append(t)
        self._current_state = next_state
        return t

    def history(self) -> list[StateTransition]:
        return list(self._history)

    def visited_states(self) -> set[str]:
        """All states that have appeared as from_state or to_state, plus initial."""
        states: set[str] = set()
        for tr in self._history:
            states.add(tr.from_state)
            states.add(tr.to_state)
        # If no transitions, the initial state is still "visited"
        if not self._history:
            states.add(self._current_state)
        return states

    def action_counts(self) -> dict[str, int]:
        """Return {action: count} across all recorded transitions."""
        counts: dict[str, int] = {}
        for tr in self._history:
            counts[tr.action] = counts.get(tr.action, 0) + 1
        return counts

    def total_reward(self) -> float:
        """Sum of all rewards across recorded transitions."""
        return sum(tr.reward for tr in self._history)

    def reset(self, initial_state: str = "start") -> None:
        """Clear history and reset to initial_state."""
        self._current_state = initial_state
        self._history = []

    def to_markov_matrix(self, states: list[str]) -> list[list[float]]:
        """Build an NxN transition probability matrix over the given state list.

        P[i][j] = count(i -> j) / count(i -> *).
        Rows for unvisited states remain all zeros.
        """
        n = len(states)
        index = {s: i for i, s in enumerate(states)}

        # Accumulate raw counts
        raw: list[list[float]] = [[0.0] * n for _ in range(n)]
        for tr in self._history:
            fi = index.get(tr.from_state)
            ti = index.get(tr.to_state)
            if fi is not None and ti is not None:
                raw[fi][ti] += 1.0

        # Row-normalize
        matrix: list[list[float]] = []
        for row in raw:
            row_sum = sum(row)
            if row_sum == 0.0:
                matrix.append([0.0] * n)
            else:
                matrix.append([v / row_sum for v in row])
        return matrix


STATE_TRACKER_REGISTRY: dict[str, type] = {"default": StateTracker}

REGISTRY = STATE_TRACKER_REGISTRY
