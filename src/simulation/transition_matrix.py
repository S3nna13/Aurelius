"""Builds and analyzes Markov transition matrices."""

from __future__ import annotations

from collections import deque


class TransitionMatrix:
    """Builds and analyzes a Markov transition matrix over a fixed state space."""

    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self._index: dict[str, int] = {s: i for i, s in enumerate(states)}
        n = len(states)
        # Accumulate raw (possibly non-integer) weights
        self._raw: list[list[float]] = [[0.0] * n for _ in range(n)]

    def state_index(self, state: str) -> int | None:
        """Return the integer index of state, or None if not present."""
        return self._index.get(state, None)

    def add_transition(
        self, from_state: str, to_state: str, weight: float = 1.0
    ) -> None:
        """Accumulate weight for the from_state -> to_state edge."""
        fi = self._index.get(from_state)
        ti = self._index.get(to_state)
        if fi is None or ti is None:
            raise KeyError(
                f"Unknown state(s): from_state={from_state!r}, to_state={to_state!r}"
            )
        self._raw[fi][ti] += weight

    def normalize(self) -> list[list[float]]:
        """Return row-normalized probability matrix.

        Each row sums to 1.0; rows with all-zero weights remain all zeros.
        """
        n = len(self._states)
        result: list[list[float]] = []
        for row in self._raw:
            row_sum = sum(row)
            if row_sum == 0.0:
                result.append([0.0] * n)
            else:
                result.append([v / row_sum for v in row])
        return result

    def stationary_distribution(
        self, max_iter: int = 1000, tol: float = 1e-8
    ) -> list[float]:
        """Compute the stationary distribution via power iteration.

        Starts from a uniform distribution and multiplies by the row-stochastic
        matrix until convergence or max_iter is reached.
        Returns a probability vector over states.
        """
        n = len(self._states)
        if n == 0:
            return []

        P = self.normalize()

        # Start uniform
        dist = [1.0 / n] * n

        for _ in range(max_iter):
            new_dist = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_dist[j] += dist[i] * P[i][j]

            # Check convergence
            diff = sum(abs(new_dist[i] - dist[i]) for i in range(n))
            dist = new_dist
            if diff < tol:
                break

        return dist

    def is_absorbing(self, state: str) -> bool:
        """Return True if all transition weight from state goes to itself (self-loop only).

        Uses the normalized matrix: state is absorbing iff P[i][i] == 1.0.
        """
        i = self._index.get(state)
        if i is None:
            raise KeyError(f"Unknown state: {state!r}")
        P = self.normalize()
        row = P[i]
        # Row must be all-zero except possibly the diagonal
        for j, v in enumerate(row):
            if j == i:
                continue
            if v != 0.0:
                return False
        # Additionally, the diagonal must be 1.0 (not all-zero row)
        return row[i] == 1.0

    def reachable_from(self, state: str) -> set[str]:
        """BFS over transitions with nonzero weight; returns reachable state names."""
        i = self._index.get(state)
        if i is None:
            raise KeyError(f"Unknown state: {state!r}")
        n = len(self._states)
        visited: set[int] = {i}
        queue: deque[int] = deque([i])
        while queue:
            cur = queue.popleft()
            for j in range(n):
                if self._raw[cur][j] != 0.0 and j not in visited:
                    visited.add(j)
                    queue.append(j)
        return {self._states[k] for k in visited}


TRANSITION_MATRIX_REGISTRY: dict[str, type] = {"default": TransitionMatrix}

REGISTRY = TRANSITION_MATRIX_REGISTRY
