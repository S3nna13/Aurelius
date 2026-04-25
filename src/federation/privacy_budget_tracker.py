"""Privacy budget tracker: tracks differential privacy epsilon/delta consumption."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyBudget:
    total_epsilon: float
    total_delta: float


@dataclass(frozen=True)
class BudgetRecord:
    round_idx: int
    epsilon_used: float
    delta_used: float
    cumulative_epsilon: float
    cumulative_delta: float


class PrivacyBudgetTracker:
    """Tracks differential privacy budget consumption across federated rounds."""

    def __init__(self, budget: PrivacyBudget) -> None:
        self._budget = budget
        self._records: list[BudgetRecord] = []
        self._cumulative_epsilon: float = 0.0
        self._cumulative_delta: float = 0.0

    def consume(self, round_idx: int, epsilon: float, delta: float) -> BudgetRecord:
        """Record privacy consumption for a round.

        Raises ValueError if the new cumulative values exceed the total budget.
        """
        new_epsilon = self._cumulative_epsilon + epsilon
        new_delta = self._cumulative_delta + delta

        if new_epsilon > self._budget.total_epsilon:
            raise ValueError(
                f"Epsilon budget exceeded: cumulative {new_epsilon:.6g} > "
                f"total {self._budget.total_epsilon:.6g}"
            )
        if new_delta > self._budget.total_delta:
            raise ValueError(
                f"Delta budget exceeded: cumulative {new_delta:.6g} > "
                f"total {self._budget.total_delta:.6g}"
            )

        self._cumulative_epsilon = new_epsilon
        self._cumulative_delta = new_delta

        record = BudgetRecord(
            round_idx=round_idx,
            epsilon_used=epsilon,
            delta_used=delta,
            cumulative_epsilon=new_epsilon,
            cumulative_delta=new_delta,
        )
        self._records.append(record)
        return record

    def remaining(self) -> tuple[float, float]:
        """Return (remaining_epsilon, remaining_delta)."""
        return (
            self._budget.total_epsilon - self._cumulative_epsilon,
            self._budget.total_delta - self._cumulative_delta,
        )

    def is_exhausted(self) -> bool:
        """Return True if either remaining budget is <= 0."""
        rem_eps, rem_delta = self.remaining()
        return rem_eps <= 0.0 or rem_delta <= 0.0

    def history(self) -> list[BudgetRecord]:
        """Return a copy of the consumption history."""
        return list(self._records)

    def summary(self) -> dict:
        """Return a summary of budget state."""
        rem_eps, rem_delta = self.remaining()
        return {
            "total_epsilon": self._budget.total_epsilon,
            "used_epsilon": self._cumulative_epsilon,
            "remaining_epsilon": rem_eps,
            "total_delta": self._budget.total_delta,
            "used_delta": self._cumulative_delta,
            "remaining_delta": rem_delta,
            "rounds": len(self._records),
        }


PRIVACY_BUDGET_TRACKER_REGISTRY: dict[str, type] = {"default": PrivacyBudgetTracker}
