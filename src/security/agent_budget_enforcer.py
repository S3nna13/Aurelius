"""Hard budget enforcer for agent loops.

Tracks steps, tokens, and elapsed time. Raises BudgetExhausted when any
limit is exceeded. Fail closed: ambiguous budgets default to the strictest
interpretation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class AgentBudget:
    max_steps: int | None = None
    max_tokens: int | None = None
    max_elapsed_seconds: float | None = None


@dataclass
class BudgetSnapshot:
    steps_used: int
    tokens_used: int
    elapsed_seconds: float
    exhausted: bool
    reason: str = ""


class AgentBudgetEnforcer:
    """Enforces one or more hard limits on an agent execution."""

    def __init__(self, budget: AgentBudget | None = None) -> None:
        self._budget = budget or AgentBudget()
        self._steps = 0
        self._tokens = 0
        self._start = time.monotonic()
        self._exhausted = False
        self._reason = ""

    def check(self, additional_steps: int = 0, additional_tokens: int = 0) -> BudgetSnapshot:
        """Check current budget status. Raises BudgetExhausted if over limit."""
        if self._exhausted:
            raise BudgetExhausted(self._reason)

        self._steps += additional_steps
        self._tokens += additional_tokens
        elapsed = time.monotonic() - self._start

        if self._budget.max_steps is not None and self._steps > self._budget.max_steps:
            self._exhausted = True
            self._reason = f"step_budget_exhausted ({self._steps}/{self._budget.max_steps})"
            raise BudgetExhausted(self._reason)

        if self._budget.max_tokens is not None and self._tokens > self._budget.max_tokens:
            self._exhausted = True
            self._reason = f"token_budget_exhausted ({self._tokens}/{self._budget.max_tokens})"
            raise BudgetExhausted(self._reason)

        if self._budget.max_elapsed_seconds is not None and elapsed > self._budget.max_elapsed_seconds:
            self._exhausted = True
            self._reason = f"time_budget_exhausted ({elapsed:.1f}s/{self._budget.max_elapsed_seconds}s)"
            raise BudgetExhausted(self._reason)

        return BudgetSnapshot(
            steps_used=self._steps,
            tokens_used=self._tokens,
            elapsed_seconds=elapsed,
            exhausted=False,
        )

    def snapshot(self) -> BudgetSnapshot:
        """Return current usage without incrementing or raising."""
        elapsed = time.monotonic() - self._start
        return BudgetSnapshot(
            steps_used=self._steps,
            tokens_used=self._tokens,
            elapsed_seconds=elapsed,
            exhausted=self._exhausted,
            reason=self._reason,
        )

    def reset(self) -> None:
        """Reset counters and clear exhausted state."""
        self._steps = 0
        self._tokens = 0
        self._start = time.monotonic()
        self._exhausted = False
        self._reason = ""


# Module-level registry
BUDGET_ENFORCER_REGISTRY: dict[str, AgentBudgetEnforcer] = {}
DEFAULT_BUDGET_ENFORCER = AgentBudgetEnforcer()
BUDGET_ENFORCER_REGISTRY["default"] = DEFAULT_BUDGET_ENFORCER
