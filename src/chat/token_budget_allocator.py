"""Per-role token budget allocation for multi-turn chat packing.

When fitting a conversation into a fixed context window, naive truncation
destroys system instructions first or last.  This allocator splits a **total
token budget** across messages using configurable **role weights** while
respecting hard floors so critical roles (typically ``system``) always
receive a minimum share before proportional distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenAllocation:
    """Per-message token caps (indices align with input ``messages``)."""

    caps: tuple[int, ...]
    total_budget: int


class TokenBudgetAllocator:
    """Compute integer per-message caps under a global token budget."""

    def __init__(
        self,
        *,
        role_weights: dict[str, float] | None = None,
        role_floors: dict[str, int] | None = None,
    ) -> None:
        self._weights = dict(role_weights or {"system": 3.0, "user": 1.5, "assistant": 1.0, "tool": 1.0})
        self._floors = dict(role_floors or {"system": 32})

    def allocate(
        self,
        messages: list[dict[str, Any]],
        measured_tokens: list[int],
        total_budget: int,
    ) -> TokenAllocation:
        """Return non-negative integer caps summing to at most ``total_budget``."""
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        if not isinstance(measured_tokens, list):
            raise TypeError("measured_tokens must be a list")
        if len(messages) != len(measured_tokens):
            raise ValueError("messages and measured_tokens must have same length")
        if total_budget < 0:
            raise ValueError("total_budget must be >= 0")
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or "role" not in m:
                raise ValueError(f"message {i} must be a dict with 'role'")
            if not isinstance(measured_tokens[i], int) or measured_tokens[i] < 0:
                raise ValueError(f"measured_tokens[{i}] must be int >= 0")

        n = len(messages)
        if n == 0:
            return TokenAllocation(caps=(), total_budget=total_budget)

        roles = [str(m["role"]) for m in messages]
        weights = [float(self._weights.get(r, 1.0)) for r in roles]
        floors = [max(0, int(self._floors.get(r, 0))) for r in roles]
        floor_sum = sum(floors)
        if floor_sum > total_budget:
            raise RuntimeError(
                f"token_budget_allocator: role floors sum to {floor_sum} "
                f"> total_budget={total_budget}"
            )

        remaining = total_budget - floor_sum
        if remaining == 0:
            return TokenAllocation(caps=tuple(floors), total_budget=total_budget)

        wsum = float(sum(weights)) or 1.0
        raw = [remaining * (weights[i] / wsum) for i in range(n)]
        extra_int = [int(x) for x in raw]
        drift = remaining - sum(extra_int)
        order = sorted(range(n), key=lambda i: (-(raw[i] - extra_int[i]), i))
        for k in range(drift):
            extra_int[order[k % n]] += 1
        caps = [floors[i] + extra_int[i] for i in range(n)]
        if sum(caps) != total_budget:
            raise RuntimeError("token_budget_allocator: internal sum mismatch")

        return TokenAllocation(caps=tuple(int(c) for c in caps), total_budget=total_budget)


__all__ = ["TokenAllocation", "TokenBudgetAllocator"]
