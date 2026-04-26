"""Token Budget v2 — Compute usage tracking, cost estimation, and limit enforcement.

Provides utilities for monitoring token consumption during LLM inference,
estimating API/compute costs, and enforcing configurable token budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Configuration for token budget limits and cost rates."""

    max_tokens: int = 10_000
    max_prompt_tokens: int = 4_096
    max_completion_tokens: int = 2_048
    cost_per_prompt_token: float = 1e-6
    cost_per_completion_token: float = 2e-6
    warn_at_fraction: float = 0.8


# ---------------------------------------------------------------------------
# Usage dataclass
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Accumulated token usage for a single call or over multiple calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0  # always prompt_tokens + completion_tokens
    estimated_cost: float = 0.0


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_token_usage(
    prompt_len: int,
    completion_len: int,
    config: BudgetConfig,
) -> TokenUsage:
    """Return a :class:`TokenUsage` computed from raw lengths and cost rates.

    Args:
        prompt_len: Number of prompt tokens.
        completion_len: Number of completion tokens.
        config: :class:`BudgetConfig` carrying cost rates.

    Returns:
        Populated :class:`TokenUsage` instance.
    """
    total = prompt_len + completion_len
    cost = (
        prompt_len * config.cost_per_prompt_token
        + completion_len * config.cost_per_completion_token
    )
    return TokenUsage(
        prompt_tokens=prompt_len,
        completion_tokens=completion_len,
        total_tokens=total,
        estimated_cost=cost,
    )


def check_budget(
    usage: TokenUsage,
    config: BudgetConfig,
) -> tuple[bool, str]:
    """Check whether *usage* is within the limits defined by *config*.

    Args:
        usage: Current :class:`TokenUsage` to validate.
        config: :class:`BudgetConfig` defining limits.

    Returns:
        ``(within_budget, message)`` where *within_budget* is ``True`` when
        all limits are satisfied and *message* describes the first violation
        found (empty string when within budget).
    """
    if usage.total_tokens > config.max_tokens:
        return (
            False,
            f"total_tokens {usage.total_tokens} exceeds max_tokens {config.max_tokens}",
        )
    if usage.prompt_tokens > config.max_prompt_tokens:
        return (
            False,
            f"prompt_tokens {usage.prompt_tokens} exceeds max_prompt_tokens {config.max_prompt_tokens}",  # noqa: E501
        )
    if usage.completion_tokens > config.max_completion_tokens:
        return (
            False,
            f"completion_tokens {usage.completion_tokens} exceeds max_completion_tokens {config.max_completion_tokens}",  # noqa: E501
        )
    return (True, "")


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------


class TokenBudgetTracker:
    """Stateful tracker that accumulates token usage across multiple inference calls.

    Example::

        config = BudgetConfig(max_tokens=1000)
        tracker = TokenBudgetTracker(config)
        tracker.record(prompt_tokens=100, completion_tokens=50)
        print(tracker.usage_summary())
    """

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._n_calls: int = 0

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def record(self, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Accumulate *prompt_tokens* and *completion_tokens* from one call.

        Args:
            prompt_tokens: Prompt token count for this call.
            completion_tokens: Completion token count for this call.

        Returns:
            Current cumulative :class:`TokenUsage` after recording.
        """
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._n_calls += 1
        return self.total_usage()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._n_calls = 0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def total_usage(self) -> TokenUsage:
        """Return cumulative :class:`TokenUsage` across all recorded calls."""
        return compute_token_usage(
            self._prompt_tokens,
            self._completion_tokens,
            self._config,
        )

    def is_within_budget(self) -> bool:
        """Return ``True`` when cumulative total is within *max_tokens*."""
        within, _ = check_budget(self.total_usage(), self._config)
        return within

    def should_warn(self) -> bool:
        """Return ``True`` when total_tokens >= warn_at_fraction * max_tokens."""
        usage = self.total_usage()
        threshold = self._config.warn_at_fraction * self._config.max_tokens
        return usage.total_tokens >= threshold

    def usage_summary(self) -> dict:
        """Return a plain-dict summary of cumulative usage.

        Keys:
            - ``prompt_tokens`` — int
            - ``completion_tokens`` — int
            - ``total_tokens`` — int
            - ``estimated_cost`` — float
            - ``budget_fraction`` — float (total / max_tokens)
            - ``n_calls`` — int
        """
        usage = self.total_usage()
        budget_fraction = (
            usage.total_tokens / self._config.max_tokens if self._config.max_tokens > 0 else 0.0
        )
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost": usage.estimated_cost,
            "budget_fraction": budget_fraction,
            "n_calls": self._n_calls,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def estimate_tokens_from_chars(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from character count.

    Args:
        text: Input string.
        chars_per_token: Average characters per token (default 4.0).

    Returns:
        Estimated token count (always >= 1 for non-empty strings, >= 0 otherwise).
    """
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


def truncate_to_budget(
    token_ids: list[int],
    max_tokens: int,
    side: str = "left",
) -> list[int]:
    """Truncate a token-id sequence to at most *max_tokens* tokens.

    Args:
        token_ids: Sequence of integer token ids.
        max_tokens: Maximum number of tokens to keep.
        side: Which side to remove from.
            ``"left"`` (default) removes tokens from the **start** of the
            sequence (keeps the most recent context).
            ``"right"`` removes tokens from the **end**.

    Returns:
        Truncated list.  Returns the original list unchanged when
        ``len(token_ids) <= max_tokens``.

    Raises:
        ValueError: If *side* is not ``"left"`` or ``"right"``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if len(token_ids) <= max_tokens:
        return list(token_ids)
    if side == "left":
        return token_ids[len(token_ids) - max_tokens :]
    # side == "right"
    return token_ids[:max_tokens]
