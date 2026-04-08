"""Budgeting helpers for draft-model speculative decoding."""

from __future__ import annotations


def allocate_draft_steps(max_new_tokens: int, draft_width: int) -> int:
    """Number of draft verification rounds needed."""
    if max_new_tokens < 0 or draft_width <= 0:
        raise ValueError("max_new_tokens must be non-negative and draft_width positive")
    if max_new_tokens == 0:
        return 0
    return (max_new_tokens + draft_width - 1) // draft_width


def accepted_tokens_per_round(acceptance_rate: float, draft_width: int) -> float:
    """Expected accepted tokens in one speculative round."""
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError("acceptance_rate must be in [0, 1]")
    if draft_width <= 0:
        raise ValueError("draft_width must be positive")
    return acceptance_rate * draft_width


def expected_speedup(acceptance_rate: float, draft_width: int) -> float:
    """Approximate speedup over one-token decoding."""
    accepted = accepted_tokens_per_round(acceptance_rate, draft_width)
    return max(accepted, 1.0)


def remaining_budget(max_new_tokens: int, emitted_tokens: int) -> int:
    """Tokens left in the decode budget."""
    if max_new_tokens < 0 or emitted_tokens < 0:
        raise ValueError("budgets must be non-negative")
    return max(max_new_tokens - emitted_tokens, 0)
