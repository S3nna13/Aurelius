"""Transition schedules between curriculum phases."""

from __future__ import annotations

import math


def linear_transition(step: int, start_step: int, end_step: int) -> float:
    """Linear interpolation factor between two curriculum phases."""
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")
    if step <= start_step:
        return 0.0
    if step >= end_step:
        return 1.0
    return (step - start_step) / (end_step - start_step)


def cosine_transition(step: int, start_step: int, end_step: int) -> float:
    """Cosine-smoothed transition factor between two phases."""
    alpha = linear_transition(step, start_step, end_step)
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def blended_weight(
    step: int,
    start_step: int,
    end_step: int,
    old_weight: float,
    new_weight: float,
    mode: str = "linear",
) -> float:
    """Blend two scalar weights through a transition schedule."""
    if mode == "linear":
        alpha = linear_transition(step, start_step, end_step)
    elif mode == "cosine":
        alpha = cosine_transition(step, start_step, end_step)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return (1.0 - alpha) * old_weight + alpha * new_weight


def stage_from_transition(step: int, boundaries: list[int]) -> int:
    """Return the active curriculum stage index from sorted boundaries."""
    if boundaries != sorted(boundaries):
        raise ValueError("boundaries must be sorted")
    stage = 0
    for boundary in boundaries:
        if step >= boundary:
            stage += 1
    return stage
