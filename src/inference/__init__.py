"""Aurelius inference subsystem."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Continuous batching (Orca, OSDI 2022)
# ---------------------------------------------------------------------------
from src.inference.continuous_batching_scheduler import (
    BatchStep,
    ContinuousBatchingScheduler,
    InferenceRequest,
)

try:  # pragma: no cover - only triggered if a decoder registry exists elsewhere
    DECODER_REGISTRY  # type: ignore[name-defined]
except NameError:
    pass
else:  # pragma: no cover
    DECODER_REGISTRY["continuous_batching"] = ContinuousBatchingScheduler  # type: ignore[name-defined]

try:
    SCHEDULER_REGISTRY  # type: ignore[name-defined]
except NameError:
    SCHEDULER_REGISTRY = {}

SCHEDULER_REGISTRY["continuous_batching"] = ContinuousBatchingScheduler

__all__ = [
    "BatchStep",
    "ContinuousBatchingScheduler",
    "InferenceRequest",
    "SCHEDULER_REGISTRY",
]
