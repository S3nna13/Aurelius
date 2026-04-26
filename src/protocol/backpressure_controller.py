"""Backpressure controller for the Aurelius protocol surface.

Evaluates queue depth and throughput to emit a :class:`BackpressureSignal`
and a suggested client delay.  Optionally applies the delay via
``time.sleep``.

Pure stdlib only.  No external dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum


class BackpressureSignal(StrEnum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"
    CRITICAL = "critical"


@dataclass
class BackpressureState:
    signal: BackpressureSignal
    queue_depth: int
    drain_rate: float  # requests / second
    suggested_delay_ms: float


class BackpressureController:
    """Generates backpressure signals based on queue depth thresholds."""

    DEFAULT_THRESHOLDS: dict[str, int] = {
        "SOFT": 50,
        "HARD": 100,
        "CRITICAL": 200,
    }

    def __init__(self, thresholds: dict[str, int] | None = None) -> None:
        self.thresholds: dict[str, int] = (
            thresholds if thresholds is not None else dict(self.DEFAULT_THRESHOLDS)
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queue_depth: int,
        throughput_rps: float,
    ) -> BackpressureState:
        """Return a :class:`BackpressureState` describing current pressure.

        *throughput_rps* is the current drain rate (requests processed/s).
        """
        drain_rate = max(throughput_rps, 0.0)

        if queue_depth >= self.thresholds["CRITICAL"]:
            return BackpressureState(
                signal=BackpressureSignal.CRITICAL,
                queue_depth=queue_depth,
                drain_rate=drain_rate,
                suggested_delay_ms=1000.0,
            )
        if queue_depth >= self.thresholds["HARD"]:
            return BackpressureState(
                signal=BackpressureSignal.HARD,
                queue_depth=queue_depth,
                drain_rate=drain_rate,
                suggested_delay_ms=100.0,
            )
        if queue_depth >= self.thresholds["SOFT"]:
            return BackpressureState(
                signal=BackpressureSignal.SOFT,
                queue_depth=queue_depth,
                drain_rate=drain_rate,
                suggested_delay_ms=10.0,
            )

        return BackpressureState(
            signal=BackpressureSignal.NONE,
            queue_depth=queue_depth,
            drain_rate=drain_rate,
            suggested_delay_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Delay application
    # ------------------------------------------------------------------

    def apply_delay(self, state: BackpressureState) -> None:
        """Sleep for the delay indicated by *state* (converts ms → s)."""
        if state.suggested_delay_ms > 0:
            time.sleep(state.suggested_delay_ms / 1000.0)

    # ------------------------------------------------------------------
    # Drain estimation
    # ------------------------------------------------------------------

    def get_drain_estimate(
        self,
        queue_depth: int,
        throughput_rps: float,
    ) -> float:
        """Return estimated seconds to drain *queue_depth* at *throughput_rps*.

        Returns ``float('inf')`` when *throughput_rps* is zero or negative.
        """
        if throughput_rps <= 0:
            return float("inf")
        return queue_depth / throughput_rps


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

PROTOCOL_REGISTRY: dict = {
    "backpressure": BackpressureController(),
}
