"""SLO tracker: define service-level objectives and evaluate compliance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum


class SLOType(StrEnum):
    LATENCY = "LATENCY"
    AVAILABILITY = "AVAILABILITY"
    ERROR_RATE = "ERROR_RATE"
    THROUGHPUT = "THROUGHPUT"


@dataclass
class SLODefinition:
    name: str
    slo_type: SLOType
    target: float
    window_seconds: float = 3600.0


@dataclass
class SLOStatus:
    definition: SLODefinition
    current_value: float
    compliant: bool
    burn_rate: float


class SLOTracker:
    """Tracks samples and evaluates SLO compliance."""

    def __init__(self) -> None:
        self._definitions: dict[str, SLODefinition] = {}
        # Maps slo_name -> list of (timestamp, value)
        self._samples: dict[str, list[tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def define_slo(self, slo: SLODefinition) -> None:
        """Register an SLO definition."""
        self._definitions[slo.name] = slo
        if slo.name not in self._samples:
            self._samples[slo.name] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, slo_name: str, value: float) -> None:
        """Append a measurement sample with the current timestamp."""
        if slo_name not in self._samples:
            self._samples[slo_name] = []
        self._samples[slo_name].append((time.monotonic(), value))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, slo_name: str) -> SLOStatus:
        """Compute compliance status for a named SLO."""
        if slo_name not in self._definitions:
            raise KeyError(f"SLO '{slo_name}' not defined")

        defn = self._definitions[slo_name]
        cutoff = time.monotonic() - defn.window_seconds
        samples_in_window = [v for ts, v in self._samples.get(slo_name, []) if ts >= cutoff]

        if not samples_in_window:
            # No data – treat as 0 mean
            mean_val = 0.0
        else:
            mean_val = sum(samples_in_window) / len(samples_in_window)

        # Compliance direction depends on SLO type
        if defn.slo_type in (SLOType.LATENCY, SLOType.ERROR_RATE):
            # Lower is better: compliant when mean <= target
            compliant = mean_val <= defn.target
        else:
            # AVAILABILITY / THROUGHPUT: higher is better
            compliant = mean_val >= defn.target

        # Burn rate (meaningful for AVAILABILITY; clamped >= 0)
        if defn.slo_type == SLOType.AVAILABILITY and defn.target < 1.0:
            denominator = 1.0 - defn.target
            compliance_rate = (
                sum(1 for v in samples_in_window if v >= defn.target) / len(samples_in_window)
                if samples_in_window
                else 1.0
            )
            burn_rate = max(0.0, (1.0 - compliance_rate) / denominator)
        else:
            burn_rate = 0.0

        return SLOStatus(
            definition=defn,
            current_value=mean_val,
            compliant=compliant,
            burn_rate=burn_rate,
        )

    def get_all_status(self) -> list[SLOStatus]:
        """Evaluate and return status for every registered SLO."""
        return [self.evaluate(name) for name in self._definitions]


# Module-level singleton
_SLO_TRACKER = SLOTracker()

try:
    from src.monitoring import MONITORING_REGISTRY as _REG  # type: ignore[import]

    _REG["slo_tracker"] = _SLO_TRACKER
except Exception:  # noqa: S110
    pass

MONITORING_REGISTRY: dict[str, object] = {"slo_tracker": _SLO_TRACKER}
