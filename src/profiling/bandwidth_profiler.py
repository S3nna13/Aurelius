from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BandwidthSample:
    timestamp: float
    bytes_read: int
    bytes_written: int
    duration_ms: float

    @property
    def read_bandwidth_gbps(self) -> float:
        """bytes_read / (duration_ms * 1e-3) / 1e9"""
        if self.duration_ms <= 0:
            return 0.0
        return self.bytes_read / (self.duration_ms * 1e-3) / 1e9

    @property
    def write_bandwidth_gbps(self) -> float:
        """bytes_written / (duration_ms * 1e-3) / 1e9"""
        if self.duration_ms <= 0:
            return 0.0
        return self.bytes_written / (self.duration_ms * 1e-3) / 1e9

    @property
    def total_bandwidth_gbps(self) -> float:
        """Sum of read and write bandwidth."""
        return self.read_bandwidth_gbps + self.write_bandwidth_gbps


class BandwidthProfiler:
    """Profiles memory bandwidth utilization."""

    def __init__(self, peak_bandwidth_gbps: float = 900.0) -> None:
        self._peak_gbps = peak_bandwidth_gbps
        self._samples: list[BandwidthSample] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, bytes_read: int, bytes_written: int, duration_ms: float) -> BandwidthSample:
        """Create and store a BandwidthSample with timestamp=time.monotonic()."""
        sample = BandwidthSample(
            timestamp=time.monotonic(),
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            duration_ms=duration_ms,
        )
        self._samples.append(sample)
        return sample

    def utilization(self, sample: BandwidthSample) -> float:
        """total_bandwidth_gbps / peak * 100.0, clamped to [0, 100]."""
        if self._peak_gbps <= 0:
            return 0.0
        raw = sample.total_bandwidth_gbps / self._peak_gbps * 100.0
        return max(0.0, min(100.0, raw))

    def summary(self) -> dict:
        n = len(self._samples)
        if n == 0:
            return {
                "samples": 0,
                "mean_read_gbps": 0.0,
                "mean_write_gbps": 0.0,
                "peak_utilization_pct": 0.0,
            }
        mean_read = sum(s.read_bandwidth_gbps for s in self._samples) / n
        mean_write = sum(s.write_bandwidth_gbps for s in self._samples) / n
        peak_util = max(self.utilization(s) for s in self._samples)
        return {
            "samples": n,
            "mean_read_gbps": mean_read,
            "mean_write_gbps": mean_write,
            "peak_utilization_pct": peak_util,
        }

    def reset(self) -> None:
        self._samples.clear()


BANDWIDTH_PROFILER_REGISTRY: dict[str, type[BandwidthProfiler]] = {"default": BandwidthProfiler}
