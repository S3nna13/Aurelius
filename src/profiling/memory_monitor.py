from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp_s: float
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    label: str = ""


@dataclass
class WatermarkConfig:
    warn_mb: float = 4096.0
    critical_mb: float = 8192.0


def _read_memory_mb() -> tuple[float, float, float]:
    """Return (allocated_mb, reserved_mb, peak_mb)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            return allocated, reserved, peak
    except Exception:
        pass

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = usage.ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's kilobytes.
        # Normalize by assuming kilobytes if value is small-ish.
        import sys

        if sys.platform == "darwin":
            rss_mb = rss_kb / (1024.0 * 1024.0)
        else:
            rss_mb = rss_kb / 1024.0
        return rss_mb, rss_mb, rss_mb
    except Exception:
        return 0.0, 0.0, 0.0


class MemoryMonitor:
    def __init__(self, watermark: WatermarkConfig | None = None):
        self._watermark = watermark if watermark is not None else WatermarkConfig()
        self._history: list[MemorySnapshot] = []

    def snapshot(self, label: str = "") -> MemorySnapshot:
        allocated, reserved, peak = _read_memory_mb()
        return MemorySnapshot(
            timestamp_s=time.monotonic(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            peak_mb=peak,
            label=label,
        )

    def set_watermark(self, config: WatermarkConfig) -> None:
        self._watermark = config

    def check_watermarks(self, snap: MemorySnapshot) -> list[str]:
        warnings: list[str] = []
        if snap.allocated_mb >= self._watermark.critical_mb:
            warnings.append(
                f"CRITICAL: allocated_mb {snap.allocated_mb:.2f} exceeds critical threshold "
                f"{self._watermark.critical_mb:.2f}"
            )
        elif snap.allocated_mb >= self._watermark.warn_mb:
            warnings.append(
                f"WARN: allocated_mb {snap.allocated_mb:.2f} exceeds warn threshold "
                f"{self._watermark.warn_mb:.2f}"
            )
        if snap.reserved_mb >= self._watermark.critical_mb:
            warnings.append(
                f"CRITICAL: reserved_mb {snap.reserved_mb:.2f} exceeds critical threshold "
                f"{self._watermark.critical_mb:.2f}"
            )
        return warnings

    def record(self, label: str = "") -> MemorySnapshot:
        snap = self.snapshot(label)
        self._history.append(snap)
        return snap

    def history(self) -> list[MemorySnapshot]:
        return list(self._history)

    def peak_snapshot(self) -> MemorySnapshot | None:
        if not self._history:
            return None
        return max(self._history, key=lambda s: s.allocated_mb)

    def reset(self) -> None:
        self._history.clear()


MEMORY_MONITOR_REGISTRY: dict[str, type[MemoryMonitor]] = {
    "default": MemoryMonitor,
}
