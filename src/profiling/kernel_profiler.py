from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KernelRecord:
    kernel_name: str
    grid_size: tuple[int, ...]
    block_size: tuple[int, ...]
    duration_us: float
    memory_bytes: int = 0

    def throughput_gflops(self, flops: int) -> float:
        """flops / (duration_us * 1e-6) / 1e9"""
        if self.duration_us <= 0:
            return 0.0
        return flops / (self.duration_us * 1e-6) / 1e9


class KernelProfiler:
    """Profiles compute kernels (CPU/GPU ops), stdlib-only stub."""

    def __init__(self) -> None:
        self._records: list[KernelRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_kernel(
        self,
        kernel_name: str,
        grid_size: tuple,
        block_size: tuple,
        duration_us: float,
        memory_bytes: int = 0,
    ) -> KernelRecord:
        record = KernelRecord(
            kernel_name=kernel_name,
            grid_size=tuple(grid_size),
            block_size=tuple(block_size),
            duration_us=duration_us,
            memory_bytes=memory_bytes,
        )
        self._records.append(record)
        return record

    def top_kernels(self, n: int = 10) -> list[KernelRecord]:
        """Return top-n kernels sorted by duration_us descending."""
        sorted_records = sorted(self._records, key=lambda r: r.duration_us, reverse=True)
        return sorted_records[:n]

    def summary_by_name(self) -> dict[str, dict]:
        """Aggregate stats grouped by kernel name."""
        grouped: dict[str, list[float]] = {}
        for r in self._records:
            grouped.setdefault(r.kernel_name, []).append(r.duration_us)
        result: dict[str, dict] = {}
        for name, durations in grouped.items():
            total = sum(durations)
            result[name] = {
                "count": len(durations),
                "total_us": total,
                "mean_us": total / len(durations),
            }
        return result

    def total_time_us(self) -> float:
        return sum(r.duration_us for r in self._records)

    def reset(self) -> None:
        self._records.clear()


KERNEL_PROFILER_REGISTRY: dict[str, type[KernelProfiler]] = {
    "default": KernelProfiler
}
