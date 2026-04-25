"""Power consumption estimation (TDP-based, stdlib-only)."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerReading:
    timestamp: float
    power_w: float
    component: str
    energy_since_start_j: float = 0.0


@dataclass(frozen=True)
class TDPConfig:
    cpu_tdp_w: float = 65.0
    gpu_tdp_w: float = 300.0
    memory_tdp_w: float = 10.0


class PowerProfiler:
    def __init__(
        self,
        tdp_config: TDPConfig | None = None,
        max_readings: int = 10000,
    ) -> None:
        self.tdp_config = tdp_config if tdp_config is not None else TDPConfig()
        self.max_readings = max_readings
        self._readings: list[PowerReading] = []
        self._last_timestamp: float | None = None
        self._cumulative_energy_j: float = 0.0

    def record(self, power_w: float, component: str = "total") -> PowerReading:
        now = time.monotonic()
        if self._last_timestamp is None:
            elapsed = 0.0
        else:
            elapsed = now - self._last_timestamp
        self._cumulative_energy_j += power_w * elapsed
        reading = PowerReading(
            timestamp=now,
            power_w=power_w,
            component=component,
            energy_since_start_j=self._cumulative_energy_j,
        )
        self._last_timestamp = now
        if len(self._readings) >= self.max_readings:
            self._readings.pop(0)
        self._readings.append(reading)
        return reading

    def total_energy_j(self) -> float:
        if len(self._readings) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._readings)):
            interval = self._readings[i].timestamp - self._readings[i - 1].timestamp
            total += self._readings[i - 1].power_w * interval
        return total

    def mean_power_w(self, component: str | None = None) -> float:
        readings = self._filter(component)
        if not readings:
            return 0.0
        return statistics.mean(r.power_w for r in readings)

    def peak_power_w(self, component: str | None = None) -> float:
        readings = self._filter(component)
        if not readings:
            return 0.0
        return max(r.power_w for r in readings)

    def efficiency_score(self) -> float:
        total_readings = self._readings
        if not total_readings:
            mean_total = 0.0
        else:
            mean_total = statistics.mean(r.power_w for r in total_readings)
        sum_tdp = (
            self.tdp_config.cpu_tdp_w
            + self.tdp_config.gpu_tdp_w
            + self.tdp_config.memory_tdp_w
        )
        if sum_tdp <= 0:
            return 1.0
        ratio = mean_total / sum_tdp
        ratio = max(0.0, min(1.0, ratio))
        return 1.0 - ratio

    def readings_for(self, component: str) -> list[PowerReading]:
        return [r for r in self._readings if r.component == component]

    def _filter(self, component: str | None) -> list[PowerReading]:
        if component is None:
            return self._readings
        return [r for r in self._readings if r.component == component]


POWER_PROFILER_REGISTRY: dict[str, object] = {"default": PowerProfiler}