"""Power profiling: energy readings, TDP comparison, efficiency scoring."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PowerReading:
    timestamp: float
    power_w: float
    component: str = "gpu"


@dataclass
class TDPConfig:
    tdp_w: float = 400.0
    peak_tdp_w: float = 600.0


@dataclass
class PowerProfiler:
    max_readings: int = 1000
    _readings: list[PowerReading] = field(default_factory=list)

    @property
    def readings(self) -> list[PowerReading]:
        return self._readings

    def record(self, power_w: float, component: str = "gpu") -> PowerReading:
        reading = PowerReading(timestamp=time.monotonic(), power_w=power_w, component=component)
        self._readings.append(reading)
        if len(self._readings) > self.max_readings:
            self._readings.pop(0)
        return reading

    def total_energy_j(self) -> float:
        if len(self._readings) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._readings)):
            dt = self._readings[i].timestamp - self._readings[i - 1].timestamp
            avg_power = (self._readings[i].power_w + self._readings[i - 1].power_w) / 2.0
            total += avg_power * dt
        return total

    def mean_power_w(self) -> float:
        if not self._readings:
            return 0.0
        return sum(r.power_w for r in self._readings) / len(self._readings)

    def peak_power_w(self) -> float:
        if not self._readings:
            return 0.0
        return max(r.power_w for r in self._readings)

    def efficiency_score(self, tdp_config: TDPConfig | None = None) -> float:
        if not self._readings:
            return 1.0
        tdp = tdp_config.tdp_w if tdp_config else 400.0
        avg_power = self.mean_power_w()
        if tdp <= 0:
            return 1.0
        raw = avg_power / tdp
        return max(0.0, min(1.0, raw))

    def readings_for(self, component: str) -> list[PowerReading]:
        return [r for r in self._readings if r.component == component]

    def _filter(self, component: str | None = None) -> list[PowerReading]:
        if component is None:
            return list(self._readings)
        return [r for r in self._readings if r.component == component]

    def clear(self) -> None:
        self._readings.clear()

    def __len__(self) -> int:
        return len(self._readings)


POWER_PROFILER_REGISTRY: dict[str, object] = {"default": PowerProfiler()}