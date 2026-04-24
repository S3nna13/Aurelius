"""Scheduled quantization: gradually reduce bit-width during training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class QuantSchedule(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    CONSTANT = "constant"


@dataclass(frozen=True)
class QuantizationStep:
    step: int
    bits: float
    scale: float


class QuantizationScheduler:
    """Schedule bit-width across training steps."""

    def __init__(
        self,
        initial_bits: float = 16.0,
        final_bits: float = 4.0,
        total_steps: int = 1000,
        schedule: QuantSchedule = QuantSchedule.LINEAR,
        step_interval: int = 100,
    ) -> None:
        self.initial_bits = float(initial_bits)
        self.final_bits = float(final_bits)
        self.total_steps = max(1, int(total_steps))
        self.schedule = schedule
        self.step_interval = max(1, int(step_interval))

    def _clamp(self, bits: float) -> float:
        lo = min(self.initial_bits, self.final_bits)
        hi = max(self.initial_bits, self.final_bits)
        return max(lo, min(hi, bits))

    def bits_at(self, step: int) -> float:
        step = max(0, int(step))
        init = self.initial_bits
        fin = self.final_bits
        T = self.total_steps

        if self.schedule == QuantSchedule.CONSTANT:
            return init

        if self.schedule == QuantSchedule.LINEAR:
            bits = init - (init - fin) * step / T
            return self._clamp(bits)

        if self.schedule == QuantSchedule.COSINE:
            s = min(step, T)
            bits = fin + 0.5 * (init - fin) * (1.0 + math.cos(math.pi * s / T))
            return self._clamp(bits)

        if self.schedule == QuantSchedule.STEP:
            if step < self.step_interval:
                return init
            num_buckets = max(1, T // self.step_interval)
            delta = (init - fin) / num_buckets
            bits = init - (step // self.step_interval) * delta
            return self._clamp(bits)

        return init

    def should_quantize(self, step: int) -> bool:
        return self.bits_at(step) < self.initial_bits - 0.5

    def get_step(self, step: int) -> QuantizationStep:
        bits = self.bits_at(step)
        safe = bits if bits != 0.0 else 1e-10
        scale = self.initial_bits / safe
        return QuantizationStep(step=int(step), bits=bits, scale=scale)

    def schedule_summary(
        self, num_checkpoints: int = 10
    ) -> list[QuantizationStep]:
        num_checkpoints = max(1, int(num_checkpoints))
        if num_checkpoints == 1:
            return [self.get_step(0)]
        out: list[QuantizationStep] = []
        for i in range(num_checkpoints):
            s = int(round(i * self.total_steps / (num_checkpoints - 1)))
            out.append(self.get_step(s))
        return out


QUANTIZATION_SCHEDULER_REGISTRY: dict[str, type[QuantizationScheduler]] = {
    "default": QuantizationScheduler,
}
