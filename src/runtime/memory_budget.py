"""Aurelius v2 Memory Budget Manager — tracks RAM/VRAM, computes budgets, pressure detection, degradation ladder."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class PressureLevel(enum.StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MemoryBudgetReport:
    """Report of current memory allocation and pressure."""

    total_memory_gb: float
    reserved_for_os_gb: float
    available_for_aurelius_gb: float
    weights_gb: float
    kv_cache_gb: float
    runtime_memory_gb: float
    vision_gb: float
    prefix_cache_gb: float
    skill_runtime_gb: float
    tool_buffers_gb: float
    safety_margin_gb: float
    free_gb: float
    used_gb: float
    pressure_level: PressureLevel
    recommended_model: str | None = None
    recommended_artifact: str | None = None
    max_context: int = 0
    active_degradation_steps: list[str] = field(default_factory=list)


@dataclass
class DegradationStep:
    """A single step in the degradation ladder."""

    order: int
    name: str
    description: str
    estimated_savings_gb: float
    applied: bool = False


@dataclass
class MemoryBudgetConfig:
    """Configuration for the memory budget manager."""

    total_memory_gb: float
    reserved_for_os_gb: float = 2.0
    safety_reserve_gb: float = 2.0
    weights_gb: float = 0.0
    expected_kv_gb: float = 0.0
    context_tokens: int = 32768


class MemoryBudgetManager:
    """Tracks memory consumers and manages the degradation ladder.

    Degradation steps (applied in order when pressure is HIGH or above):
      1. Unload inactive skills
      2. Reduce batch size to 1
      3. Reduce context budget
      4. Quantize KV cache (KIVI-style)
      5. Shrink active prompt context
      6. Evict prefix cache
      7. Offload cold KV to CPU/unified memory
      8. Downshift or disable vision encoder
      9. Switch to smaller model artifact
     10. Switch to split/remote execution mode
     11. Ask user before cancelling
    """

    DEGRADATION_LADDER: list[DegradationStep] = [
        DegradationStep(1, "unload_inactive_skills", "Release memory from skills not actively in use", 0.5),
        DegradationStep(2, "reduce_batch_size", "Drop batch size to 1", 0.3),
        DegradationStep(3, "reduce_context", "Shrink max context: 128K -> 64K -> 32K -> 16K", 1.0),
        DegradationStep(4, "quantize_kv_cache", "Apply 2-bit/4-bit KV quantization", 2.0),
        DegradationStep(5, "shrink_active_prompt", "Compress or truncate non-essential context", 0.5),
        DegradationStep(6, "evict_prefix_cache", "Drop cached prefix/suffix blocks", 1.0),
        DegradationStep(7, "offload_cold_kv", "Move least-recently-accessed KV to CPU/unified memory", 2.0),
        DegradationStep(8, "downshift_vision", "Disable VLM encoder or reduce image resolution", 1.5),
        DegradationStep(9, "switch_artifact", "Fall back to smaller/quantized model artifact", 3.0),
        DegradationStep(10, "switch_split_remote", "Offload inference to remote endpoint", 5.0),
        DegradationStep(11, "ask_user_before_cancel", "Present user options before aborting task", 0.0),
    ]

    # Pressure thresholds (fraction of available memory used)
    PRESSURE_THRESHOLDS: dict[PressureLevel, float] = {
        PressureLevel.LOW: 0.0,
        PressureLevel.MODERATE: 0.50,
        PressureLevel.HIGH: 0.70,
        PressureLevel.CRITICAL: 0.85,
        PressureLevel.EMERGENCY: 0.95,
    }

    def __init__(self, config: MemoryBudgetConfig) -> None:
        self.config = config
        self.available_for_aurelius = config.total_memory_gb - config.reserved_for_os_gb
        self.active_degradation_steps: list[int] = []  # indices into DEGRADATION_LADDER
        self._current_values: dict[str, float] = {
            "weights_gb": config.weights_gb,
            "kv_cache_gb": config.expected_kv_gb,
            "runtime_memory_gb": 1.0,
            "vision_gb": 1.5,
            "prefix_cache_gb": 0.4,
            "skill_runtime_gb": 0.2,
            "tool_buffers_gb": 0.1,
        }

    def update_consumer(self, consumer: str, gb_value: float) -> None:
        """Update a memory consumer's allocation in GB."""
        valid_consumers = {
            "weights_gb", "kv_cache_gb", "runtime_memory_gb", "vision_gb",
            "prefix_cache_gb", "skill_runtime_gb", "tool_buffers_gb",
        }
        if consumer in valid_consumers:
            self._current_values[consumer] = gb_value

    def get_used_total_gb(self) -> float:
        """Sum all current memory consumers."""
        return sum(self._current_values.values())

    def get_free_gb(self) -> float:
        """Remaining memory in the Aurelius budget."""
        used = self.get_used_total_gb()
        free = self.available_for_aurelius - used - self.config.safety_reserve_gb
        return max(0.0, free)

    def pressure_fraction(self) -> float:
        """Return the fraction of available memory that is used."""
        if self.available_for_aurelius <= 0:
            return 1.0
        return min(1.0, self.get_used_total_gb() / self.available_for_aurelius)

    def pressure_level(self) -> PressureLevel:
        """Determine current pressure level."""
        fraction = self.pressure_fraction()
        if fraction >= self.PRESSURE_THRESHOLDS[PressureLevel.EMERGENCY]:
            return PressureLevel.EMERGENCY
        if fraction >= self.PRESSURE_THRESHOLDS[PressureLevel.CRITICAL]:
            return PressureLevel.CRITICAL
        if fraction >= self.PRESSURE_THRESHOLDS[PressureLevel.HIGH]:
            return PressureLevel.HIGH
        if fraction >= self.PRESSURE_THRESHOLDS[PressureLevel.MODERATE]:
            return PressureLevel.MODERATE
        return PressureLevel.LOW

    def should_degrade(self) -> bool:
        """Check whether to apply degradation steps."""
        return self.pressure_level() in (PressureLevel.HIGH, PressureLevel.CRITICAL, PressureLevel.EMERGENCY)

    def apply_next_degradation(self) -> DegradationStep | None:
        """Apply the next degradation step that hasn't been applied yet."""
        for step in self.DEGRADATION_LADDER:
            if step.order not in self.active_degradation_steps and not step.applied:
                step.applied = True
                self.active_degradation_steps.append(step.order)
                # Simulate savings by reducing consumers
                consumer_map = {
                    "unload_inactive_skills": "skill_runtime_gb",
                    "reduce_batch_size": "runtime_memory_gb",
                    "reduce_context": "kv_cache_gb",
                    "quantize_kv_cache": "kv_cache_gb",
                    "shrink_active_prompt": "kv_cache_gb",
                    "evict_prefix_cache": "prefix_cache_gb",
                    "offload_cold_kv": "kv_cache_gb",
                    "downshift_vision": "vision_gb",
                }
                consumer = consumer_map.get(step.name)
                if consumer and consumer in self._current_values:
                    reduction = min(step.estimated_savings_gb, self._current_values[consumer])
                    self._current_values[consumer] = max(0.0, self._current_values[consumer] - reduction)
                return step
        return None

    def reset_degradation(self) -> None:
        """Reset all degradation steps (e.g., after memory freed)."""
        for step in self.DEGRADATION_LADDER:
            step.applied = False
        self.active_degradation_steps.clear()

    def generate_report(self) -> MemoryBudgetReport:
        """Generate a full memory budget report."""
        used = self.get_used_total_gb()
        free = self.get_free_gb()
        level = self.pressure_level()
        steps = [s.name for s in self.DEGRADATION_LADDER if s.applied]

        # Derive recommendations based on pressure
        recommended_model: str | None = None
        recommended_artifact: str | None = None
        if level == PressureLevel.EMERGENCY and "switch_artifact" not in steps:
            recommended_model = "swift"
            recommended_artifact = "swift-q3-gguf"
        elif level == PressureLevel.CRITICAL and "switch_artifact" not in steps:
            recommended_model = "forge"
            recommended_artifact = "forge-q4-gguf"

        # Derive max context based on available memory
        max_context_map = {
            PressureLevel.LOW: 131072,
            PressureLevel.MODERATE: 65536,
            PressureLevel.HIGH: 32768,
            PressureLevel.CRITICAL: 16384,
            PressureLevel.EMERGENCY: 4096,
        }

        return MemoryBudgetReport(
            total_memory_gb=self.config.total_memory_gb,
            reserved_for_os_gb=self.config.reserved_for_os_gb,
            available_for_aurelius_gb=self.available_for_aurelius,
            weights_gb=self._current_values.get("weights_gb", 0),
            kv_cache_gb=self._current_values.get("kv_cache_gb", 0),
            runtime_memory_gb=self._current_values.get("runtime_memory_gb", 0),
            vision_gb=self._current_values.get("vision_gb", 0),
            prefix_cache_gb=self._current_values.get("prefix_cache_gb", 0),
            skill_runtime_gb=self._current_values.get("skill_runtime_gb", 0),
            tool_buffers_gb=self._current_values.get("tool_buffers_gb", 0),
            safety_margin_gb=self.config.safety_reserve_gb,
            free_gb=free,
            used_gb=used,
            pressure_level=level,
            recommended_model=recommended_model,
            recommended_artifact=recommended_artifact,
            max_context=max_context_map.get(level, 32768),
            active_degradation_steps=steps,
        )
