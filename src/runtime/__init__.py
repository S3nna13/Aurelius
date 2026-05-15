"""Aurelius v2 Runtime — Hardware detection, memory budgets, backend selection, profiling."""

from src.runtime.backend_selector import (
    BackendSelection,
    BackendSelector,
    BackendType,
)
from src.runtime.capability_report import CapabilityReport, CapabilityStatus
from src.runtime.hardware_detector import HardwareDetector, HardwareInfo
from src.runtime.memory_budget import MemoryBudgetManager, MemoryBudgetReport, PressureLevel
from src.runtime.profile_schema import (
    ArtifactRef,
    CapabilityMode,
    HardwareProfile,
    RuntimePolicy,
    UserProfile,
)

__all__ = [
    "HardwareProfile",
    "RuntimePolicy",
    "UserProfile",
    "CapabilityMode",
    "ArtifactRef",
    "HardwareDetector",
    "HardwareInfo",
    "MemoryBudgetManager",
    "PressureLevel",
    "MemoryBudgetReport",
    "CapabilityReport",
    "CapabilityStatus",
    "BackendSelector",
    "BackendType",
    "BackendSelection",
]
