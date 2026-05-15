"""Aurelius v2 Runtime — Hardware detection, memory budgets, backend selection, profiling."""

from src.runtime.profile_schema import (
    HardwareProfile,
    RuntimePolicy,
    UserProfile,
    CapabilityMode,
    ArtifactRef,
)
from src.runtime.hardware_detector import HardwareDetector, HardwareInfo
from src.runtime.memory_budget import MemoryBudgetManager, PressureLevel, MemoryBudgetReport
from src.runtime.capability_report import CapabilityReport, CapabilityStatus
from src.runtime.backend_selector import (
    BackendSelector,
    BackendType,
    BackendSelection,
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
