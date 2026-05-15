"""Aurelius CLI — Runtime client that bridges CLI commands to the Python runtime.

Provides unified access to hardware detection, backend selection, memory budgets,
capability reports, skill system, CUA, and other runtime subsystems.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed status objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeStatus:
    """Aggregated runtime state for CLI consumption."""

    model_name: str
    family: str
    backend: str
    quantization: str
    context_length: int
    capability_mode: str
    profile_id: str
    ram_total_gb: float
    ram_used_gb: float
    vram_total_gb: float
    vram_used_gb: float
    skill_count: int
    tool_count: int
    cua_mode: str
    mcp_server_count: int
    session_count: int


@dataclass(frozen=True)
class SkillInfo:
    """Describes a single skill."""

    id: str
    name: str
    category: str
    enabled: bool
    version: str = ""
    description: str = ""


@dataclass(frozen=True)
class ToolInfo:
    """Describes a single tool."""

    id: str
    name: str
    enabled: bool
    requires_permission: bool = False
    description: str = ""


@dataclass(frozen=True)
class HardwareSnapshot:
    """Snapshot of current hardware state."""

    system: str
    machine: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: str
    gpu_vram_gb: float
    cuda_available: bool
    mps_available: bool
    mlx_available: bool
    unified_memory: bool
    ram_used_gb: float = 0.0


# ---------------------------------------------------------------------------
# Protocol layer (runtime subsystem interface)
# ---------------------------------------------------------------------------

class RuntimeSubsystems(Protocol):
    """Protocol exposing runtime subsystems the CLI needs to talk to."""

    def detect_hardware(self) -> HardwareSnapshot:
        ...

    def select_backend(self, model: str, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_capability_report(self) -> dict[str, Any]:
        ...

    def list_skills(self, category: str | None = None) -> list[SkillInfo]:
        ...

    def list_tools(self) -> list[ToolInfo]:
        ...

    def get_memory_status(self) -> dict[str, float]:
        ...

    def get_session_count(self) -> int:
        ...


# ---------------------------------------------------------------------------
# Default implementation using src.runtime modules
# ---------------------------------------------------------------------------

class _DefaultRuntimeSubsystems:
    """Default implementation that delegates to src.runtime modules."""

    def __init__(self) -> None:
        self._sys_info_cache: HardwareSnapshot | None = None
        self._memory_budget_manager: Any = None
        self._backend_selector: Any = None
        self._hardware_detector: Any = None

    def _ensure_runtime_available(self) -> bool:
        """Check if the full src.runtime is importable. Graceful degradation."""
        try:
            import src.runtime  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def detect_hardware(self) -> HardwareSnapshot:
        """Detect hardware using platform + runtime modules."""
        if self._hardware_detector is None and self._ensure_runtime_available():
            from src.runtime.hardware_detector import HardwareDetector
            self._hardware_detector = HardwareDetector()

        if self._hardware_detector:
            info = self._hardware_detector.detect()
            return HardwareSnapshot(
                system=platform.system(),
                machine=platform.machine(),
                cpu_count=os.cpu_count() or 0,
                ram_total_gb=info.total_ram_gb,
                ram_available_gb=getattr(info, "available_ram_gb", info.total_ram_gb * 0.5),
                gpu_name=info.gpu_name if hasattr(info, "gpu_name") else "",
                gpu_vram_gb=info.gpu_vram_gb,
                cuda_available=info.cuda_available,
                mps_available=getattr(info, "mps_available", False),
                mlx_available=info.mlx_available,
                unified_memory=info.unified_memory,
            )

        # Fallback: basic platform detection
        ram_total = self._get_ram_gb()
        return HardwareSnapshot(
            system=platform.system(),
            machine=platform.machine(),
            cpu_count=os.cpu_count() or 0,
            ram_total_gb=ram_total,
            ram_available_gb=ram_total * 0.5,
            gpu_name="",
            gpu_vram_gb=0.0,
            cuda_available=False,
            mps_available=False,
            mlx_available=False,
            unified_memory=False,
        )

    def select_backend(self, model: str, **kwargs: Any) -> dict[str, Any]:
        """Select the optimal backend via src.runtime.backend_selector."""
        if self._backend_selector is None and self._ensure_runtime_available():
            from src.runtime.backend_selector import BackendSelector
            self._backend_selector = BackendSelector()

        if self._backend_selector and self._hardware_detector:
            hw = self._hardware_detector.detect()
            selection = self._backend_selector.select(model, hw)
            return {
                "backend": selection.backend.value,
                "quantization": selection.quantization,
                "context_budget": selection.context_budget,
                "capability_mode": selection.capability_mode.value,
                "skill_preload_count": selection.skill_preload_count,
                "reasons": selection.reasons,
            }

        return {
            "backend": "mock",
            "quantization": "q4",
            "context_budget": 8192,
            "capability_mode": "reduced_local",
            "skill_preload_count": 0,
            "reasons": [f"Mock backend for {model} — runtime not fully initialized"],
        }

    def get_capability_report(self) -> dict[str, Any]:
        """Get capability report."""
        if not self._ensure_runtime_available():
            return {
                "model": "mock",
                "backend": "mock",
                "capability_mode": "reduced_local",
                "label": "Mock (runtime not initialized)",
            }

        from src.runtime.capability_report import CapabilityReport
        report = CapabilityReport.create_full_local(
            model="forge",
            backend="mock",
            artifact="",
            quantization="q4",
            context=8192,
            hardware="default",
        )
        return report.to_dict()

    def list_skills(self, category: str | None = None) -> list[SkillInfo]:
        """List skills from the builtin skill registry."""
        skills_base = Path(__file__).resolve().parents[1] / "src" / "skills" / "builtin"
        if not skills_base.exists():
            return []

        results: list[SkillInfo] = []
        for cat_dir in sorted(skills_base.iterdir()):
            if not cat_dir.is_dir():
                continue
            if category and cat_dir.name != category:
                continue
            for skill_dir in sorted(cat_dir.iterdir()):
                skill_json = skill_dir / "skill.json"
                if skill_json.exists():
                    import json
                    try:
                        with open(skill_json) as f:
                            data = json.load(f)
                        results.append(SkillInfo(
                            id=data.get("id", skill_dir.name),
                            name=data.get("name", skill_dir.name),
                            category=cat_dir.name,
                            enabled=data.get("enabled", True),
                            version=data.get("version", ""),
                            description=data.get("description", ""),
                        ))
                    except (json.JSONDecodeError, OSError):
                        results.append(SkillInfo(
                            id=skill_dir.name,
                            name=skill_dir.name,
                            category=cat_dir.name,
                            enabled=True,
                        ))
                else:
                    results.append(SkillInfo(
                        id=skill_dir.name,
                        name=skill_dir.name,
                        category=cat_dir.name,
                        enabled=True,
                    ))
        return results

    def list_tools(self) -> list[ToolInfo]:
        """List tools from the src.tools directory."""
        tools_dir = Path(__file__).resolve().parents[1] / "src" / "tools"
        if not tools_dir.exists():
            return []

        results: list[ToolInfo] = []
        for tool_file in sorted(tools_dir.iterdir()):
            if not tool_file.is_file() or tool_file.name.startswith("_"):
                continue
            if not tool_file.name.endswith(".py"):
                continue
            name = tool_file.stem
            results.append(ToolInfo(
                id=f"tool:{name}",
                name=name,
                enabled=True,
                requires_permission="shell" in name or "exec" in name or "file" in name,
                description="",
            ))
        return results

    def get_memory_status(self) -> dict[str, float]:
        """Get current RAM/VRAM usage."""
        result: dict[str, float] = {
            "ram_total_gb": 0.0,
            "ram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
        }

        try:
            import psutil
            mem = psutil.virtual_memory()
            result["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
            result["ram_used_gb"] = round(mem.used / (1024 ** 3), 1)
        except ImportError:
            # Fallback without psutil
            hw = self.detect_hardware()
            result["ram_total_gb"] = hw.ram_total_gb
            result["ram_used_gb"] = round(hw.ram_total_gb * 0.5, 1)  # rough estimate

        # Try to get GPU memory
        if platform.system() == "Darwin":
            result["vram_total_gb"] = result["ram_total_gb"]  # unified
            result["vram_used_gb"] = result["ram_used_gb"]
        else:
            try:
                import subprocess as _sub

                output = _sub.check_output(
                    ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                     "--format=csv,noheader,nounits"],
                    timeout=5, text=True,
                )
                parts = output.strip().split(",")
                if len(parts) >= 2:
                    result["vram_total_gb"] = round(float(parts[0].strip()) / 1024, 1)
                    result["vram_used_gb"] = round(float(parts[1].strip()) / 1024, 1)
            except (FileNotFoundError, ValueError, OSError):
                pass

        return result

    def get_session_count(self) -> int:
        """Approximate active session count."""
        try:
            from src.runtime.session_manager import SessionManager
            mgr = SessionManager()
            return mgr.active_count()
        except (ImportError, AttributeError):
            return 0

    @staticmethod
    def _get_ram_gb() -> float:
        """Get RAM in GB via platform-specific methods."""
        if platform.system() == "Darwin":
            try:
                import subprocess as _sub2
                output = _sub2.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
                )
                return round(int(output.strip()) / (1024 ** 3), 1)
            except (OSError, ValueError):
                return 8.0
        elif platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return round(kb / (1024 ** 2), 1)
            except OSError:
                pass
        return 8.0



# ---------------------------------------------------------------------------
# Public runtime client
# ---------------------------------------------------------------------------

class RuntimeClient:
    """High-level CLI-to-runtime bridge.

    All CLI commands should go through this client.  It handles lazy
    initialisation, graceful degradation, and caching.
    """

    def __init__(self, subsystems: RuntimeSubsystems | None = None) -> None:
        self._subsystems = subsystems or _DefaultRuntimeSubsystems()
        self._hardware_cache: HardwareSnapshot | None = None
        self._memory_cache: dict[str, float] | None = None

    # -- Hardware -----------------------------------------------------------

    def detect_hardware(self) -> HardwareSnapshot:
        if self._hardware_cache is None:
            self._hardware_cache = self._subsystems.detect_hardware()
        return self._hardware_cache

    # -- Backend selection --------------------------------------------------

    def select_backend(self, model: str, profile: str = "default") -> dict[str, Any]:
        return self._subsystems.select_backend(model)

    # -- Capability report --------------------------------------------------

    def capability_report(self) -> dict[str, Any]:
        return self._subsystems.get_capability_report()

    # -- Skills -------------------------------------------------------------

    def list_skills(
        self, category: str | None = None, enabled_only: bool = False,
    ) -> list[SkillInfo]:
        skills = self._subsystems.list_skills(category)
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return skills

    # -- Tools --------------------------------------------------------------

    def list_tools(self, enabled_only: bool = False) -> list[ToolInfo]:
        tools = self._subsystems.list_tools()
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools

    # -- Memory -------------------------------------------------------------

    def memory_status(self) -> dict[str, float]:
        self._memory_cache = self._subsystems.get_memory_status()
        return self._memory_cache

    # -- Sessions -----------------------------------------------------------

    def session_count(self) -> int:
        return self._subsystems.get_session_count()

    # -- Aggregated status --------------------------------------------------

    def get_runtime_status(self, config: Any = None) -> RuntimeStatus:
        """Build a complete RuntimeStatus for the status bar."""
        if config is None:
            model_name = "forge"
            family = "forge"
            profile = "default"
            backend = ""
            quant = "q4"
            ctx = 32768
            capability_mode = "reduced_local"
            cua_mode = "off"
        else:
            model_name = getattr(config.model, "name", "forge")
            family = getattr(config.model, "family", "forge")
            profile = getattr(config.runtime, "profile", "default")
            backend = getattr(config.model, "backend", "")
            quant = getattr(config.model, "quantization", "")
            ctx = getattr(config.model, "context_length", 32768)
            capability_mode = getattr(config.runtime, "capability_mode", "")
            cua_mode = "local_full" if getattr(config.cua, "enabled", False) else "off"

        # Select backend if not already configured
        if not backend:
            backend_info = self.select_backend(model_name)
            backend = backend_info.get("backend", "mock")
            if not quant:
                quant = backend_info.get("quantization", "q4")
            if not capability_mode:
                capability_mode = backend_info.get("capability_mode", "reduced_local")

        mem = self.memory_status()
        skills = self.list_skills(enabled_only=False)
        tools = self.list_tools(enabled_only=False)

        ram_total_val: float = mem.get("ram_total_gb", 0.0) or 0.0
        ram_avail_val: float = mem.get("ram_available_gb", 0.0) or 0.0
        ram_used_val: float = mem.get("ram_used_gb", max(ram_total_val - ram_avail_val, 0.0)) or 0.0

        return RuntimeStatus(
            model_name=model_name,
            family=family,
            backend=backend,
            quantization=quant,
            context_length=ctx,
            capability_mode=capability_mode,
            profile_id=profile,
            ram_total_gb=ram_total_val,
            ram_used_gb=ram_used_val,
            vram_total_gb=mem.get("vram_total_gb", 0.0) or 0.0,
            vram_used_gb=mem.get("vram_used_gb", 0.0) or 0.0,
            skill_count=len(skills),
            tool_count=len(tools),
            cua_mode=cua_mode,
            mcp_server_count=0,
            session_count=self.session_count(),
        )

    def refresh(self) -> None:
        """Clear cached state."""
        self._hardware_cache = None
        self._memory_cache = None
