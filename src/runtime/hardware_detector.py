"""Aurelius v2 Hardware Detector — auto-detects CPU, GPU, memory, and recommends profiles."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

from src.runtime.profile_schema import ArtifactRef, HardwareProfile, QuantLevel


@dataclass
class HardwareInfo:
    """Raw detection results from all available probes."""

    cpu_arch: str = ""
    cpu_count: int = 0
    cpu_brand: str = ""
    total_ram_gb: float = 0.0
    gpu_count: int = 0
    gpu_info: list[dict[str, Any]] = field(default_factory=list)
    unified_memory: bool = False
    unified_memory_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: str = ""
    cuda_device_count: int = 0
    cuda_devices: list[dict[str, Any]] = field(default_factory=list)
    mlx_available: bool = False
    metal_available: bool = False
    tensorrt_available: bool = False
    tensorrt_version: str = ""
    os_name: str = ""
    is_jetson: bool = False
    jetson_model: str = ""


class HardwareDetector:
    """Detects hardware capabilities and maps them to Aurelius HardwareProfiles."""

    @staticmethod
    def detect() -> HardwareInfo:
        """Run full hardware detection across all available probes."""
        info = HardwareInfo()

        # Basic system info
        info.cpu_arch = platform.machine()
        info.cpu_count = _cpu_count()
        info.total_ram_gb = _total_ram_gb()
        info.os_name = platform.system()

        # Platform-specific detection
        if info.os_name == "Darwin":
            _detect_macos(info)
        elif info.os_name == "Linux":
            _detect_linux(info)
        elif info.os_name == "Windows":
            _detect_windows(info)

        # Always probe GPU via NVIDIA tools if available
        _probe_cuda(info)

        # Try MLX on macOS
        if info.os_name == "Darwin" and not info.mlx_available:
            _probe_mlx(info)

        return info

    @staticmethod
    def recommend_profile(info: HardwareInfo) -> HardwareProfile:
        """Map raw hardware detection to an Aurelius hardware profile."""
        profile = _build_profile_from_info(info)
        profile.recommended_models = _recommend_models_for_profile(profile)
        return profile


# ─── Internal helpers ────────────────────────────────────────────────────────


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def _total_ram_gb() -> float:
    import os

    pages = os.sysconf("SC_PAGE_SIZE")
    total = os.sysconf("SC_PHYS_PAGES")
    return round((pages * total) / (1024**3), 1)


def _run_cmd(cmd: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return ""


def _detect_macos(info: HardwareInfo) -> None:
    info.cpu_brand = "Apple Silicon" if info.cpu_arch == "arm64" else platform.processor()
    info.metal_available = info.cpu_arch == "arm64"
    if info.cpu_arch == "arm64":
        info.unified_memory = True
        info.unified_memory_gb = info.total_ram_gb
    # Detect Apple Silicon memory via sysctl
    mem_bytes = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if mem_bytes:
        try:
            info.total_ram_gb = round(float(mem_bytes) / (1024**3), 1)
            info.unified_memory_gb = info.total_ram_gb
        except ValueError:
            pass


def _detect_linux(info: HardwareInfo) -> None:
    info.cpu_brand = _run_cmd(["cat", "/proc/cpuinfo"]).split("model name\t: ")[-1].split("\n")[0]
    # Check for Jetson
    dt_model = _run_cmd(["cat", "/proc/device-tree/model"])
    if "jetson" in dt_model.lower() or "nvidia" in dt_model.lower():
        info.is_jetson = True
        info.jetson_model = dt_model
    # GPU via lshw or lspci
    gpu_out = _run_cmd(["lspci"])
    if "NVIDIA" in gpu_out:
        for line in gpu_out.split("\n"):
            if "NVIDIA" in line and "VGA" in line.upper():
                info.gpu_count += 1
                info.gpu_info.append({"name": line.split(":")[-1].strip()})


def _detect_windows(info: HardwareInfo) -> None:
    info.cpu_brand = _run_cmd(["wmic", "cpu", "get", "name"])
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ram_bytes = kernel32.GetPhysicallyInstalledSystemMemory()
        if ram_bytes:
            info.total_ram_gb = round(ctypes.c_ulonglong(ram_bytes).value / (1024**2) / 1024, 1)
    except Exception:
        pass


def _probe_cuda(info: HardwareInfo) -> None:
    """Detect NVIDIA CUDA GPU availability and version."""
    nvcc = _run_cmd(["nvcc", "--version"])
    if nvcc:
        info.cuda_available = True
        parts = nvcc.split("release ")[-1].split(",")
        info.cuda_version = parts[0] if parts else ""
    nvidia_smi = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if nvidia_smi:
        for line in nvidia_smi.split("\n"):
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    info.gpu_info.append({"name": parts[0], "vram_gb": float(parts[1]) / 1024})
                    info.cuda_device_count += 1
    if info.gpu_info:
        info.gpu_count = max(info.gpu_count, len(info.gpu_info))


def _probe_mlx(info: HardwareInfo) -> None:
    """Check if Apple MLX is available."""
    try:
        result = subprocess.run(
            [platform.sys.executable, "-c", "import mlx; print(mlx.__version__)"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info.mlx_available = True
    except Exception:
        pass


def _build_profile_from_info(info: HardwareInfo) -> HardwareProfile:
    """Convert raw HardwareInfo to an Aurelius HardwareProfile."""
    total_gpu_vram = sum(g.get("vram_gb", 0.0) for g in info.gpu_info)
    if info.unified_memory:
        total_gpu_vram = info.unified_memory_gb

    # Build profile id
    parts = []
    if info.is_jetson:
        parts.append("jetson")
        if "orin" in info.jetson_model.lower():
            parts.append("orin")
        else:
            parts.append("nano")
    elif info.metal_available:
        parts.append("mac_silicon")
    else:
        parts.append("linux")

    ram_round = round(info.total_ram_gb)
    ram_bucket = 4
    for b in [4, 8, 16, 24, 32, 48, 64, 96, 128, 192]:
        if ram_round <= b:
            ram_bucket = b
            break
    parts.append(f"{ram_bucket}gb")

    profile_id = "_".join(parts)

    # Detect available GPU/MLX/TensorRT
    artifact_paths = []
    for ai in info.gpu_info:
        vram = ai.get("vram_gb", 0)
        if vram >= 40:
            artifact_paths.append(QuantLevel.FP8)
        if vram >= 12:
            artifact_paths.append(QuantLevel.Q5)
        artifact_paths.append(QuantLevel.Q4)

    if info.mlx_available:
        artifact_paths.append(QuantLevel.Q4)
        artifact_paths.append(QuantLevel.BF16)

    if info.cuda_available:
        artifact_paths.append(QuantLevel.FP16)
        artifact_paths.append(QuantLevel.FP8)

    recommended: dict[str, str] = {}
    vram = total_gpu_vram if not info.unified_memory else info.total_ram_gb
    if vram >= 40:
        recommended = {"swift": "swift-safetensors", "forge": "forge-bf16", "atlas": "atlas-fp8"}
    elif vram >= 16:
        recommended = {"swift": "swift-q5-gguf", "forge": "forge-q4", "atlas": "atlas-q4-offload"}
    elif vram >= 8:
        recommended = {"swift": "swift-q4-gguf", "forge": "forge-q4-offload", "atlas": "atlas-remote"}
    else:
        recommended = {"swift": "swift-q3-gguf", "forge": "forge-remote", "atlas": "atlas-remote"}

    gpu_name = ""
    if info.gpu_info:
        gpu_name = info.gpu_info[0].get("name", "")

    return HardwareProfile(
        id=profile_id,
        label=profile_id.replace("_", " ").title(),
        cpu_arch=info.cpu_arch,
        total_ram_gb=info.total_ram_gb,
        gpu_vram_gb=total_gpu_vram,
        unified_memory=info.unified_memory,
        gpu_count=info.gpu_count,
        gpu_name=gpu_name,
        cuda_available=info.cuda_available,
        cuda_version=info.cuda_version,
        mlx_available=info.mlx_available,
        tensorrt_available=info.tensorrt_available,
        detected_artifacts=[
            ArtifactRef(
                model_name="swift",
                format="gguf",
                quant=QuantLevel.Q4,
                path="",
            )
        ],
        recommended_models=recommended,
    )


def _recommend_models_for_profile(profile: HardwareProfile) -> dict[str, str]:
    """Returns model_name → recommended_artifact for the given profile."""
    vram = profile.gpu_vram_gb
    if vram >= 40:
        return {"swift": "swift-safetensors", "forge": "forge-bf16", "atlas": "atlas-fp8"}
    if vram >= 16:
        return {"swift": "swift-q5-gguf", "forge": "forge-q4", "atlas": "atlas-q4-offload"}
    if vram >= 8:
        return {"swift": "swift-q4-gguf", "forge": "forge-q4-offload", "atlas": "atlas-remote"}
    return {"swift": "swift-q3-gguf", "forge": "forge-remote", "atlas": "atlas-remote"}
