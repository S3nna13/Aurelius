"""Aurelius v2 Profile Schema — Hardware, runtime, and user profile definitions."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class CapabilityMode(enum.StrEnum):
    """Describes which tier of capabilities is available on current profile."""

    FULL_LOCAL = "full_local"
    """All capabilities available locally."""

    REDUCED_LOCAL = "reduced_local"
    """Local but with reduced context, no VLM, limited CUA."""

    VERIFIER_ONLY = "verifier_only"
    """Local cannot execute CUA; only verifies proposed actions."""

    SPLIT = "split"
    """Local tools/CUA/memory with remote inference."""

    REMOTE = "remote"
    """All inference runs remotely; local device is a thin client."""

    CONTROLLER_ONLY = "controller_only"
    """Local device only orchestrates and routes work."""


class MemoryPolicy(enum.StrEnum):
    """Memory pressure policy applied at runtime."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    FRONTIER = "frontier"


class QuantLevel(enum.StrEnum):
    """Supported quantization levels."""

    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    Q5 = "q5"
    Q4 = "q4"
    Q3 = "q3"
    NF4 = "nf4"


@dataclass
class ArtifactRef:
    """Reference to a model artifact on disk or remote."""

    model_name: str  # "swift", "forge", "atlas"
    format: str  # "gguf", "safetensors", "mlx", "onnx", "tensorrt_llm"
    quant: QuantLevel
    path: str
    remote_url: str | None = None
    size_bytes: int = 0
    sha256: str = ""


@dataclass
class HardwareProfile:
    """Auto-detected hardware fingerprint."""

    id: str  # slug, e.g. "mac_silicon_32gb"
    label: str
    cpu_arch: str  # "arm64", "x86_64"
    total_ram_gb: float
    gpu_vram_gb: float  # 0.0 for unified, >0 for dedicated
    unified_memory: bool
    gpu_count: int = 1
    gpu_name: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    mlx_available: bool = False
    tensorrt_available: bool = False
    detected_artifacts: list[ArtifactRef] = field(default_factory=list)
    recommended_models: dict[str, str] = field(default_factory=dict)


@dataclass
class RuntimePolicy:
    """Policy that governs how the runtime behaves on this profile."""

    memory_policy: MemoryPolicy = MemoryPolicy.BALANCED
    max_context_tokens: int = 32768
    max_batch_size: int = 4
    kv_quant_level: QuantLevel = QuantLevel.NONE
    prefix_cache_enabled: bool = True
    chunked_prefill: bool = True
    mtp_enabled: bool = False
    mlp_sparsity: float = 0.0
    vision_enabled: bool = True
    skill_preload_count: int = 0
    skill_on_demand: bool = True
    cua_mode: str = "local_basic"
    safety_reserve_gb: float = 2.0


@dataclass
class UserProfile:
    """User-scoped preferences layered on top of hardware/runtime policy."""

    preferred_model: str = "forge"
    preferred_backend: str = ""
    max_tokens_per_response: int = 4096
    approval_mode: str = "inline"  # inline, batch, always, never
    language: str = "en"
    timezone: str = "UTC"
    custom: dict[str, Any] = field(default_factory=dict)
