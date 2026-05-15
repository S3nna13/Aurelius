"""Aurelius v2 Backend Selector — chooses the best backend based on hardware, model, artifacts, and policy."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from src.runtime.capability_report import CapabilityMode, CapabilityReport
from src.runtime.profile_schema import HardwareProfile, MemoryPolicy


class BackendType(enum.StrEnum):
    PYTORCH_EAGER = "pytorch_eager"
    TORCH_COMPILE = "torch_compile"
    LLAMA_CPP_GGUF = "llama_cpp_gguf"
    MLX = "mlx"
    ONNX_RUNTIME = "onnx_runtime"
    TENSORRT_EDGE_LLM = "tensorrt_edge_llm"
    TENSORRT_LLM = "tensorrt_llm"
    VLLM = "vllm"
    REMOTE_AURELIUS = "remote_aurelius"


@dataclass
class BackendSelection:
    """Result of backend selection for a given model + profile."""

    backend: BackendType
    quantization: str
    context_budget: int
    capability_mode: CapabilityMode
    skill_preload_count: int
    reasons: list[str] = field(default_factory=list)
    unavailable_backends: list[str] = field(default_factory=list)

    def to_capability_report(
        self,
        model: str,
        actual_model: str,
        hardware_profile: HardwareProfile,
        artifact: str = "",
    ) -> CapabilityReport:
        """Convert selection to a capability report for CLI/UI consumption."""
        exec_mode_map = {
            BackendType.PYTORCH_EAGER: CapabilityMode.FULL_LOCAL,
            BackendType.TORCH_COMPILE: CapabilityMode.FULL_LOCAL,
            BackendType.LLAMA_CPP_GGUF: CapabilityMode.FULL_LOCAL,
            BackendType.MLX: CapabilityMode.FULL_LOCAL,
            BackendType.ONNX_RUNTIME: CapabilityMode.FULL_LOCAL,
            BackendType.TENSORRT_EDGE_LLM: CapabilityMode.REDUCED_LOCAL,
            BackendType.TENSORRT_LLM: CapabilityMode.FULL_LOCAL,
            BackendType.VLLM: CapabilityMode.FULL_LOCAL,
            BackendType.REMOTE_AURELIUS: CapabilityMode.REMOTE,
        }
        exec_mode_map.get(self.backend, CapabilityMode.FULL_LOCAL)
        if self.capability_mode == CapabilityMode.REMOTE:
            pass
        elif self.capability_mode == CapabilityMode.VERIFIER_ONLY:
            pass
        elif self.capability_mode == CapabilityMode.SPLIT:
            pass


        return CapabilityReport.create_full_local(
            model=model,
            backend=self.backend.value,
            artifact=artifact,
            quantization=self.quantization,
            context=self.context_budget,
            hardware=hardware_profile.id,
        )


class BackendSelector:
    """Selects the optimal backend for a given model and hardware profile.

    Selection flow:
    1. Detect hardware (already done before calling this)
    2. Detect RAM/VRAM/unified memory
    3. Inspect requested model
    4. Inspect available local artifacts
    5. Inspect remote endpoints
    6. Choose backend
    7. Choose quantization
    8. Choose context budget
    9. Choose capability mode
    10. Choose skill preload policy
    11. Produce capability report
    """

    def select(
        self,
        requested_model: str,
        hardware: HardwareProfile,
        available_artifacts: list[str] | None = None,
        remote_endpoint: str | None = None,
        memory_policy: MemoryPolicy = MemoryPolicy.BALANCED,
    ) -> BackendSelection:
        """Select the best backend for the given model + hardware."""
        reasons: list[str] = []
        unavailable: list[str] = []
        artifacts = available_artifacts or []

        vram = hardware.gpu_vram_gb
        total_mem = hardware.total_ram_gb
        max(vram, total_mem) if hardware.unified_memory else max(vram, 0)

        # Check available backends
        has_cuda = hardware.cuda_available
        has_mlx = hardware.mlx_available
        has_tensorrt = hardware.tensorrt_available
        has_remote = remote_endpoint is not None and len(remote_endpoint) > 0

        # Swift selection (~0.6B)
        if requested_model.lower() == "swift":
            return self._select_swift(
                hardware, has_cuda, has_mlx, has_tensorrt, has_remote, artifacts, reasons, unavailable,
            )

        # Forge selection (~3B)
        if requested_model.lower() == "forge":
            return self._select_forge(
                hardware, has_cuda, has_mlx, has_tensorrt, has_remote, artifacts, reasons, unavailable, memory_policy,
            )

        # Atlas selection (~32B MoE or 14B dense)
        if requested_model.lower() == "atlas":
            return self._select_atlas(
                hardware, has_cuda, has_mlx, has_tensorrt, has_remote, artifacts, reasons, unavailable, memory_policy,
            )

        # Unknown model: default to remote or most capable available
        if has_remote:
            return BackendSelection(
                backend=BackendType.REMOTE_AURELIUS,
                quantization="fp8",
                context_budget=32768,
                capability_mode=CapabilityMode.REMOTE,
                skill_preload_count=0,
                reasons=[f"Unknown model={requested_model}, defaulting to remote"],
            )

        return BackendSelection(
            backend=BackendType.PYTORCH_EAGER,
            quantization="bf16",
            context_budget=8192,
            capability_mode=CapabilityMode.REDUCED_LOCAL,
            skill_preload_count=0,
            reasons=[f"Unknown model={requested_model}, defaulting to pytorch_eager"],
            unavailable_backends=unavailable,
        )

    def _select_swift(
        self, hw: HardwareProfile, has_cuda: bool, has_mlx: bool, has_trt: bool,
        has_remote: bool, artifacts: list[str], reasons: list[str], unavailable: list[str],
    ) -> BackendSelection:
        """Select backend for Swift (~0.6B, runs anywhere)."""
        # Swift is tiny — prefer the fastest local backend
        if has_mlx and any("swift" in a and "mlx" in a for a in artifacts):
            return BackendSelection(BackendType.MLX, "q4", 32768, CapabilityMode.FULL_LOCAL, 5,
                                    reasons=["Swift: MLX q4 native on Apple Silicon"])
        if has_cuda and any("swift" in a and "tensorrt" in a for a in artifacts):
            return BackendSelection(BackendType.TENSORRT_LLM, "fp16", 32768, CapabilityMode.FULL_LOCAL, 5,
                                    reasons=["Swift: TensorRT-LLM FP16 on CUDA"])
        if any("swift" in a and "gguf" in a for a in artifacts):
            return BackendSelection(BackendType.LLAMA_CPP_GGUF, "q4", 32768, CapabilityMode.FULL_LOCAL, 5,
                                    reasons=["Swift: llama.cpp GGUF q4"])

        # Fallbacks
        if has_mlx:
            return BackendSelection(BackendType.MLX, "q4", 16384, CapabilityMode.FULL_LOCAL, 5,
                                    reasons=["Swift: MLX (downloading artifact)"])
        if has_cuda:
            return BackendSelection(BackendType.PYTORCH_EAGER, "bf16", 16384, CapabilityMode.FULL_LOCAL, 5,
                                    reasons=["Swift: PyTorch eager bf16 on CUDA"])
        return BackendSelection(BackendType.ONNX_RUNTIME, "int8", 8192, CapabilityMode.FULL_LOCAL, 0,
                                reasons=["Swift: ONNX Runtime INT8"])

    def _select_forge(
        self, hw: HardwareProfile, has_cuda: bool, has_mlx: bool, has_trt: bool,
        has_remote: bool, artifacts: list[str], reasons: list[str], unavailable: list[str],
        memory_policy: MemoryPolicy = MemoryPolicy.BALANCED,
    ) -> BackendSelection:
        """Select backend for Forge (~3B)."""
        vram = hw.gpu_vram_gb
        total_mem = hw.total_ram_gb

        if hw.unified_memory:
            effective = total_mem
        else:
            effective = vram

        # Best: TensorRT-LLM on CUDA with sufficient VRAM
        if has_cuda and vram >= 8 and any("forge" in a and "tensorrt" in a for a in artifacts):
            ctx = 65536 if memory_policy in (MemoryPolicy.PERFORMANCE, MemoryPolicy.FRONTIER) else 32768
            return BackendSelection(BackendType.TENSORRT_LLM, "fp8" if vram >= 12 else "q5", ctx,
                                    CapabilityMode.FULL_LOCAL, 10,
                                    reasons=["Forge: TensorRT-LLM on CUDA"])

        # MLX on Apple Silicon
        if has_mlx and effective >= 16:
            ctx = 32768 if memory_policy == MemoryPolicy.BALANCED else 65536
            return BackendSelection(BackendType.MLX, "q4", ctx,
                                    CapabilityMode.FULL_LOCAL, 10,
                                    reasons=["Forge: MLX q4 on Apple Silicon"])

        # llama.cpp GGUF as universal fallback
        if any("forge" in a and "gguf" in a for a in artifacts):
            return BackendSelection(BackendType.LLAMA_CPP_GGUF, "q4", 16384,
                                    CapabilityMode.FULL_LOCAL, 5,
                                    reasons=["Forge: llama.cpp GGUF q4"])

        # If not enough local resources, try remote
        if has_remote:
            return BackendSelection(BackendType.REMOTE_AURELIUS, "fp8", 32768,
                                    CapabilityMode.SPLIT, 0,
                                    reasons=["Forge insufficient local memory; using remote inference"])

        # Last resort: pytorch eager with heavy offload
        if has_cuda:
            return BackendSelection(BackendType.PYTORCH_EAGER, "q5", 8192,
                                    CapabilityMode.REDUCED_LOCAL, 0,
                                    reasons=["Forge: PyTorch eager with heavy offload"])

        return BackendSelection(BackendType.REMOTE_AURELIUS, "fp8", 16384,
                                CapabilityMode.REMOTE, 0,
                                reasons=["Forge: no viable local backend; remote only"])

    def _select_atlas(
        self, hw: HardwareProfile, has_cuda: bool, has_mlx: bool, has_trt: bool,
        has_remote: bool, artifacts: list[str], reasons: list[str], unavailable: list[str],
        memory_policy: MemoryPolicy = MemoryPolicy.BALANCED,
    ) -> BackendSelection:
        """Select backend for Atlas (~32B MoE or 14B dense fallback)."""
        vram = hw.gpu_vram_gb
        total_mem = hw.total_ram_gb
        effective = total_mem if hw.unified_memory else vram

        # Best: TensorRT-LLM FP8/FP4 on large GPU
        if has_cuda and vram >= 40:
            ctx = 131072 if memory_policy == MemoryPolicy.FRONTIER else 65536
            quant = "fp4" if vram >= 80 else "fp8"
            return BackendSelection(BackendType.TENSORRT_LLM, quant, ctx,
                                    CapabilityMode.FULL_LOCAL, 20,
                                    reasons=[f"Atlas: TensorRT-LLM {quant} on {int(vram)}GB VRAM"])

        # Large unified memory (Mac Ultra)
        if has_mlx and effective >= 96:
            return BackendSelection(BackendType.MLX, "q4", 65536,
                                    CapabilityMode.FULL_LOCAL, 15,
                                    reasons=["Atlas: MLX on Mac Ultra unified memory"])

        # Offload Atlas to CPU/RAM with CUDA assist
        if has_cuda and vram >= 16:
            return BackendSelection(BackendType.PYTORCH_EAGER, "q5", 16384,
                                    CapabilityMode.REDUCED_LOCAL, 5,
                                    reasons=["Atlas: PyTorch with CUDA assist + CPU offload"])

        # Remote is most likely for Atlas on most hardware
        if has_remote:
            return BackendSelection(BackendType.REMOTE_AURELIUS, "fp8", 131072,
                                    CapabilityMode.SPLIT, 10,
                                    reasons=["Atlas: remote inference with local tools/CUA"])

        # Nothing available
        return BackendSelection(BackendType.REMOTE_AURELIUS, "fp8", 32768,
                                CapabilityMode.REMOTE, 0,
                                reasons=["Atlas: no local backend available; remote only"])
