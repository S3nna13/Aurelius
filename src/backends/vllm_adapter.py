"""vLLM inference engine adapter for Aurelius.

Inspired by vllm-project/vllm (Apache-2.0) and sgl-project/sglang (Apache-2.0),
clean-room reimplementation of the adapter interface.

This module is an explicit Policy-B exception: vLLM may be imported here, but
only inside methods or try/except blocks so that the module remains importable
when vLLM is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.backends.base import (
    BackendAdapter,
    BackendAdapterError,
    BackendContract,
    EngineAdapter,
)

__all__ = [
    "VLLMAdapterError",
    "VLLMEngineConfig",
    "VLLMAdapter",
    "VLLMEngineAdapter",
    "register",
]


class VLLMAdapterError(BackendAdapterError):
    """Raised for any vLLM adapter contract or runtime failure."""


@dataclass
class VLLMEngineConfig:
    """Configuration for a vLLM-backed inference engine.

    All fields map directly to ``vllm.LLM`` / ``AsyncLLMEngine`` kwargs.
    """

    model_path: str
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    dtype: str = "auto"
    quantization: str | None = None
    gpu_memory_utilization: float = 0.90


class VLLMAdapter(BackendAdapter):
    """Inference-only :class:`BackendAdapter` backed by vLLM.

    The vLLM engine is initialised lazily on first use so that this class
    can be instantiated — and the module imported — without vLLM installed.
    """

    _CONTRACT = BackendContract(
        backend_name="vllm",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=False,
        supports_inference=True,
        capability_tags=("cuda", "inference", "continuous-batching"),
    )

    def __init__(self, config: VLLMEngineConfig | None = None) -> None:
        self._config = config
        self._engine: Any = None

    @property
    def contract(self) -> BackendContract:
        return self._CONTRACT

    def _load_engine(self) -> Any:
        """Lazily initialise and return the vLLM LLM instance.

        Raises :class:`VLLMAdapterError` if vLLM is not installed or if no
        config has been provided.
        """
        if self._engine is not None:
            return self._engine

        try:
            import vllm  # noqa: F401 – availability check
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as exc:
            raise VLLMAdapterError("vLLM is not installed; pip install vllm") from exc

        if self._config is None:
            raise VLLMAdapterError(
                "VLLMAdapter requires a VLLMEngineConfig to load the engine; "
                "pass config= at construction time"
            )

        from vllm import LLM

        self._engine = LLM(
            model=self._config.model_path,
            tensor_parallel_size=self._config.tensor_parallel_size,
            max_num_seqs=self._config.max_num_seqs,
            dtype=self._config.dtype,
            quantization=self._config.quantization,
            gpu_memory_utilization=self._config.gpu_memory_utilization,
        )
        return self._engine

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> list[int]:
        """Run a single greedy/sampled generation via vLLM.

        Parameters
        ----------
        input_ids:
            Token IDs for the prompt.
        max_new_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature (0 = greedy).

        Returns
        -------
        list[int]
            Generated token IDs (not including the prompt).
        """
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise VLLMAdapterError("vLLM is not installed; pip install vllm") from exc

        engine = self._load_engine()
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        # vLLM accepts prompt token IDs directly via prompt_token_ids.
        outputs = engine.generate(
            prompts=None,
            sampling_params=sampling_params,
            prompt_token_ids=[input_ids],
        )
        return list(outputs[0].outputs[0].token_ids)

    # ------------------------------------------------------------------
    # BackendAdapter abstract interface
    # ------------------------------------------------------------------

    def tensor_ns(self) -> object:
        """vLLM is inference-only; tensor namespace is not exposed."""
        raise VLLMAdapterError(
            "VLLMAdapter is inference-only and does not expose a tensor namespace"
        )

    def dtype_of(self, obj: Any) -> str:
        raise VLLMAdapterError("VLLMAdapter is inference-only; dtype_of is not supported")

    def device_of(self, obj: Any) -> str:
        raise VLLMAdapterError("VLLMAdapter is inference-only; device_of is not supported")

    def is_available(self) -> bool:
        """Return True iff vLLM can be imported in the current environment."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def runtime_info(self) -> dict:
        info = super().runtime_info()
        if self.is_available():
            import vllm

            info["vllm_version"] = vllm.__version__
        return info


class VLLMEngineAdapter(EngineAdapter):
    """Inference :class:`EngineAdapter` backed by vLLM.

    Provides engine-level discovery / introspection without requiring vLLM
    to be installed at import time.
    """

    _CONTRACT = BackendContract(
        backend_name="vllm",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=False,
        supports_inference=True,
        capability_tags=("cuda", "inference", "continuous-batching"),
    )

    @property
    def contract(self) -> BackendContract:
        return self._CONTRACT

    def is_available(self) -> bool:
        """Return True iff vLLM is importable."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def describe(self) -> dict:
        desc: dict[str, Any] = {
            "engine": "vllm",
            "available": self.is_available(),
            "supports_training": False,
            "supports_inference": True,
        }
        if self.is_available():
            import vllm

            desc["vllm_version"] = vllm.__version__
        return desc

    def supported_ops(self) -> tuple[str, ...]:
        return ("generate",)


def register() -> None:
    """Register :class:`VLLMAdapter` and :class:`VLLMEngineAdapter` instances.

    Idempotent: calling this twice is a no-op once both registries already
    contain entries under ``"vllm"``.
    """
    from src.backends.registry import (
        BACKEND_REGISTRY,
        ENGINE_ADAPTER_REGISTRY,
        register_backend,
        register_engine_adapter,
    )

    if "vllm" not in BACKEND_REGISTRY:
        register_backend(VLLMAdapter())
    if "vllm" not in ENGINE_ADAPTER_REGISTRY:
        register_engine_adapter(VLLMEngineAdapter())
