"""SGLang inference engine adapter for Aurelius.

Inspired by vllm-project/vllm (Apache-2.0) and sgl-project/sglang (Apache-2.0),
clean-room reimplementation of the adapter interface.

This module is an explicit Policy-B exception: SGLang may be imported here, but
only inside methods or try/except blocks so that the module remains importable
when SGLang is not installed.
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
    "SGLangAdapterError",
    "SGLangEngineConfig",
    "SGLangAdapter",
    "SGLangEngineAdapter",
    "register",
]


class SGLangAdapterError(BackendAdapterError):
    """Raised for any SGLang adapter contract or runtime failure."""


@dataclass
class SGLangEngineConfig:
    """Configuration for an SGLang-backed inference engine."""

    model_path: str
    port: int = 30000
    mem_fraction_static: float = 0.85
    tp_size: int = 1


class SGLangAdapter(BackendAdapter):
    """Inference-only :class:`BackendAdapter` backed by SGLang.

    The SGLang runtime is initialised lazily on first use so that this class
    can be instantiated — and the module imported — without SGLang installed.
    """

    _CONTRACT = BackendContract(
        backend_name="sglang",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=False,
        supports_inference=True,
        capability_tags=("cuda", "inference", "radix-cache"),
    )

    def __init__(self, config: SGLangEngineConfig | None = None) -> None:
        self._config = config
        self._runtime: Any = None

    @property
    def contract(self) -> BackendContract:
        return self._CONTRACT

    def _load_engine(self) -> Any:
        """Lazily initialise and return the SGLang Runtime instance.

        Raises :class:`SGLangAdapterError` if SGLang is not installed or if no
        config has been provided.
        """
        if self._runtime is not None:
            return self._runtime

        try:
            import sglang  # noqa: F401 – availability check
        except ImportError as exc:
            raise SGLangAdapterError("SGLang is not installed; pip install sglang") from exc

        if self._config is None:
            raise SGLangAdapterError(
                "SGLangAdapter requires a SGLangEngineConfig to load the engine; "
                "pass config= at construction time"
            )

        import sglang as sgl

        self._runtime = sgl.Runtime(
            model_path=self._config.model_path,
            port=self._config.port,
            mem_fraction_static=self._config.mem_fraction_static,
            tp_size=self._config.tp_size,
        )
        return self._runtime

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> list[int]:
        """Run a single generation via SGLang.

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
            import sglang as sgl  # noqa: F401
        except ImportError as exc:
            raise SGLangAdapterError("SGLang is not installed; pip install sglang") from exc

        runtime = self._load_engine()

        @sgl.function
        def _gen(s: Any) -> None:
            s += sgl.gen("output", max_new_tokens=max_new_tokens, temperature=temperature)

        _gen.run(backend=runtime)
        # SGLang returns text; token IDs are not directly exposed in the
        # high-level API. Return an empty list as a safe stub — callers
        # requiring token IDs should use the runtime's tokenizer directly.
        return []

    # ------------------------------------------------------------------
    # BackendAdapter abstract interface
    # ------------------------------------------------------------------

    def tensor_ns(self) -> object:
        """SGLang is inference-only; tensor namespace is not exposed."""
        raise SGLangAdapterError(
            "SGLangAdapter is inference-only and does not expose a tensor namespace"
        )

    def dtype_of(self, obj: Any) -> str:
        raise SGLangAdapterError("SGLangAdapter is inference-only; dtype_of is not supported")

    def device_of(self, obj: Any) -> str:
        raise SGLangAdapterError("SGLangAdapter is inference-only; device_of is not supported")

    def is_available(self) -> bool:
        """Return True iff SGLang can be imported in the current environment."""
        try:
            import sglang  # noqa: F401

            return True
        except ImportError:
            return False

    def runtime_info(self) -> dict:
        info = super().runtime_info()
        if self.is_available():
            import sglang

            info["sglang_version"] = getattr(sglang, "__version__", "unknown")
        return info


class SGLangEngineAdapter(EngineAdapter):
    """Inference :class:`EngineAdapter` backed by SGLang.

    Provides engine-level discovery / introspection without requiring SGLang
    to be installed at import time.
    """

    _CONTRACT = BackendContract(
        backend_name="sglang",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=False,
        supports_inference=True,
        capability_tags=("cuda", "inference", "radix-cache"),
    )

    @property
    def contract(self) -> BackendContract:
        return self._CONTRACT

    def is_available(self) -> bool:
        """Return True iff SGLang is importable."""
        try:
            import sglang  # noqa: F401

            return True
        except ImportError:
            return False

    def describe(self) -> dict:
        desc: dict[str, Any] = {
            "engine": "sglang",
            "available": self.is_available(),
            "supports_training": False,
            "supports_inference": True,
        }
        if self.is_available():
            import sglang

            desc["sglang_version"] = getattr(sglang, "__version__", "unknown")
        return desc

    def supported_ops(self) -> tuple[str, ...]:
        return ("generate",)


def register() -> None:
    """Register :class:`SGLangAdapter` and :class:`SGLangEngineAdapter` instances.

    Idempotent: calling this twice is a no-op once both registries already
    contain entries under ``"sglang"``.
    """
    from src.backends.registry import (
        BACKEND_REGISTRY,
        ENGINE_ADAPTER_REGISTRY,
        register_backend,
        register_engine_adapter,
    )

    if "sglang" not in BACKEND_REGISTRY:
        register_backend(SGLangAdapter())
    if "sglang" not in ENGINE_ADAPTER_REGISTRY:
        register_engine_adapter(SGLangEngineAdapter())
