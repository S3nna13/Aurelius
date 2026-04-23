"""Aurelius-native GGUF / llama.cpp-style engine adapter.

This module stays stdlib-only at load time and does not import any foreign
runtime packages. Runtime availability is probed lazily with
``importlib.util.find_spec`` so the adapter can describe itself even when the
underlying engine is not installed.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from src.backends.base import BackendAdapterError, BackendContract, EngineAdapter
from src.backends.registry import ENGINE_ADAPTER_REGISTRY, register_engine_adapter

__all__ = [
    "GGUFEngineAdapter",
    "register",
]


_DEFAULT_PROBE_MODULES: tuple[str, ...] = ("llama_cpp", "llama_cpp_python")
_SUPPORTED_OPS: tuple[str, ...] = (
    "load_model",
    "generate",
    "stream_generate",
    "tokenize",
    "detokenize",
)
_QUANTIZATION_FORMATS: tuple[str, ...] = (
    "GGUF",
    "Q2_K",
    "Q3_K_S",
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
)


class GGUFEngineAdapter(EngineAdapter):
    """Contract-only adapter for GGUF-backed inference engines.

    The adapter intentionally does not attempt to load or wrap an actual
    llama.cpp binding. It only exposes a stable, JSON-safe description of
    the engine surface plus a lazy availability probe.
    """

    _CONTRACT = BackendContract(
        backend_name="gguf",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=False,
        supports_inference=True,
        capability_tags=("cpu", "gguf", "inference", "llama_cpp", "quantized"),
    )

    def __init__(
        self,
        probe_modules: tuple[str, ...] | None = None,
    ) -> None:
        self._probe_modules = (
            _DEFAULT_PROBE_MODULES if probe_modules is None else probe_modules
        )
        if not isinstance(self._probe_modules, tuple):
            raise BackendAdapterError(
                "probe_modules must be a tuple of strings, got "
                f"{type(self._probe_modules).__name__}"
            )
        if not self._probe_modules:
            raise BackendAdapterError(
                "probe_modules must not be empty"
            )
        for module_name in self._probe_modules:
            if not isinstance(module_name, str) or not module_name:
                raise BackendAdapterError(
                    "probe_modules must contain non-empty strings, got "
                    f"{module_name!r}"
                )

    @property
    def contract(self) -> BackendContract:
        return self._CONTRACT

    def _probe_results(self) -> dict[str, bool]:
        return {
            module_name: importlib.util.find_spec(module_name) is not None
            for module_name in self._probe_modules
        }

    def supported_ops(self) -> tuple[str, ...]:
        return _SUPPORTED_OPS

    def is_available(self) -> bool:
        return any(self._probe_results().values())

    def runtime_info(self) -> dict[str, Any]:
        probe_results = self._probe_results()
        contract = self.contract
        return {
            "backend_name": contract.backend_name,
            "engine_family": "llama.cpp",
            "format": "gguf",
            "available": any(probe_results.values()),
            "probe_modules": list(self._probe_modules),
            "probe_results": probe_results,
            "supported_ops": list(self.supported_ops()),
            "quantization_formats": list(_QUANTIZATION_FORMATS),
            "contract": {
                "backend_name": contract.backend_name,
                "engine_contract": contract.engine_contract,
                "adapter_contract": contract.adapter_contract,
                "supports_training": contract.supports_training,
                "supports_inference": contract.supports_inference,
                "capability_tags": list(contract.capability_tags),
            },
        }

    def describe(self) -> dict[str, Any]:
        return self.runtime_info()


def register() -> None:
    """Register a :class:`GGUFEngineAdapter` instance as ``"gguf"``."""
    if "gguf" in ENGINE_ADAPTER_REGISTRY:
        return
    register_engine_adapter(GGUFEngineAdapter())
