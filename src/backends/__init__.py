"""Aurelius backend adapter surface -- isolates foreign frameworks behind native interfaces."""

from __future__ import annotations

from src.backends.base import (
    BackendAdapter,
    BackendAdapterError,
    BackendContract,
    EngineAdapter,
)
from src.backends.factory_bridge import (
    BackendBuildResult,
    build_with_backend,
)
from src.backends.registry import (
    BACKEND_REGISTRY,
    ENGINE_ADAPTER_REGISTRY,
    get_backend,
    get_engine_adapter,
    list_backends,
    list_engine_adapters,
    register_backend,
    register_engine_adapter,
    select_backend_for_manifest,
)
from src.backends.vllm_adapter import (
    VLLMAdapter,
    VLLMAdapterError,
    VLLMEngineAdapter,
    VLLMEngineConfig,
)
from src.backends.sglang_adapter import (
    SGLangAdapter,
    SGLangAdapterError,
    SGLangEngineAdapter,
    SGLangEngineConfig,
)

__all__ = [
    "BACKEND_REGISTRY",
    "ENGINE_ADAPTER_REGISTRY",
    "BackendAdapter",
    "BackendAdapterError",
    "BackendBuildResult",
    "BackendContract",
    "EngineAdapter",
    "SGLangAdapter",
    "SGLangAdapterError",
    "SGLangEngineAdapter",
    "SGLangEngineConfig",
    "VLLMAdapter",
    "VLLMAdapterError",
    "VLLMEngineAdapter",
    "VLLMEngineConfig",
    "build_with_backend",
    "get_backend",
    "get_engine_adapter",
    "list_backends",
    "list_engine_adapters",
    "register_backend",
    "register_backend_adapter_pytorch",
    "register_engine_adapter",
    "select_backend_for_manifest",
]


def register_backend_adapter_pytorch() -> None:
    """Import and register the reference PyTorch adapter.

    Kept as a module-level function so tests and external callers can
    re-invoke the registration explicitly (it is idempotent).
    """
    from src.backends import pytorch_adapter as _pt

    _pt.register()


# Register the reference PyTorch adapter at import time so that a bare
# ``import src.backends`` is sufficient to get ``"pytorch"`` in the
# registry. The registration is idempotent.
register_backend_adapter_pytorch()

# --- Ollama adapter (additive, cycle-146) --------------------------------------
from src.backends.ollama_adapter import (  # noqa: E402,F401
    OLLAMA_REGISTRY,
    OllamaAdapter,
    OllamaConfig,
)

# --- Generic HTTP backend (additive, cycle-146) --------------------------------
from src.backends.http_backend import (  # noqa: E402,F401
    HTTP_BACKEND_REGISTRY,
    HTTPBackend,
    HTTPBackendConfig,
)

# --- Backend health checker (additive, cycle-146) ------------------------------
from src.backends.backend_health import (  # noqa: E402,F401
    BACKEND_HEALTH_REGISTRY,
    BackendHealthChecker,
    HealthReport,
    HealthStatus,
)
