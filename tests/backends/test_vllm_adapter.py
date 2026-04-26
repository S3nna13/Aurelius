"""Tests for the vLLM backend adapter.

vLLM is not installed in this environment, so all tests exercise the lazy-import
guard and the fallback error path. Import of the adapter module itself must
always succeed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import-time sanity: module must load even without vLLM installed
# ---------------------------------------------------------------------------


def test_vllm_adapter_module_importable_without_vllm() -> None:
    """src.backends.vllm_adapter must import cleanly when vLLM is absent."""
    import src.backends.vllm_adapter  # noqa: F401 – import is the assertion


def test_vllm_adapter_class_instantiable_without_vllm() -> None:
    """VLLMAdapter() can be constructed without vLLM installed."""
    from src.backends.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter()
    assert adapter is not None


def test_vllm_engine_adapter_instantiable_without_vllm() -> None:
    """VLLMEngineAdapter() can be constructed without vLLM installed."""
    from src.backends.vllm_adapter import VLLMEngineAdapter

    engine = VLLMEngineAdapter()
    assert engine is not None


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_vllm_adapter_error_is_subclass_of_backend_adapter_error() -> None:
    from src.backends.base import BackendAdapterError
    from src.backends.vllm_adapter import VLLMAdapterError

    assert issubclass(VLLMAdapterError, BackendAdapterError)


def test_vllm_adapter_error_is_exception() -> None:
    from src.backends.vllm_adapter import VLLMAdapterError

    assert issubclass(VLLMAdapterError, Exception)


# ---------------------------------------------------------------------------
# Lazy import guard: _load_engine raises VLLMAdapterError when vLLM absent
# ---------------------------------------------------------------------------


def test_load_engine_raises_vllm_adapter_error_when_vllm_not_installed() -> None:
    """_load_engine() must raise VLLMAdapterError (not ImportError) when vLLM absent."""
    from src.backends.vllm_adapter import VLLMAdapter, VLLMAdapterError, VLLMEngineConfig

    config = VLLMEngineConfig(model_path="/fake/model")
    adapter = VLLMAdapter(config=config)

    # Patch importlib so that 'import vllm' raises ImportError.
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    with patch(
        "builtins.__import__",
        side_effect=lambda name, *a, **kw: (
            (_ for _ in ()).throw(ImportError("No module named 'vllm'"))
            if name == "vllm"
            else real_import(name, *a, **kw)
        ),
    ):
        with pytest.raises(VLLMAdapterError, match="vLLM is not installed"):
            adapter._load_engine()


def test_load_engine_raises_when_vllm_missing_no_config() -> None:
    """_load_engine() raises VLLMAdapterError mentioning 'vLLM is not installed'."""
    from src.backends.vllm_adapter import VLLMAdapter, VLLMAdapterError

    adapter = VLLMAdapter()  # no config
    with patch(
        "builtins.__import__",
        side_effect=lambda name, *a, **kw: (
            (_ for _ in ()).throw(ImportError("No module named 'vllm'"))
            if name == "vllm"
            else (__import__(name, *a, **kw))
        ),
    ):
        with pytest.raises(VLLMAdapterError, match="vLLM is not installed"):
            adapter._load_engine()


# ---------------------------------------------------------------------------
# VLLMEngineConfig dataclass defaults
# ---------------------------------------------------------------------------


def test_vllm_engine_config_defaults() -> None:
    from src.backends.vllm_adapter import VLLMEngineConfig

    cfg = VLLMEngineConfig(model_path="/tmp/model")
    assert cfg.model_path == "/tmp/model"
    assert cfg.tensor_parallel_size == 1
    assert cfg.max_num_seqs == 256
    assert cfg.dtype == "auto"
    assert cfg.quantization is None
    assert cfg.gpu_memory_utilization == pytest.approx(0.90)


def test_vllm_engine_config_custom_values() -> None:
    from src.backends.vllm_adapter import VLLMEngineConfig

    cfg = VLLMEngineConfig(
        model_path="/models/llama",
        tensor_parallel_size=4,
        max_num_seqs=512,
        dtype="float16",
        quantization="awq",
        gpu_memory_utilization=0.80,
    )
    assert cfg.tensor_parallel_size == 4
    assert cfg.dtype == "float16"
    assert cfg.quantization == "awq"


# ---------------------------------------------------------------------------
# Contract / adapter identity
# ---------------------------------------------------------------------------


def test_vllm_adapter_contract_identity() -> None:
    from src.backends.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter()
    assert adapter.contract.backend_name == "vllm"
    assert adapter.contract.supports_training is False
    assert adapter.contract.supports_inference is True


def test_vllm_engine_adapter_contract_identity() -> None:
    from src.backends.vllm_adapter import VLLMEngineAdapter

    engine = VLLMEngineAdapter()
    assert engine.contract.backend_name == "vllm"
    assert engine.contract.supports_inference is True


def test_vllm_adapter_is_available_false_without_vllm() -> None:
    from src.backends.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter()
    # vLLM is not installed in this environment
    assert adapter.is_available() is False


def test_vllm_engine_adapter_is_available_false_without_vllm() -> None:
    from src.backends.vllm_adapter import VLLMEngineAdapter

    engine = VLLMEngineAdapter()
    assert engine.is_available() is False


def test_vllm_engine_adapter_supported_ops() -> None:
    from src.backends.vllm_adapter import VLLMEngineAdapter

    engine = VLLMEngineAdapter()
    assert "generate" in engine.supported_ops()


def test_vllm_engine_adapter_describe_returns_dict() -> None:
    from src.backends.vllm_adapter import VLLMEngineAdapter

    engine = VLLMEngineAdapter()
    desc = engine.describe()
    assert isinstance(desc, dict)
    assert desc["engine"] == "vllm"
    assert desc["available"] is False


# ---------------------------------------------------------------------------
# Registry membership after explicit register()
# ---------------------------------------------------------------------------


def test_backend_registry_contains_vllm_after_register() -> None:
    from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY
    from src.backends.vllm_adapter import VLLMAdapter, register

    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    try:
        # Remove any existing entry so register() actually inserts.
        BACKEND_REGISTRY.pop("vllm", None)
        ENGINE_ADAPTER_REGISTRY.pop("vllm", None)

        register()

        assert "vllm" in BACKEND_REGISTRY
        assert isinstance(BACKEND_REGISTRY["vllm"], VLLMAdapter)
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(before_backend)
        ENGINE_ADAPTER_REGISTRY.clear()
        ENGINE_ADAPTER_REGISTRY.update(before_engine)


def test_engine_adapter_registry_contains_vllm_after_register() -> None:
    from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY
    from src.backends.vllm_adapter import VLLMEngineAdapter, register

    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    try:
        BACKEND_REGISTRY.pop("vllm", None)
        ENGINE_ADAPTER_REGISTRY.pop("vllm", None)

        register()

        assert "vllm" in ENGINE_ADAPTER_REGISTRY
        assert isinstance(ENGINE_ADAPTER_REGISTRY["vllm"], VLLMEngineAdapter)
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(before_backend)
        ENGINE_ADAPTER_REGISTRY.clear()
        ENGINE_ADAPTER_REGISTRY.update(before_engine)


def test_register_is_idempotent() -> None:
    from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY
    from src.backends.vllm_adapter import register

    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    try:
        BACKEND_REGISTRY.pop("vllm", None)
        ENGINE_ADAPTER_REGISTRY.pop("vllm", None)

        register()
        first_adapter = BACKEND_REGISTRY["vllm"]
        register()  # second call must not raise
        assert BACKEND_REGISTRY["vllm"] is first_adapter
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(before_backend)
        ENGINE_ADAPTER_REGISTRY.clear()
        ENGINE_ADAPTER_REGISTRY.update(before_engine)
