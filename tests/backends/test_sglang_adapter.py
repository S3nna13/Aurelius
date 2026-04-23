"""Tests for the SGLang backend adapter.

SGLang is not installed in this environment, so all tests exercise the lazy-import
guard and the fallback error path. Import of the adapter module itself must
always succeed.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Import-time sanity: module must load even without SGLang installed
# ---------------------------------------------------------------------------


def test_sglang_adapter_module_importable_without_sglang() -> None:
    """src.backends.sglang_adapter must import cleanly when SGLang is absent."""
    import src.backends.sglang_adapter  # noqa: F401


def test_sglang_adapter_class_instantiable_without_sglang() -> None:
    """SGLangAdapter() can be constructed without SGLang installed."""
    from src.backends.sglang_adapter import SGLangAdapter

    adapter = SGLangAdapter()
    assert adapter is not None


def test_sglang_engine_adapter_instantiable_without_sglang() -> None:
    """SGLangEngineAdapter() can be constructed without SGLang installed."""
    from src.backends.sglang_adapter import SGLangEngineAdapter

    engine = SGLangEngineAdapter()
    assert engine is not None


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_sglang_adapter_error_is_subclass_of_backend_adapter_error() -> None:
    from src.backends.base import BackendAdapterError
    from src.backends.sglang_adapter import SGLangAdapterError

    assert issubclass(SGLangAdapterError, BackendAdapterError)


def test_sglang_adapter_error_is_exception() -> None:
    from src.backends.sglang_adapter import SGLangAdapterError

    assert issubclass(SGLangAdapterError, Exception)


# ---------------------------------------------------------------------------
# Lazy import guard: _load_engine raises SGLangAdapterError when SGLang absent
# ---------------------------------------------------------------------------


def test_load_engine_raises_sglang_adapter_error_when_sglang_not_installed() -> None:
    """_load_engine() must raise SGLangAdapterError (not ImportError) when SGLang absent."""
    from src.backends.sglang_adapter import SGLangAdapter, SGLangAdapterError, SGLangEngineConfig

    config = SGLangEngineConfig(model_path="/fake/model")
    adapter = SGLangAdapter(config=config)

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
        (_ for _ in ()).throw(ImportError("No module named 'sglang'"))
        if name == "sglang"
        else real_import(name, *a, **kw)
    )):
        with pytest.raises(SGLangAdapterError, match="SGLang is not installed"):
            adapter._load_engine()


def test_load_engine_raises_when_sglang_missing_no_config() -> None:
    """_load_engine() raises SGLangAdapterError mentioning 'SGLang is not installed'."""
    from src.backends.sglang_adapter import SGLangAdapter, SGLangAdapterError

    adapter = SGLangAdapter()  # no config
    with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
        (_ for _ in ()).throw(ImportError("No module named 'sglang'"))
        if name == "sglang"
        else (__import__(name, *a, **kw))
    )):
        with pytest.raises(SGLangAdapterError, match="SGLang is not installed"):
            adapter._load_engine()


# ---------------------------------------------------------------------------
# SGLangEngineConfig dataclass defaults
# ---------------------------------------------------------------------------


def test_sglang_engine_config_defaults() -> None:
    from src.backends.sglang_adapter import SGLangEngineConfig

    cfg = SGLangEngineConfig(model_path="/tmp/model")
    assert cfg.model_path == "/tmp/model"
    assert cfg.port == 30000
    assert cfg.mem_fraction_static == pytest.approx(0.85)
    assert cfg.tp_size == 1


def test_sglang_engine_config_custom_values() -> None:
    from src.backends.sglang_adapter import SGLangEngineConfig

    cfg = SGLangEngineConfig(
        model_path="/models/qwen",
        port=30001,
        mem_fraction_static=0.75,
        tp_size=2,
    )
    assert cfg.port == 30001
    assert cfg.mem_fraction_static == pytest.approx(0.75)
    assert cfg.tp_size == 2


# ---------------------------------------------------------------------------
# Contract / adapter identity
# ---------------------------------------------------------------------------


def test_sglang_adapter_contract_identity() -> None:
    from src.backends.sglang_adapter import SGLangAdapter

    adapter = SGLangAdapter()
    assert adapter.contract.backend_name == "sglang"
    assert adapter.contract.supports_training is False
    assert adapter.contract.supports_inference is True


def test_sglang_engine_adapter_contract_identity() -> None:
    from src.backends.sglang_adapter import SGLangEngineAdapter

    engine = SGLangEngineAdapter()
    assert engine.contract.backend_name == "sglang"
    assert engine.contract.supports_inference is True


def test_sglang_adapter_is_available_false_without_sglang() -> None:
    from src.backends.sglang_adapter import SGLangAdapter

    adapter = SGLangAdapter()
    assert adapter.is_available() is False


def test_sglang_engine_adapter_is_available_false_without_sglang() -> None:
    from src.backends.sglang_adapter import SGLangEngineAdapter

    engine = SGLangEngineAdapter()
    assert engine.is_available() is False


def test_sglang_engine_adapter_supported_ops() -> None:
    from src.backends.sglang_adapter import SGLangEngineAdapter

    engine = SGLangEngineAdapter()
    assert "generate" in engine.supported_ops()


def test_sglang_engine_adapter_describe_returns_dict() -> None:
    from src.backends.sglang_adapter import SGLangEngineAdapter

    engine = SGLangEngineAdapter()
    desc = engine.describe()
    assert isinstance(desc, dict)
    assert desc["engine"] == "sglang"
    assert desc["available"] is False


# ---------------------------------------------------------------------------
# Registry membership after explicit register()
# ---------------------------------------------------------------------------


def test_backend_registry_contains_sglang_after_register() -> None:
    from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY
    from src.backends.sglang_adapter import register, SGLangAdapter

    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    try:
        BACKEND_REGISTRY.pop("sglang", None)
        ENGINE_ADAPTER_REGISTRY.pop("sglang", None)

        register()

        assert "sglang" in BACKEND_REGISTRY
        assert isinstance(BACKEND_REGISTRY["sglang"], SGLangAdapter)
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(before_backend)
        ENGINE_ADAPTER_REGISTRY.clear()
        ENGINE_ADAPTER_REGISTRY.update(before_engine)


def test_engine_adapter_registry_contains_sglang_after_register() -> None:
    from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY
    from src.backends.sglang_adapter import register, SGLangEngineAdapter

    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    try:
        BACKEND_REGISTRY.pop("sglang", None)
        ENGINE_ADAPTER_REGISTRY.pop("sglang", None)

        register()

        assert "sglang" in ENGINE_ADAPTER_REGISTRY
        assert isinstance(ENGINE_ADAPTER_REGISTRY["sglang"], SGLangEngineAdapter)
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(before_backend)
        ENGINE_ADAPTER_REGISTRY.clear()
        ENGINE_ADAPTER_REGISTRY.update(before_engine)


def test_register_is_idempotent() -> None:
    from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY
    from src.backends.sglang_adapter import register

    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    try:
        BACKEND_REGISTRY.pop("sglang", None)
        ENGINE_ADAPTER_REGISTRY.pop("sglang", None)

        register()
        first_adapter = BACKEND_REGISTRY["sglang"]
        register()  # second call must not raise
        assert BACKEND_REGISTRY["sglang"] is first_adapter
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(before_backend)
        ENGINE_ADAPTER_REGISTRY.clear()
        ENGINE_ADAPTER_REGISTRY.update(before_engine)
