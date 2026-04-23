"""Integration tests for the GGUF / llama.cpp-style engine adapter."""

from __future__ import annotations

import json

import pytest

from src.backends.base import BackendAdapterError
from src.backends.gguf_engine_adapter import GGUFEngineAdapter, register
from src.backends.registry import (
    ENGINE_ADAPTER_REGISTRY,
    get_engine_adapter,
    list_engine_adapters,
)


@pytest.fixture(autouse=True)
def _snapshot_registry() -> None:
    before = dict(ENGINE_ADAPTER_REGISTRY)
    yield
    ENGINE_ADAPTER_REGISTRY.clear()
    ENGINE_ADAPTER_REGISTRY.update(before)


def test_register_and_lookup_gguf_engine_adapter() -> None:
    register()
    names = list_engine_adapters()
    assert "gguf" in names
    adapter = get_engine_adapter("gguf")
    assert adapter.contract.backend_name == "gguf"
    assert adapter.supported_ops()[-1] == "detokenize"


def test_json_safe_describe_and_runtime_info() -> None:
    adapter = GGUFEngineAdapter()
    json.dumps(adapter.describe())
    json.dumps(adapter.runtime_info())


def test_lookup_missing_engine_adapter_rejects_unknown_name() -> None:
    with pytest.raises(BackendAdapterError, match="not registered"):
        get_engine_adapter("gguf")
