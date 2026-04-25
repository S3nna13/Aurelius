"""Tests for the Aurelius GGUF / llama.cpp-style engine adapter."""

from __future__ import annotations

import json
import sys

import pytest

from src.backends.base import BackendAdapterError, EngineAdapter
from src.backends.gguf_engine_adapter import GGUFEngineAdapter, register
from src.backends.registry import ENGINE_ADAPTER_REGISTRY


@pytest.fixture(autouse=True)
def _snapshot_registry() -> None:
    before = dict(ENGINE_ADAPTER_REGISTRY)
    yield
    ENGINE_ADAPTER_REGISTRY.clear()
    ENGINE_ADAPTER_REGISTRY.update(before)


def test_module_does_not_import_foreign_runtime_packages() -> None:
    assert "llama_cpp" not in sys.modules
    assert "llama_cpp_python" not in sys.modules


def test_adapter_is_engine_adapter_instance() -> None:
    adapter = GGUFEngineAdapter()
    assert isinstance(adapter, EngineAdapter)


def test_contract_identity_and_capabilities() -> None:
    adapter = GGUFEngineAdapter()
    contract = adapter.contract
    assert contract.backend_name == "gguf"
    assert contract.engine_contract == "1.0.0"
    assert contract.adapter_contract == "1.0.0"
    assert contract.supports_training is False
    assert contract.supports_inference is True
    assert "gguf" in contract.capability_tags
    assert "llama_cpp" in contract.capability_tags


def test_supported_ops_are_stable() -> None:
    adapter = GGUFEngineAdapter()
    assert adapter.supported_ops() == (
        "load_model",
        "generate",
        "stream_generate",
        "tokenize",
        "detokenize",
    )


def test_describe_is_json_safe_and_structured() -> None:
    adapter = GGUFEngineAdapter()
    payload = adapter.describe()
    json.dumps(payload)
    assert payload["backend_name"] == "gguf"
    assert payload["engine_family"] == "llama.cpp"
    assert payload["format"] == "gguf"
    assert isinstance(payload["probe_modules"], list)
    assert isinstance(payload["probe_results"], dict)
    assert isinstance(payload["supported_ops"], list)
    assert isinstance(payload["quantization_formats"], list)
    assert payload["contract"]["backend_name"] == "gguf"


def test_runtime_info_reports_availability_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str) -> object | None:
        return object() if name == "llama_cpp" else None

    monkeypatch.setattr(
        "src.backends.gguf_engine_adapter.importlib.util.find_spec",
        fake_find_spec,
    )
    adapter = GGUFEngineAdapter()
    info = adapter.runtime_info()
    json.dumps(info)
    assert info["available"] is True
    assert info["probe_results"]["llama_cpp"] is True
    assert info["probe_results"]["llama_cpp_python"] is False


def test_runtime_info_is_false_when_no_probe_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.backends.gguf_engine_adapter.importlib.util.find_spec",
        lambda name: None,
    )
    adapter = GGUFEngineAdapter()
    assert adapter.is_available() is False
    assert adapter.runtime_info()["available"] is False


def test_constructor_rejects_empty_probe_module_names() -> None:
    with pytest.raises(BackendAdapterError, match="probe_modules"):
        GGUFEngineAdapter(probe_modules=())


def test_constructor_rejects_probe_module_names_with_empty_entry() -> None:
    with pytest.raises(BackendAdapterError, match="probe_modules"):
        GGUFEngineAdapter(probe_modules=("llama_cpp", ""))


def test_register_is_idempotent_and_populates_registry() -> None:
    assert "gguf" not in ENGINE_ADAPTER_REGISTRY
    register()
    before = ENGINE_ADAPTER_REGISTRY["gguf"]
    register()
    assert ENGINE_ADAPTER_REGISTRY["gguf"] is before
