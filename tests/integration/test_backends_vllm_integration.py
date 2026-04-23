"""Integration tests for the vLLM and SGLang backend adapters.

Verifies that both adapters register cleanly into the global registries and
that select_backend_for_manifest resolves a vLLM-tagged manifest to a
VLLMAdapter instance.
"""

from __future__ import annotations

import pytest

from src.backends.registry import BACKEND_REGISTRY, ENGINE_ADAPTER_REGISTRY


# ---------------------------------------------------------------------------
# Registry-level fixtures: snapshot + restore both registries around each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Snapshot both registries before each test and restore afterwards."""
    before_backend = dict(BACKEND_REGISTRY)
    before_engine = dict(ENGINE_ADAPTER_REGISTRY)
    yield
    BACKEND_REGISTRY.clear()
    BACKEND_REGISTRY.update(before_backend)
    ENGINE_ADAPTER_REGISTRY.clear()
    ENGINE_ADAPTER_REGISTRY.update(before_engine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_vllm() -> None:
    """Register the vLLM adapters after clearing any stale entries."""
    BACKEND_REGISTRY.pop("vllm", None)
    ENGINE_ADAPTER_REGISTRY.pop("vllm", None)
    from src.backends.vllm_adapter import register
    register()


def _register_sglang() -> None:
    """Register the SGLang adapters after clearing any stale entries."""
    BACKEND_REGISTRY.pop("sglang", None)
    ENGINE_ADAPTER_REGISTRY.pop("sglang", None)
    from src.backends.sglang_adapter import register
    register()


# ---------------------------------------------------------------------------
# vLLM registry tests
# ---------------------------------------------------------------------------


def test_vllm_in_backend_registry_after_import_and_register() -> None:
    """'vllm' must appear in BACKEND_REGISTRY after register() is called."""
    import src.backends  # noqa: F401 – ensure package is loaded
    _register_vllm()
    assert "vllm" in BACKEND_REGISTRY


def test_vllm_in_engine_adapter_registry_after_import_and_register() -> None:
    """'vllm' must appear in ENGINE_ADAPTER_REGISTRY after register() is called."""
    import src.backends  # noqa: F401
    _register_vllm()
    assert "vllm" in ENGINE_ADAPTER_REGISTRY


def test_vllm_backend_adapter_is_correct_type() -> None:
    from src.backends.vllm_adapter import VLLMAdapter

    _register_vllm()
    assert isinstance(BACKEND_REGISTRY["vllm"], VLLMAdapter)


def test_vllm_engine_adapter_is_correct_type() -> None:
    from src.backends.vllm_adapter import VLLMEngineAdapter

    _register_vllm()
    assert isinstance(ENGINE_ADAPTER_REGISTRY["vllm"], VLLMEngineAdapter)


# ---------------------------------------------------------------------------
# SGLang registry tests
# ---------------------------------------------------------------------------


def test_sglang_in_backend_registry_after_import_and_register() -> None:
    """'sglang' must appear in BACKEND_REGISTRY after register() is called."""
    import src.backends  # noqa: F401
    _register_sglang()
    assert "sglang" in BACKEND_REGISTRY


def test_sglang_in_engine_adapter_registry_after_import_and_register() -> None:
    """'sglang' must appear in ENGINE_ADAPTER_REGISTRY after register() is called."""
    import src.backends  # noqa: F401
    _register_sglang()
    assert "sglang" in ENGINE_ADAPTER_REGISTRY


def test_sglang_backend_adapter_is_correct_type() -> None:
    from src.backends.sglang_adapter import SGLangAdapter

    _register_sglang()
    assert isinstance(BACKEND_REGISTRY["sglang"], SGLangAdapter)


def test_sglang_engine_adapter_is_correct_type() -> None:
    from src.backends.sglang_adapter import SGLangEngineAdapter

    _register_sglang()
    assert isinstance(ENGINE_ADAPTER_REGISTRY["sglang"], SGLangEngineAdapter)


# ---------------------------------------------------------------------------
# select_backend_for_manifest with backend_name="vllm"
# ---------------------------------------------------------------------------


def test_select_backend_for_manifest_vllm_returns_vllm_adapter() -> None:
    """select_backend_for_manifest must return the registered VLLMAdapter instance."""
    import dataclasses
    from src.backends.registry import select_backend_for_manifest
    from src.backends.vllm_adapter import VLLMAdapter
    from src.model.manifest import AURELIUS_REFERENCE_MANIFEST

    _register_vllm()

    # Derive a manifest that targets the vllm backend.
    manifest = dataclasses.replace(AURELIUS_REFERENCE_MANIFEST, backend_name="vllm")
    adapter = select_backend_for_manifest(manifest)
    assert isinstance(adapter, VLLMAdapter)
    assert adapter is BACKEND_REGISTRY["vllm"]


def test_select_backend_for_manifest_sglang_returns_sglang_adapter() -> None:
    """select_backend_for_manifest must return the registered SGLangAdapter instance."""
    import dataclasses
    from src.backends.registry import select_backend_for_manifest
    from src.backends.sglang_adapter import SGLangAdapter
    from src.model.manifest import AURELIUS_REFERENCE_MANIFEST

    _register_sglang()

    manifest = dataclasses.replace(AURELIUS_REFERENCE_MANIFEST, backend_name="sglang")
    adapter = select_backend_for_manifest(manifest)
    assert isinstance(adapter, SGLangAdapter)
    assert adapter is BACKEND_REGISTRY["sglang"]


# ---------------------------------------------------------------------------
# Both adapters can coexist in the same registry
# ---------------------------------------------------------------------------


def test_vllm_and_sglang_coexist_in_registry() -> None:
    _register_vllm()
    _register_sglang()

    assert "vllm" in BACKEND_REGISTRY
    assert "sglang" in BACKEND_REGISTRY
    assert "pytorch" in BACKEND_REGISTRY  # should still be present
