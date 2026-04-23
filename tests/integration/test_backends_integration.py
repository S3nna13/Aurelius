"""Integration tests for the Aurelius backend adapter surface."""

from __future__ import annotations

import pytest

from src.backends import (
    BACKEND_REGISTRY,
    ENGINE_ADAPTER_REGISTRY,
    BackendBuildResult,
    build_with_backend,
    register_engine_adapter,
)
from src.backends.base import BackendContract, EngineAdapter
from src.backends.registry import (
    get_engine_adapter,
    list_engine_adapters,
)


class _RoundTripEngine(EngineAdapter):
    def __init__(self, name: str) -> None:
        self._contract = BackendContract(
            backend_name=name,
            engine_contract="0.1.0",
            adapter_contract="0.1.0",
            supports_training=False,
            supports_inference=True,
            capability_tags=("inference",),
        )

    @property
    def contract(self) -> BackendContract:
        return self._contract

    def is_available(self) -> bool:
        return True

    def describe(self) -> dict:
        return {"name": self._contract.backend_name}

    def supported_ops(self) -> tuple[str, ...]:
        return ("generate",)


@pytest.fixture(autouse=True)
def _snapshot_engine_registry():
    before = dict(ENGINE_ADAPTER_REGISTRY)
    yield
    ENGINE_ADAPTER_REGISTRY.clear()
    ENGINE_ADAPTER_REGISTRY.update(before)


def test_backend_registry_contains_pytorch() -> None:
    assert "pytorch" in BACKEND_REGISTRY
    adapter = BACKEND_REGISTRY["pytorch"]
    assert adapter.contract.backend_name == "pytorch"


def test_select_backend_for_manifest_reexported_from_src_model() -> None:
    # The re-export must be usable without touching src.backends directly.
    from src.model import select_backend_for_manifest as select_from_model
    from src.model import AURELIUS_REFERENCE_MANIFEST

    adapter = select_from_model(AURELIUS_REFERENCE_MANIFEST)
    assert adapter.contract.backend_name == "pytorch"


def test_select_reference_manifest_returns_pytorch_adapter() -> None:
    from src.backends import select_backend_for_manifest
    from src.model import AURELIUS_REFERENCE_MANIFEST

    adapter = select_backend_for_manifest(AURELIUS_REFERENCE_MANIFEST)
    assert adapter.contract.backend_name == "pytorch"
    assert adapter is BACKEND_REGISTRY["pytorch"]


def test_engine_adapter_registry_reachable_and_initially_empty() -> None:
    # ``initially empty`` here means no engine adapters have been
    # registered by the import-time surface. The fixture ensures other
    # tests can't leak state into this check.
    assert isinstance(ENGINE_ADAPTER_REGISTRY, dict)
    assert ENGINE_ADAPTER_REGISTRY == {}


def test_engine_adapter_round_trip() -> None:
    engine = _RoundTripEngine("integration-engine")
    register_engine_adapter(engine)
    assert "integration-engine" in list_engine_adapters()
    fetched = get_engine_adapter("integration-engine")
    assert fetched is engine
    assert fetched.describe() == {"name": "integration-engine"}


def test_build_with_backend_returns_adapter_bundle() -> None:
    from src.model import AURELIUS_REFERENCE_MANIFEST

    result = build_with_backend(AURELIUS_REFERENCE_MANIFEST)
    assert isinstance(result, BackendBuildResult)
    assert result.adapter.contract.backend_name == "pytorch"
    assert result.backbone is None


def test_build_with_backend_rejects_non_manifest() -> None:
    from src.backends.base import BackendAdapterError

    with pytest.raises(BackendAdapterError, match="FamilyManifest"):
        build_with_backend({"not": "a manifest"})
