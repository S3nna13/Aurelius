"""Registry tests for the backend surface.

These tests exercise the module-level registries in isolation using a
tiny stub adapter, so that registry semantics stay honest without any
foreign imports. No ``torch`` import here.
"""

from __future__ import annotations

import pytest

import src.backends as backends_pkg
from src.backends.base import (
    BackendAdapter,
    BackendAdapterError,
    BackendContract,
    EngineAdapter,
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
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST, FamilyManifest


class _StubBackend(BackendAdapter):
    def __init__(self, name: str = "stub-backend") -> None:
        self._contract = BackendContract(
            backend_name=name,
            engine_contract="1.0.0",
            adapter_contract="1.0.0",
            supports_training=False,
            supports_inference=False,
            capability_tags=("test",),
        )

    @property
    def contract(self) -> BackendContract:
        return self._contract

    def tensor_ns(self) -> object:
        class _NS:
            def zeros(self, shape, dtype=None):
                return [0] * (shape[0] if shape else 0)

            def ones(self, shape, dtype=None):
                return [1] * (shape[0] if shape else 0)

            def as_array(self, obj):
                return list(obj)

        return _NS()

    def dtype_of(self, obj: object) -> str:
        return type(obj).__name__

    def device_of(self, obj: object) -> str:
        return "stub"

    def is_available(self) -> bool:
        return True


class _StubEngine(EngineAdapter):
    def __init__(self, name: str = "stub-engine") -> None:
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
def _snapshot_registries():
    """Snapshot registries before each test and restore after."""
    back_before = dict(BACKEND_REGISTRY)
    eng_before = dict(ENGINE_ADAPTER_REGISTRY)
    yield
    BACKEND_REGISTRY.clear()
    BACKEND_REGISTRY.update(back_before)
    ENGINE_ADAPTER_REGISTRY.clear()
    ENGINE_ADAPTER_REGISTRY.update(eng_before)


# ---------------------------------------------------------------------------
# Backend registry API
# ---------------------------------------------------------------------------


def test_register_backend_happy_path() -> None:
    adapter = _StubBackend("stub-one")
    register_backend(adapter)
    assert get_backend("stub-one") is adapter


def test_register_backend_duplicate_rejected() -> None:
    adapter = _StubBackend("stub-dup")
    register_backend(adapter)
    with pytest.raises(BackendAdapterError, match="already registered"):
        register_backend(_StubBackend("stub-dup"))


def test_register_backend_overwrite_allowed() -> None:
    first = _StubBackend("stub-over")
    second = _StubBackend("stub-over")
    register_backend(first)
    register_backend(second, overwrite=True)
    assert get_backend("stub-over") is second


def test_register_backend_rejects_non_adapter() -> None:
    with pytest.raises(BackendAdapterError, match="BackendAdapter"):
        register_backend("not an adapter")  # type: ignore[arg-type]


def test_get_backend_missing_raises() -> None:
    with pytest.raises(BackendAdapterError, match="not registered"):
        get_backend("no-such-backend")


def test_get_backend_rejects_empty_name() -> None:
    with pytest.raises(BackendAdapterError, match="non-empty"):
        get_backend("")


def test_list_backends_sorted() -> None:
    register_backend(_StubBackend("zeta"))
    register_backend(_StubBackend("alpha"))
    names = list_backends()
    assert names == tuple(sorted(names))
    assert "alpha" in names and "zeta" in names


# ---------------------------------------------------------------------------
# Engine registry API (mirror)
# ---------------------------------------------------------------------------


def test_engine_registry_round_trip() -> None:
    engine = _StubEngine("stub-engine-a")
    register_engine_adapter(engine)
    assert get_engine_adapter("stub-engine-a") is engine
    assert "stub-engine-a" in list_engine_adapters()


def test_engine_registry_rejects_duplicate() -> None:
    register_engine_adapter(_StubEngine("engine-dup"))
    with pytest.raises(BackendAdapterError, match="already registered"):
        register_engine_adapter(_StubEngine("engine-dup"))


def test_engine_registry_rejects_non_engine() -> None:
    with pytest.raises(BackendAdapterError, match="EngineAdapter"):
        register_engine_adapter(_StubBackend("not-an-engine"))  # type: ignore[arg-type]


def test_engine_registry_overwrite() -> None:
    first = _StubEngine("engine-over")
    second = _StubEngine("engine-over")
    register_engine_adapter(first)
    register_engine_adapter(second, overwrite=True)
    assert get_engine_adapter("engine-over") is second


def test_get_engine_adapter_missing_raises() -> None:
    with pytest.raises(BackendAdapterError, match="not registered"):
        get_engine_adapter("nope")


# ---------------------------------------------------------------------------
# select_backend_for_manifest
# ---------------------------------------------------------------------------


def test_select_backend_for_manifest_default_is_pytorch() -> None:
    adapter = select_backend_for_manifest(AURELIUS_REFERENCE_MANIFEST)
    assert adapter.contract.backend_name == "pytorch"
    # This is also what BACKEND_REGISTRY["pytorch"] points at.
    assert adapter is BACKEND_REGISTRY["pytorch"]


def test_select_backend_for_manifest_with_explicit_backend() -> None:
    register_backend(_StubBackend("explicit-backend"))
    manifest = FamilyManifest(
        family_name="stub",
        variant_name="v1",
        backbone_class="src.model.factory._DummyBackbone",
        tokenizer_name="stub-tok",
        tokenizer_hash=None,
        vocab_size=256,
        max_seq_len=64,
        context_policy="none",
        rope_config={},
        capability_tags=("test",),
        checkpoint_format_version="1.0.0",
        config_version="1.0.0",
        compatibility_version="1.0.0",
        release_track="research",
        backend_name="explicit-backend",
    )
    adapter = select_backend_for_manifest(manifest)
    assert adapter.contract.backend_name == "explicit-backend"


def test_select_backend_for_manifest_unknown_backend_raises() -> None:
    manifest = FamilyManifest(
        family_name="stub",
        variant_name="v1",
        backbone_class="src.model.factory._DummyBackbone",
        tokenizer_name="stub-tok",
        tokenizer_hash=None,
        vocab_size=256,
        max_seq_len=64,
        context_policy="none",
        rope_config={},
        capability_tags=("test",),
        checkpoint_format_version="1.0.0",
        config_version="1.0.0",
        compatibility_version="1.0.0",
        release_track="research",
        backend_name="no-such-backend",
    )
    with pytest.raises(BackendAdapterError, match="not registered"):
        select_backend_for_manifest(manifest)


def test_select_backend_for_manifest_rejects_non_manifest() -> None:
    with pytest.raises(BackendAdapterError, match="FamilyManifest"):
        select_backend_for_manifest({"backend_name": "pytorch"})


# ---------------------------------------------------------------------------
# Module-level registry reachability
# ---------------------------------------------------------------------------


def test_registries_are_dicts_on_package() -> None:
    assert isinstance(backends_pkg.BACKEND_REGISTRY, dict)
    assert isinstance(backends_pkg.ENGINE_ADAPTER_REGISTRY, dict)
    # Same objects as those exported from registry submodule.
    assert backends_pkg.BACKEND_REGISTRY is BACKEND_REGISTRY
    assert backends_pkg.ENGINE_ADAPTER_REGISTRY is ENGINE_ADAPTER_REGISTRY
