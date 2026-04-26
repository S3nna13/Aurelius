"""Tests for src/model/variant_adapter.py."""

from __future__ import annotations

import pytest

from src.model.variant_adapter import (
    VARIANT_ADAPTER_ATTACHMENTS,
    VARIANT_ADAPTER_REGISTRY,
    AdapterKind,
    AdapterValidationError,
    VariantAdapter,
    adapters_for_variant,
    attach_to_variant,
    get_adapter,
    list_adapters,
    register_adapter,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved_reg = dict(VARIANT_ADAPTER_REGISTRY)
    saved_att = {k: list(v) for k, v in VARIANT_ADAPTER_ATTACHMENTS.items()}
    VARIANT_ADAPTER_REGISTRY.clear()
    VARIANT_ADAPTER_ATTACHMENTS.clear()
    try:
        yield
    finally:
        VARIANT_ADAPTER_REGISTRY.clear()
        VARIANT_ADAPTER_REGISTRY.update(saved_reg)
        VARIANT_ADAPTER_ATTACHMENTS.clear()
        VARIANT_ADAPTER_ATTACHMENTS.update(saved_att)


def test_all_adapter_kinds_allowed():
    constructors = {
        AdapterKind.LORA: dict(target_modules=("q_proj",), rank=4),
        AdapterKind.PROMPT_TUNE: dict(),
        AdapterKind.PREFIX_TUNE: dict(target_modules=("layer0",)),
        AdapterKind.HEAD_SWAP: dict(weights_path="/tmp/h.pt"),
        AdapterKind.FULL_WEIGHTS_DELTA: dict(),
        AdapterKind.NONE: dict(),
    }
    for kind, extra in constructors.items():
        a = VariantAdapter(id=f"v/{kind.value}", kind=kind, **extra)
        assert a.kind is kind


def test_lora_validates_rank_and_target_modules():
    a = VariantAdapter(id="v/lora", kind=AdapterKind.LORA, target_modules=("q", "k"), rank=8)
    assert a.rank == 8
    assert a.target_modules == ("q", "k")


def test_invalid_lora_missing_rank_raises():
    with pytest.raises(AdapterValidationError):
        VariantAdapter(id="v/bad", kind=AdapterKind.LORA, target_modules=("q",))


def test_invalid_lora_zero_rank_raises():
    with pytest.raises(AdapterValidationError):
        VariantAdapter(id="v/bad", kind=AdapterKind.LORA, target_modules=("q",), rank=0)


def test_invalid_lora_empty_target_modules_raises():
    with pytest.raises(AdapterValidationError):
        VariantAdapter(id="v/bad", kind=AdapterKind.LORA, rank=4)


def test_head_swap_without_weights_path_raises():
    with pytest.raises(AdapterValidationError):
        VariantAdapter(id="v/h", kind=AdapterKind.HEAD_SWAP)


def test_register_get_list():
    a = VariantAdapter(id="v/a", kind=AdapterKind.NONE)
    register_adapter(a)
    assert get_adapter("v/a") is a
    assert "v/a" in list_adapters()


def test_duplicate_registration_raises():
    a = VariantAdapter(id="v/a", kind=AdapterKind.NONE)
    register_adapter(a)
    with pytest.raises(AdapterValidationError):
        register_adapter(a)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get_adapter("missing")


def test_attach_to_variant_round_trip():
    a = VariantAdapter(id="v1/lora", kind=AdapterKind.LORA, target_modules=("q",), rank=2)
    register_adapter(a)
    attach_to_variant("v1", "v1/lora")
    assert VARIANT_ADAPTER_ATTACHMENTS["v1"] == ["v1/lora"]


def test_attach_unknown_adapter_raises():
    with pytest.raises(KeyError):
        attach_to_variant("v1", "nope")


def test_adapters_for_variant_filter():
    a1 = VariantAdapter(id="v1/a", kind=AdapterKind.NONE)
    a2 = VariantAdapter(id="v1/b", kind=AdapterKind.NONE)
    a3 = VariantAdapter(id="v2/a", kind=AdapterKind.NONE)
    register_adapter(a1)
    register_adapter(a2)
    register_adapter(a3)
    got = adapters_for_variant("v1")
    assert {a.id for a in got} == {"v1/a", "v1/b"}
    assert adapters_for_variant("v2") == (a3,)
    assert adapters_for_variant("missing") == ()


def test_unicode_id_allowed():
    a = VariantAdapter(id="variante-ñ/лора", kind=AdapterKind.NONE)
    register_adapter(a)
    assert get_adapter("variante-ñ/лора") is a


def test_metadata_defaults_empty_dict():
    a = VariantAdapter(id="v/m", kind=AdapterKind.NONE)
    assert a.metadata == {}


def test_rank_none_permitted_for_non_lora():
    a = VariantAdapter(id="v/pt", kind=AdapterKind.PROMPT_TUNE)
    assert a.rank is None


def test_determinism_of_listing():
    for k in ("v/c", "v/a", "v/b"):
        register_adapter(VariantAdapter(id=k, kind=AdapterKind.NONE))
    assert list_adapters() == ("v/a", "v/b", "v/c")
    assert list_adapters() == list_adapters()


def test_prefix_tune_requires_target_modules():
    with pytest.raises(AdapterValidationError):
        VariantAdapter(id="v/pt", kind=AdapterKind.PREFIX_TUNE)
    # prompt_tune allows empty target_modules
    VariantAdapter(id="v/pt2", kind=AdapterKind.PROMPT_TUNE)


def test_frozen_dataclass():
    a = VariantAdapter(id="v/f", kind=AdapterKind.NONE)
    with pytest.raises(Exception):
        a.id = "other"  # type: ignore[misc]
