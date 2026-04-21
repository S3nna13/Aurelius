"""Unit tests for src.model.family."""

from __future__ import annotations

import pytest

from src.model.family import (
    AURELIUS_FAMILY,
    MODEL_FAMILY_REGISTRY,
    MODEL_VARIANT_REGISTRY,
    ModelFamily,
    ModelVariant,
    get_family,
    get_variant_by_id,
    register_family,
    register_variant,
)
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST


def _make_variant(description: str = "test variant") -> ModelVariant:
    return ModelVariant(
        manifest=AURELIUS_REFERENCE_MANIFEST,
        description=description,
    )


def test_aurelius_family_registered() -> None:
    assert AURELIUS_FAMILY is not None
    assert "aurelius" in MODEL_FAMILY_REGISTRY
    assert MODEL_FAMILY_REGISTRY["aurelius"] is AURELIUS_FAMILY


def test_aurelius_default_variant_resolves() -> None:
    family = get_family("aurelius")
    assert family.default_variant == "base-1.395b"
    variant = family.get_variant(family.default_variant)
    assert variant.manifest is AURELIUS_REFERENCE_MANIFEST


def test_add_variant_rejects_duplicates() -> None:
    family = ModelFamily(family_name="dup-test")
    variant = _make_variant()
    family.add_variant("v1", variant)
    with pytest.raises(ValueError):
        family.add_variant("v1", variant)


def test_get_variant_missing_raises() -> None:
    family = ModelFamily(family_name="missing-test")
    with pytest.raises(KeyError):
        family.get_variant("does-not-exist")


def test_list_variants_returns_names_in_order() -> None:
    family = ModelFamily(family_name="list-test")
    family.add_variant("a", _make_variant("a"))
    family.add_variant("b", _make_variant("b"))
    family.add_variant("c", _make_variant("c"))
    assert family.list_variants() == ("a", "b", "c")


def test_register_family_duplicate_raises() -> None:
    with pytest.raises(ValueError):
        register_family(ModelFamily(family_name="aurelius"))


def test_get_family_missing_raises() -> None:
    with pytest.raises(KeyError):
        get_family("no-such-family-xyz")


def test_register_variant_keys_as_expected() -> None:
    family = ModelFamily(family_name="kv-test")
    register_family(family)
    try:
        variant = _make_variant("kv variant")
        variant_id = register_variant("kv-test", "exp-1", variant)
        assert variant_id == "kv-test/exp-1"
        assert MODEL_VARIANT_REGISTRY[variant_id] is variant
        assert family.get_variant("exp-1") is variant
    finally:
        MODEL_FAMILY_REGISTRY.pop("kv-test", None)
        MODEL_VARIANT_REGISTRY.pop("kv-test/exp-1", None)


def test_model_family_registry_populated() -> None:
    assert len(MODEL_FAMILY_REGISTRY) >= 1
    assert "aurelius" in MODEL_FAMILY_REGISTRY


def test_model_variant_registry_populated() -> None:
    assert "aurelius/base-1.395b" in MODEL_VARIANT_REGISTRY
    variant = MODEL_VARIANT_REGISTRY["aurelius/base-1.395b"]
    assert variant.manifest is AURELIUS_REFERENCE_MANIFEST


def test_unicode_family_name_allowed() -> None:
    fam = ModelFamily(family_name="auréliüs-π")
    register_family(fam)
    try:
        assert get_family("auréliüs-π") is fam
    finally:
        MODEL_FAMILY_REGISTRY.pop("auréliüs-π", None)


def test_variant_missing_manifest_raises_type_error() -> None:
    with pytest.raises(TypeError):
        ModelVariant(manifest=None, description="broken")  # type: ignore[arg-type]


def test_release_notes_default_empty() -> None:
    variant = ModelVariant(
        manifest=AURELIUS_REFERENCE_MANIFEST,
        description="no notes",
    )
    assert variant.release_notes == ""


def test_roundtrip_create_register_lookup() -> None:
    family = ModelFamily(family_name="roundtrip")
    register_family(family)
    try:
        variant = _make_variant("rt")
        vid = register_variant("roundtrip", "only", variant)
        fetched_family = get_family("roundtrip")
        fetched_variant = get_variant_by_id(vid)
        assert fetched_family is family
        assert fetched_variant is variant
        assert fetched_family.get_variant("only") is variant
    finally:
        MODEL_FAMILY_REGISTRY.pop("roundtrip", None)
        MODEL_VARIANT_REGISTRY.pop("roundtrip/only", None)


def test_determinism_same_listing_order() -> None:
    family1 = ModelFamily(family_name="det1")
    for n in ("x", "y", "z"):
        family1.add_variant(n, _make_variant(n))
    family2 = ModelFamily(family_name="det2")
    for n in ("x", "y", "z"):
        family2.add_variant(n, _make_variant(n))
    assert family1.list_variants() == family2.list_variants() == ("x", "y", "z")


def test_get_variant_by_id_missing_raises() -> None:
    with pytest.raises(KeyError):
        get_variant_by_id("no-family/no-variant")


def test_register_variant_unknown_family_raises() -> None:
    with pytest.raises(KeyError):
        register_variant("nonexistent-fam", "v1", _make_variant())
