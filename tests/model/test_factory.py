"""Tests for src.model.factory (Cycle-122, Meta-Prompt v5)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.model.factory import (
    DEFAULT_BACKBONE_BUILDERS,
    FactoryError,
    _DummyBackbone,
    build_backbone_from_manifest,
    build_from_variant_id,
    register_backbone_builder,
)
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST, FamilyManifest


def _dummy_manifest(**overrides) -> FamilyManifest:
    base = dict(
        family_name="test-family",
        variant_name="tiny",
        backbone_class="src.model.factory._DummyBackbone",
        tokenizer_name="aurelius-bpe",
        tokenizer_hash=None,
        vocab_size=1000,
        max_seq_len=128,
        context_policy="rope",
        rope_config={"theta": 500000, "yarn_scale": 1.0},
        capability_tags=("base",),
        checkpoint_format_version="1.0.0",
        config_version="1.0.0",
        compatibility_version="1.0.0",
        release_track="research",
    )
    base.update(overrides)
    return FamilyManifest(**base)


# ---------------------------------------------------------------------------


def test_build_from_manifest_with_stub_class():
    m = _dummy_manifest()
    obj = build_backbone_from_manifest(m)
    assert isinstance(obj, _DummyBackbone)
    assert obj.vocab_size == 1000
    assert obj.max_seq_len == 128
    assert obj.variant_name == "tiny"
    assert obj.family_name == "test-family"


def test_variant_lookup_via_id():
    # Register a throwaway variant pointing at _DummyBackbone.
    from src.model.family import (
        MODEL_VARIANT_REGISTRY,
        ModelVariant,
        get_family,
    )

    m = _dummy_manifest(family_name="aurelius", variant_name="dummy-tiny")
    variant = ModelVariant(
        manifest=m,
        description="Dummy for factory tests",
    )
    # Insert only if not already present.
    variant_id = "aurelius/dummy-tiny"
    if variant_id not in MODEL_VARIANT_REGISTRY:
        fam = get_family("aurelius")
        fam.add_variant("dummy-tiny", variant)
        MODEL_VARIANT_REGISTRY[variant_id] = variant

    obj = build_from_variant_id(variant_id)
    assert isinstance(obj, _DummyBackbone)
    assert obj.variant_name == "dummy-tiny"


def test_factory_error_on_unknown_variant():
    with pytest.raises(FactoryError):
        build_from_variant_id("no-such-family/no-such-variant")


def test_factory_error_on_missing_backbone_class():
    m = _dummy_manifest(backbone_class="src.model.does_not_exist.Nope")
    with pytest.raises(FactoryError):
        build_backbone_from_manifest(m)


def test_factory_error_on_missing_attribute_in_module():
    m = _dummy_manifest(backbone_class="src.model.factory.MissingAttr")
    with pytest.raises(FactoryError):
        build_backbone_from_manifest(m)


def test_register_backbone_builder_override():
    sentinel = object()

    def _override(cls, manifest, cfg):
        return sentinel

    path = "src.model.factory._DummyBackbone"
    original = DEFAULT_BACKBONE_BUILDERS.get(path)
    try:
        register_backbone_builder(path, _override)
        m = _dummy_manifest()
        assert build_backbone_from_manifest(m) is sentinel
    finally:
        if original is not None:
            DEFAULT_BACKBONE_BUILDERS[path] = original


def test_default_builders_has_reference_entry():
    assert (
        "src.model.transformer.AureliusTransformer"
        in DEFAULT_BACKBONE_BUILDERS
    )
    assert callable(
        DEFAULT_BACKBONE_BUILDERS["src.model.transformer.AureliusTransformer"]
    )


def test_build_with_none_config_uses_stub_defaults():
    m = _dummy_manifest()
    obj = build_backbone_from_manifest(m, aurelius_config=None)
    # _DummyBackbone has d_model default 64 and n_layers default 2.
    assert obj.d_model == 64
    assert obj.n_layers == 2


def test_build_is_deterministic_for_dummy():
    m = _dummy_manifest()
    a = build_backbone_from_manifest(m)
    b = build_backbone_from_manifest(m)
    assert (a.vocab_size, a.max_seq_len, a.variant_name) == (
        b.vocab_size,
        b.max_seq_len,
        b.variant_name,
    )


def test_reference_variant_builds_something_with_override():
    # Override the heavy AureliusTransformer builder to avoid a 1.4B alloc.
    path = "src.model.transformer.AureliusTransformer"
    original = DEFAULT_BACKBONE_BUILDERS.get(path)

    def _tiny(cls, manifest, cfg):
        return _DummyBackbone(
            vocab_size=manifest.vocab_size,
            max_seq_len=manifest.max_seq_len,
            variant_name=manifest.variant_name,
            family_name=manifest.family_name,
        )

    try:
        register_backbone_builder(path, _tiny)
        obj = build_from_variant_id("aurelius/base-1.395b")
        assert isinstance(obj, _DummyBackbone)
        assert obj.family_name == "aurelius"
    finally:
        if original is not None:
            DEFAULT_BACKBONE_BUILDERS[path] = original


def test_unicode_variant_name_handling():
    m = _dummy_manifest(variant_name="tiny-\u00fcnicode")
    obj = build_backbone_from_manifest(m)
    assert obj.variant_name == "tiny-\u00fcnicode"


def test_register_backbone_builder_validates_inputs():
    with pytest.raises(FactoryError):
        register_backbone_builder("", lambda *a, **k: None)
    with pytest.raises(FactoryError):
        register_backbone_builder("some.path", "not-callable")  # type: ignore[arg-type]


def test_build_rejects_non_manifest_input():
    with pytest.raises(FactoryError):
        build_backbone_from_manifest({"not": "a manifest"})  # type: ignore[arg-type]


def test_build_from_variant_id_rejects_empty_string():
    with pytest.raises(FactoryError):
        build_from_variant_id("")
