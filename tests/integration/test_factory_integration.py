"""Integration: factory + compatibility exposed from src.model; config stable."""

from __future__ import annotations

from dataclasses import fields

from src.model import (
    AureliusConfig,
    CompatibilityError,
    CompatibilityVerdict,
    DEFAULT_BACKBONE_BUILDERS,
    FactoryError,
    FamilyManifest,
    assert_compatible,
    build_backbone_from_manifest,
    build_from_variant_id,
    check_checkpoint_compatibility,
    check_manifest_compatibility,
    parse_semver,
    register_backbone_builder,
)


def test_factory_and_compatibility_exported_from_src_model():
    # Smoke-check that every public surface imports cleanly.
    assert callable(build_backbone_from_manifest)
    assert callable(build_from_variant_id)
    assert callable(register_backbone_builder)
    assert callable(check_manifest_compatibility)
    assert callable(check_checkpoint_compatibility)
    assert callable(assert_compatible)
    assert callable(parse_semver)
    assert isinstance(DEFAULT_BACKBONE_BUILDERS, dict)
    assert issubclass(FactoryError, Exception)
    assert issubclass(CompatibilityError, Exception)
    assert CompatibilityVerdict is not None
    assert FamilyManifest is not None


def test_aurelius_config_surface_unchanged():
    # Sanity: factory must not have added new flags to AureliusConfig.
    cfg = AureliusConfig()
    assert cfg.d_model == 2048
    assert cfg.n_layers == 24
    assert cfg.vocab_size == 128_000
    # Check a representative sample of existing fields survive intact.
    field_names = {f.name for f in fields(AureliusConfig)}
    for required in (
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "max_seq_len",
        "rope_theta",
    ):
        assert required in field_names


def test_compatibility_roundtrip_on_reference_manifest():
    from src.model.manifest import AURELIUS_REFERENCE_MANIFEST

    verdict = check_manifest_compatibility(
        AURELIUS_REFERENCE_MANIFEST, AURELIUS_REFERENCE_MANIFEST
    )
    assert verdict.compatible is True
    assert verdict.severity == "exact"
    assert_compatible(verdict)
