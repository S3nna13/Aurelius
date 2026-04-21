"""Integration tests for the model family surface (Cycle-122)."""

from __future__ import annotations

import src.model as model_pkg
from src.model import AureliusConfig
from src.model.family import (
    AURELIUS_FAMILY,
    MODEL_FAMILY_REGISTRY,
    MODEL_VARIANT_REGISTRY,
    ModelFamily,
    ModelVariant,
)


def test_family_registry_exported_from_package():
    assert hasattr(model_pkg, "MODEL_FAMILY_REGISTRY")
    assert model_pkg.MODEL_FAMILY_REGISTRY is MODEL_FAMILY_REGISTRY


def test_variant_registry_exported_from_package():
    assert hasattr(model_pkg, "MODEL_VARIANT_REGISTRY")
    assert model_pkg.MODEL_VARIANT_REGISTRY is MODEL_VARIANT_REGISTRY


def test_aurelius_family_exported_from_package():
    assert hasattr(model_pkg, "AURELIUS_FAMILY")
    assert model_pkg.AURELIUS_FAMILY is AURELIUS_FAMILY


def test_model_family_classes_exported_from_package():
    assert model_pkg.ModelFamily is ModelFamily
    assert model_pkg.ModelVariant is ModelVariant


def test_reference_family_accessible():
    fam = MODEL_FAMILY_REGISTRY["aurelius"]
    assert fam.family_name == "aurelius"
    assert fam.default_variant == "base-1.395b"
    assert "base-1.395b" in fam.variants


def test_reference_variant_links_to_reference_manifest():
    variant = MODEL_VARIANT_REGISTRY["aurelius/base-1.395b"]
    assert variant.manifest is model_pkg.AURELIUS_REFERENCE_MANIFEST


def test_aurelius_config_default_unchanged():
    # v5: no new config flags introduced by the family surface.
    cfg = AureliusConfig()
    assert cfg is not None
    # Field set must not have gained a family-specific flag.
    field_names = {f for f in vars(cfg)}
    for forbidden in ("family", "variant", "model_family", "model_variant"):
        assert forbidden not in field_names, (
            f"AureliusConfig unexpectedly grew a {forbidden!r} field"
        )
