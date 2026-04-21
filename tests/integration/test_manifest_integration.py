"""Integration tests for the family manifest surface (Cycle-122)."""

from __future__ import annotations

import src.model as model_pkg
import src.model.manifest as manifest_mod
from src.model import AureliusConfig
from src.model.manifest import (
    AURELIUS_REFERENCE_MANIFEST,
    MODEL_MANIFEST_REGISTRY,
    FamilyManifest,
)


def test_registry_exported_from_package():
    assert hasattr(model_pkg, "MODEL_MANIFEST_REGISTRY")
    assert model_pkg.MODEL_MANIFEST_REGISTRY is MODEL_MANIFEST_REGISTRY


def test_reference_manifest_exported_from_package():
    assert hasattr(model_pkg, "AURELIUS_REFERENCE_MANIFEST")
    assert model_pkg.AURELIUS_REFERENCE_MANIFEST is AURELIUS_REFERENCE_MANIFEST


def test_reference_manifest_accessible_via_module():
    assert manifest_mod.AURELIUS_REFERENCE_MANIFEST is AURELIUS_REFERENCE_MANIFEST
    assert isinstance(manifest_mod.AURELIUS_REFERENCE_MANIFEST, FamilyManifest)


def test_aurelius_config_constructs_unchanged():
    # v5: no new config key for manifest -- AureliusConfig construction must
    # remain unaffected by the family manifest surface.
    cfg = AureliusConfig()
    assert cfg is not None
    # Sanity: the reference manifest's max_seq_len aligns with existing config
    # sizing, but this is not enforced by AureliusConfig.
    assert AURELIUS_REFERENCE_MANIFEST.max_seq_len == 8192


def test_reference_is_registered_under_canonical_key():
    assert MODEL_MANIFEST_REGISTRY["aurelius/base-1.395b"] is AURELIUS_REFERENCE_MANIFEST
