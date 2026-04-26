"""Tests for src/model/manifest.py (Cycle-122, Meta-Prompt v5 family edition)."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from src.model.manifest import (
    AURELIUS_REFERENCE_MANIFEST,
    MODEL_MANIFEST_REGISTRY,
    FamilyManifest,
    ManifestValidationError,
    ReleaseTrack,
    dump_manifest,
    get_manifest,
    list_manifests,
    load_manifest,
    register_manifest,
)

EXPECTED_FIELDS = {
    "family_name",
    "variant_name",
    "backbone_class",
    "tokenizer_name",
    "tokenizer_hash",
    "vocab_size",
    "max_seq_len",
    "context_policy",
    "rope_config",
    "capability_tags",
    "checkpoint_format_version",
    "config_version",
    "compatibility_version",
    "release_track",
    "migration_notes",
}


def _base_payload(**overrides):
    data = {
        "family_name": "aurelius",
        "variant_name": "test-variant",
        "backbone_class": "src.model.transformer.AureliusTransformer",
        "tokenizer_name": "aurelius-bpe",
        "tokenizer_hash": None,
        "vocab_size": 128000,
        "max_seq_len": 8192,
        "context_policy": "rope_yarn",
        "rope_config": {"theta": 500000, "yarn_scale": 1.0},
        "capability_tags": ("chat",),
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "compatibility_version": "1.0.0",
        "release_track": "research",
        "migration_notes": ("initial",),
    }
    data.update(overrides)
    return data


def test_reference_manifest_loads_and_is_autoregistered():
    assert isinstance(AURELIUS_REFERENCE_MANIFEST, FamilyManifest)
    key = "aurelius/base-1.395b"
    assert key in MODEL_MANIFEST_REGISTRY
    assert MODEL_MANIFEST_REGISTRY[key] is AURELIUS_REFERENCE_MANIFEST


def test_all_14_plus_fields_present():
    names = {f.name for f in fields(FamilyManifest)}
    assert EXPECTED_FIELDS <= names
    assert len(EXPECTED_FIELDS) == 15  # 14 required + migration_notes listed


def test_semver_accepts_valid():
    for v in ("1.0.0", "2.10.5", "0.0.1", "10.20.30"):
        m = load_manifest(
            _base_payload(
                variant_name=f"v-{v}",
                checkpoint_format_version=v,
                config_version=v,
                compatibility_version=v,
            )
        )
        assert m.checkpoint_format_version == v


@pytest.mark.parametrize("bad", ["v1", "1.0", "abc", "1.0.0.0", "", "1..0"])
def test_semver_rejects_invalid(bad):
    with pytest.raises(ManifestValidationError):
        load_manifest(_base_payload(checkpoint_format_version=bad))


@pytest.mark.parametrize("track", ["research", "beta", "stable", "deprecated"])
def test_release_track_valid(track):
    m = load_manifest(_base_payload(variant_name=f"rt-{track}", release_track=track))
    assert m.release_track == track


def test_release_track_invalid():
    with pytest.raises(ManifestValidationError):
        load_manifest(_base_payload(release_track="alpha"))


def test_empty_capability_tags_allowed():
    m = load_manifest(_base_payload(variant_name="empty-caps", capability_tags=()))
    assert m.capability_tags == ()


def test_migration_notes_coerced_to_tuple():
    m = load_manifest(_base_payload(variant_name="mig-list", migration_notes=["a", "b", "c"]))
    assert isinstance(m.migration_notes, tuple)
    assert m.migration_notes == ("a", "b", "c")


@pytest.mark.parametrize("bad", [0, -1, -1000])
def test_bad_vocab_size_raises(bad):
    with pytest.raises(ManifestValidationError):
        load_manifest(_base_payload(vocab_size=bad))


def test_bad_max_seq_len_raises():
    with pytest.raises(ManifestValidationError):
        load_manifest(_base_payload(max_seq_len=0))


def test_round_trip_dump_load():
    original = AURELIUS_REFERENCE_MANIFEST
    data = dump_manifest(original)
    restored = load_manifest(data)
    assert restored == original


def test_register_duplicate_raises():
    m = load_manifest(_base_payload(variant_name="dup-test"))
    register_manifest(m)
    with pytest.raises(ManifestValidationError):
        register_manifest(m)


def test_get_manifest_missing_raises():
    with pytest.raises(ManifestValidationError):
        get_manifest("nonexistent-family", "nonexistent-variant")


def test_list_manifests_returns_all():
    manifests = list_manifests()
    assert AURELIUS_REFERENCE_MANIFEST in manifests
    assert len(manifests) == len(MODEL_MANIFEST_REGISTRY)


def test_dump_manifest_is_json_safe():
    data = dump_manifest(AURELIUS_REFERENCE_MANIFEST)
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded["family_name"] == "aurelius"
    assert decoded["variant_name"] == "base-1.395b"


def test_unicode_family_name_allowed():
    m = load_manifest(
        _base_payload(family_name="aurelius-\u4e2d\u6587", variant_name="unicode-variant")
    )
    assert m.family_name == "aurelius-\u4e2d\u6587"
    # Round-trip through JSON.
    json.dumps(dump_manifest(m))


def test_loader_determinism():
    payload = _base_payload(variant_name="det-variant")
    a = load_manifest(payload)
    b = load_manifest(payload)
    assert a == b
    assert dump_manifest(a) == dump_manifest(b)


def test_missing_required_field_raises():
    payload = _base_payload(variant_name="miss")
    del payload["vocab_size"]
    with pytest.raises(ManifestValidationError):
        load_manifest(payload)


def test_unknown_field_rejected():
    payload = _base_payload(variant_name="extra-field")
    payload["unknown_key"] = "oops"
    with pytest.raises(ManifestValidationError):
        load_manifest(payload)


def test_string_capability_tags_rejected():
    with pytest.raises(ManifestValidationError):
        load_manifest(_base_payload(capability_tags="chat"))


def test_manifest_is_frozen():
    with pytest.raises(Exception):
        AURELIUS_REFERENCE_MANIFEST.vocab_size = 42  # type: ignore[misc]


def test_reference_manifest_values():
    m = AURELIUS_REFERENCE_MANIFEST
    assert m.family_name == "aurelius"
    assert m.variant_name == "base-1.395b"
    assert m.vocab_size == 128000
    assert m.max_seq_len == 8192
    assert m.context_policy == "rope_yarn"
    assert m.rope_config == {"theta": 500000, "yarn_scale": 1.0}
    assert m.capability_tags == ("base",)
    assert m.release_track == "research"
    assert ReleaseTrack(m.release_track) is ReleaseTrack.RESEARCH
