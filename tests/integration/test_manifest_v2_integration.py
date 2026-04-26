"""Integration tests for manifest_v2 surface (Cycle-132, Meta-Prompt v6).

Covers:
    - Reference manifest is importable from the public ``src.model`` surface
      and qualifies as v2.
    - Registry invariant: every registered manifest is either fully v1
      (all three v2 fields ``None``) or fully v2 (all three populated) -- no
      partial-v2 rows.
    - ``compare_backend_contracts(ref, ref)`` is ``"exact"``.
    - Registering + retrieving a new v2 manifest preserves the three
      v2 fields through the single-source-of-truth registry.
    - ``check_manifest_compatibility`` (from compatibility.py) still
      produces a valid verdict when fed v2 manifests.
    - ``dump_manifest(ref)`` round-trips through JSON + ``load_manifest``
      and preserves the three v2 fields.

Pure stdlib.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import src.model as model_pkg
from src.model import (
    AURELIUS_REFERENCE_MANIFEST,
    MANIFEST_SCHEMA_VERSION,
    MODEL_MANIFEST_REGISTRY,
    ManifestValidationError,
    check_checkpoint_compatibility,
    check_manifest_compatibility,
    compare_backend_contracts,
    dump_manifest,
    get_manifest,
    is_v2_manifest,
    list_v2_manifests,
    load_manifest,
    register_manifest,
    upgrade_to_v2,
    v2_to_v1_dict,
)


def _v1_payload(**overrides):
    data = {
        "family_name": "aurelius",
        "variant_name": "integration-v2-variant",
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


# ---------------------------------------------------------------------------
# 1. Public surface exposes v2 symbols and the reference manifest is v2.
# ---------------------------------------------------------------------------


def test_reference_importable_from_src_model_and_is_v2():
    assert hasattr(model_pkg, "AURELIUS_REFERENCE_MANIFEST")
    assert hasattr(model_pkg, "is_v2_manifest")
    assert hasattr(model_pkg, "MANIFEST_SCHEMA_VERSION")
    assert model_pkg.MANIFEST_SCHEMA_VERSION == "2.0.0"
    assert MANIFEST_SCHEMA_VERSION == "2.0.0"
    assert is_v2_manifest(AURELIUS_REFERENCE_MANIFEST) is True


# ---------------------------------------------------------------------------
# 2. Registry invariant: no partial-v2 manifests.
# ---------------------------------------------------------------------------


def test_registry_has_no_partial_v2_manifests():
    for key, m in MODEL_MANIFEST_REGISTRY.items():
        fields = (m.backend_name, m.engine_contract, m.adapter_contract)
        all_none = all(f is None for f in fields)
        all_set = all(f is not None for f in fields)
        assert all_none or all_set, (
            f"manifest {key!r} is partial-v2: "
            f"backend_name={m.backend_name!r}, "
            f"engine_contract={m.engine_contract!r}, "
            f"adapter_contract={m.adapter_contract!r}"
        )


# ---------------------------------------------------------------------------
# 3. compare_backend_contracts on the reference manifest.
# ---------------------------------------------------------------------------


def test_compare_reference_against_itself_is_exact():
    assert (
        compare_backend_contracts(AURELIUS_REFERENCE_MANIFEST, AURELIUS_REFERENCE_MANIFEST)
        == "exact"
    )


# ---------------------------------------------------------------------------
# 4. Registering + retrieving a v2 manifest preserves the three fields.
# ---------------------------------------------------------------------------


def test_register_and_get_manifest_preserves_v2_fields():
    payload = _v1_payload(
        family_name="aurelius-int",
        variant_name="v2-registration-test",
        backend_name="jax",
        engine_contract="1.4.2",
        adapter_contract="2.0.1",
    )
    m = load_manifest(payload)
    assert is_v2_manifest(m) is True

    register_manifest(m)
    try:
        restored = get_manifest("aurelius-int", "v2-registration-test")
        assert restored is m
        assert restored.backend_name == "jax"
        assert restored.engine_contract == "1.4.2"
        assert restored.adapter_contract == "2.0.1"
        # list_v2_manifests sees it lazily.
        assert m in list_v2_manifests()
    finally:
        MODEL_MANIFEST_REGISTRY.pop(m.registry_key, None)


# ---------------------------------------------------------------------------
# 5. check_manifest_compatibility still works on v2 manifests.
# ---------------------------------------------------------------------------


def test_compatibility_check_still_works_on_v2_manifests():
    # Self-comparison -> exact.
    verdict = check_manifest_compatibility(AURELIUS_REFERENCE_MANIFEST, AURELIUS_REFERENCE_MANIFEST)
    assert verdict.compatible is True
    assert verdict.severity == "exact"

    # Backend identity drift is now part of compatibility verdicts.
    alt = dataclasses.replace(
        AURELIUS_REFERENCE_MANIFEST,
        variant_name="compat-v2-alt",
        backend_name="jax",
        engine_contract="1.1.0",
        adapter_contract="1.0.0",
    )
    alt_verdict = check_manifest_compatibility(AURELIUS_REFERENCE_MANIFEST, alt)
    assert alt_verdict.severity == "major_break"
    assert alt_verdict.compatible is False
    assert any("backend contract" in r for r in alt_verdict.reasons)


def test_compatibility_check_allows_legacy_manifest_backend_gap():
    legacy = dataclasses.replace(
        AURELIUS_REFERENCE_MANIFEST,
        backend_name=None,
        engine_contract=None,
        adapter_contract=None,
    )
    verdict = check_manifest_compatibility(legacy, AURELIUS_REFERENCE_MANIFEST)
    assert verdict.compatible is True
    assert verdict.severity == "minor_mismatch"


def test_checkpoint_compatibility_tracks_backend_identity():
    meta = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": None,
        "backend_name": AURELIUS_REFERENCE_MANIFEST.backend_name,
        "engine_contract": AURELIUS_REFERENCE_MANIFEST.engine_contract,
        "adapter_contract": AURELIUS_REFERENCE_MANIFEST.adapter_contract,
    }
    verdict = check_checkpoint_compatibility(AURELIUS_REFERENCE_MANIFEST, meta)
    assert verdict.compatible is True
    assert verdict.severity == "exact"

    meta["backend_name"] = "jax"
    verdict = check_checkpoint_compatibility(AURELIUS_REFERENCE_MANIFEST, meta)
    assert verdict.compatible is False
    assert verdict.severity == "major_break"


# ---------------------------------------------------------------------------
# 6. JSON round-trip preserves the three v2 fields.
# ---------------------------------------------------------------------------


def test_dump_reference_roundtrips_through_json_with_v2_fields():
    encoded = json.dumps(dump_manifest(AURELIUS_REFERENCE_MANIFEST))
    decoded = json.loads(encoded)
    assert decoded["backend_name"] == "pytorch"
    assert decoded["engine_contract"] == "1.0.0"
    assert decoded["adapter_contract"] == "1.0.0"
    restored = load_manifest(decoded)
    assert restored == AURELIUS_REFERENCE_MANIFEST
    assert is_v2_manifest(restored) is True


# ---------------------------------------------------------------------------
# Extra integration coverage (still stdlib-only).
# ---------------------------------------------------------------------------


def test_v2_to_v1_dict_reloads_as_valid_v1_manifest():
    legacy = v2_to_v1_dict(AURELIUS_REFERENCE_MANIFEST)
    restored = load_manifest(legacy)
    assert restored.family_name == AURELIUS_REFERENCE_MANIFEST.family_name
    assert is_v2_manifest(restored) is False


def test_upgrade_to_v2_then_register_is_retrievable():
    base = load_manifest(
        _v1_payload(
            variant_name="int-upgrade-then-register",
        )
    )
    upgraded = upgrade_to_v2(
        base, backend_name="llamacpp", engine_contract="0.9.0", adapter_contract="0.9.0"
    )
    register_manifest(upgraded)
    try:
        got = get_manifest("aurelius", "int-upgrade-then-register")
        assert got.backend_name == "llamacpp"
        assert got.engine_contract == "0.9.0"
        assert got.adapter_contract == "0.9.0"
    finally:
        MODEL_MANIFEST_REGISTRY.pop(upgraded.registry_key, None)


def test_bad_v2_payload_rejected_at_load():
    payload = _v1_payload(
        variant_name="int-bad-v2",
        backend_name="Bad Name",
    )
    with pytest.raises(ManifestValidationError):
        load_manifest(payload)
