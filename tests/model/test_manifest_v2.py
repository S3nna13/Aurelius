"""Tests for src/model/manifest_v2.py (Cycle-132, Meta-Prompt v6).

Covers:
    - v1 manifests (all three new fields None) remain legal.
    - backend_name / engine_contract / adapter_contract validation.
    - AURELIUS_REFERENCE_MANIFEST has backend_name == "pytorch".
    - is_v2_manifest predicate.
    - upgrade_to_v2 returns a new frozen manifest, not a mutation.
    - v2_to_v1_dict drops exactly the three v2 keys and stays JSON-safe.
    - compare_backend_contracts exhaustive verdict matrix.
    - list_v2_manifests reads from the single registry source of truth.
    - load_manifest / dump_manifest preserve the new fields both ways.

Pure stdlib: json, pytest.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from src.model.manifest import (
    AURELIUS_REFERENCE_MANIFEST,
    MODEL_MANIFEST_REGISTRY,
    FamilyManifest,
    ManifestValidationError,
    dump_manifest,
    load_manifest,
)
from src.model.manifest_v2 import (
    MANIFEST_SCHEMA_VERSION,
    compare_backend_contracts,
    is_v2_manifest,
    list_v2_manifests,
    upgrade_to_v2,
    v2_to_v1_dict,
)


def _v1_payload(**overrides):
    data = {
        "family_name": "aurelius",
        "variant_name": "v2-test-variant",
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


def _v1_manifest(**overrides) -> FamilyManifest:
    return load_manifest(_v1_payload(**overrides))


# ---------------------------------------------------------------------------
# Schema version + basic invariants.
# ---------------------------------------------------------------------------


def test_manifest_schema_version_is_2_0_0():
    assert MANIFEST_SCHEMA_VERSION == "2.0.0"


def test_v1_manifest_is_legal_with_all_three_new_fields_none():
    m = _v1_manifest(variant_name="pure-v1")
    assert m.backend_name is None
    assert m.engine_contract is None
    assert m.adapter_contract is None


def test_reference_manifest_has_backend_name_pytorch():
    assert AURELIUS_REFERENCE_MANIFEST.backend_name == "pytorch"
    assert AURELIUS_REFERENCE_MANIFEST.engine_contract == "1.0.0"
    assert AURELIUS_REFERENCE_MANIFEST.adapter_contract == "1.0.0"


# ---------------------------------------------------------------------------
# backend_name validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "pytorch",
        "jax",
        "vllm",
        "llamacpp",
        "onnx",
        "sglang",
        "transformers",
        "pytorch-2",
        "custom_backend",
        "a",
        "a-b_c-d",
    ],
)
def test_backend_name_accepts_valid(name):
    m = _v1_manifest(variant_name=f"bn-{name}")
    upgraded = dataclasses.replace(
        m, backend_name=name, engine_contract="1.0.0", adapter_contract="1.0.0"
    )
    assert upgraded.backend_name == name


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "PyTorch",
        "Py Torch",
        "py torch",
        "pytorch!",
        "pytorch.dev",
        "py/torch",
        "pyTorch",
        "PYTORCH",
        "back end",
    ],
)
def test_backend_name_rejects_invalid(bad):
    m = _v1_manifest(variant_name="bn-bad")
    with pytest.raises(ManifestValidationError):
        dataclasses.replace(m, backend_name=bad)


def test_backend_name_none_is_allowed():
    m = _v1_manifest(variant_name="bn-none")
    assert m.backend_name is None


# ---------------------------------------------------------------------------
# engine_contract / adapter_contract validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", ["1.0.0", "0.0.1", "10.20.30", "2.3.4"])
def test_engine_contract_accepts_valid_semver(v):
    m = _v1_manifest(variant_name=f"ec-{v}")
    out = dataclasses.replace(m, engine_contract=v)
    assert out.engine_contract == v


@pytest.mark.parametrize("bad", ["1.0", "abc", "", "1", "1.0.0.0", "v1.0.0"])
def test_engine_contract_rejects_invalid(bad):
    m = _v1_manifest(variant_name="ec-bad")
    with pytest.raises(ManifestValidationError):
        dataclasses.replace(m, engine_contract=bad)


@pytest.mark.parametrize("v", ["1.0.0", "0.0.1", "10.20.30", "2.3.4"])
def test_adapter_contract_accepts_valid_semver(v):
    m = _v1_manifest(variant_name=f"ac-{v}")
    out = dataclasses.replace(m, adapter_contract=v)
    assert out.adapter_contract == v


@pytest.mark.parametrize("bad", ["1.0", "abc", "", "1", "1.0.0.0", "v1.0.0"])
def test_adapter_contract_rejects_invalid(bad):
    m = _v1_manifest(variant_name="ac-bad")
    with pytest.raises(ManifestValidationError):
        dataclasses.replace(m, adapter_contract=bad)


# ---------------------------------------------------------------------------
# is_v2_manifest.
# ---------------------------------------------------------------------------


def test_is_v2_true_for_reference():
    assert is_v2_manifest(AURELIUS_REFERENCE_MANIFEST) is True


def test_is_v2_false_for_bare_v1():
    m = _v1_manifest(variant_name="v1-bare")
    assert is_v2_manifest(m) is False


@pytest.mark.parametrize(
    "field_name",
    [
        "backend_name",
        "engine_contract",
        "adapter_contract",
    ],
)
def test_is_v2_false_when_any_single_field_missing(field_name):
    m = _v1_manifest(variant_name=f"partial-{field_name}")
    kwargs = {"backend_name": "pytorch", "engine_contract": "1.0.0", "adapter_contract": "1.0.0"}
    kwargs[field_name] = None
    partial = dataclasses.replace(m, **kwargs)
    assert is_v2_manifest(partial) is False


def test_is_v2_rejects_non_manifest():
    with pytest.raises(ManifestValidationError):
        is_v2_manifest({"backend_name": "pytorch"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# upgrade_to_v2.
# ---------------------------------------------------------------------------


def test_upgrade_to_v2_returns_new_object_not_mutate():
    m = _v1_manifest(variant_name="upg-new")
    upgraded = upgrade_to_v2(m, backend_name="jax")
    assert upgraded is not m
    assert m.backend_name is None
    assert upgraded.backend_name == "jax"
    assert upgraded.engine_contract == "1.0.0"
    assert upgraded.adapter_contract == "1.0.0"
    assert is_v2_manifest(upgraded) is True


def test_upgrade_to_v2_preserves_other_fields():
    m = _v1_manifest(variant_name="upg-preserve", capability_tags=("chat", "code"))
    upgraded = upgrade_to_v2(
        m, backend_name="vllm", engine_contract="2.1.0", adapter_contract="1.3.0"
    )
    assert upgraded.family_name == m.family_name
    assert upgraded.variant_name == m.variant_name
    assert upgraded.backbone_class == m.backbone_class
    assert upgraded.vocab_size == m.vocab_size
    assert upgraded.capability_tags == ("chat", "code")
    assert upgraded.engine_contract == "2.1.0"
    assert upgraded.adapter_contract == "1.3.0"


def test_upgrade_to_v2_rejects_invalid_semver():
    m = _v1_manifest(variant_name="upg-bad")
    with pytest.raises(ManifestValidationError):
        upgrade_to_v2(m, backend_name="pytorch", engine_contract="1.0")
    with pytest.raises(ManifestValidationError):
        upgrade_to_v2(m, backend_name="pytorch", adapter_contract="abc")


def test_upgrade_to_v2_rejects_invalid_backend_name():
    m = _v1_manifest(variant_name="upg-bn-bad")
    with pytest.raises(ManifestValidationError):
        upgrade_to_v2(m, backend_name="")
    with pytest.raises(ManifestValidationError):
        upgrade_to_v2(m, backend_name="PyTorch")
    with pytest.raises(ManifestValidationError):
        upgrade_to_v2(m, backend_name="py torch")


def test_upgrade_to_v2_rejects_non_manifest():
    with pytest.raises(ManifestValidationError):
        upgrade_to_v2({"x": 1}, backend_name="pytorch")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# v2_to_v1_dict.
# ---------------------------------------------------------------------------


def test_v2_to_v1_dict_drops_exactly_three_keys():
    d = v2_to_v1_dict(AURELIUS_REFERENCE_MANIFEST)
    assert "backend_name" not in d
    assert "engine_contract" not in d
    assert "adapter_contract" not in d
    # v1-required keys still present.
    for expected in (
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
    ):
        assert expected in d


def test_v2_to_v1_dict_is_json_roundtrippable():
    d = v2_to_v1_dict(AURELIUS_REFERENCE_MANIFEST)
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["family_name"] == "aurelius"
    # Re-load through load_manifest (which treats v2 fields as optional).
    restored = load_manifest(decoded)
    assert restored.family_name == AURELIUS_REFERENCE_MANIFEST.family_name
    assert restored.backend_name is None


def test_v2_to_v1_dict_rejects_non_manifest():
    with pytest.raises(ManifestValidationError):
        v2_to_v1_dict({})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compare_backend_contracts.
# ---------------------------------------------------------------------------


def test_compare_both_none_is_exact():
    a = _v1_manifest(variant_name="cmp-none-a")
    b = _v1_manifest(variant_name="cmp-none-b")
    assert compare_backend_contracts(a, b) == "exact"


def test_compare_one_none_is_minor():
    a = _v1_manifest(variant_name="cmp-one-a")
    b = upgrade_to_v2(_v1_manifest(variant_name="cmp-one-b"), backend_name="pytorch")
    assert compare_backend_contracts(a, b) == "minor_mismatch"
    assert compare_backend_contracts(b, a) == "minor_mismatch"


def test_compare_different_backends_is_major():
    a = upgrade_to_v2(_v1_manifest(variant_name="cmp-diff-a"), backend_name="pytorch")
    b = upgrade_to_v2(_v1_manifest(variant_name="cmp-diff-b"), backend_name="jax")
    assert compare_backend_contracts(a, b) == "major_break"


def test_compare_same_backend_same_semver_is_exact():
    a = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-same-a"),
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
    )
    b = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-same-b"),
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
    )
    assert compare_backend_contracts(a, b) == "exact"


def test_compare_same_backend_minor_diff_semver_is_minor():
    a = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-minor-a"),
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
    )
    b = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-minor-b"),
        backend_name="pytorch",
        engine_contract="1.2.0",
        adapter_contract="1.0.0",
    )
    assert compare_backend_contracts(a, b) == "minor_mismatch"


def test_compare_same_backend_major_diff_semver_is_major():
    a = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-major-a"),
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
    )
    b = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-major-b"),
        backend_name="pytorch",
        engine_contract="2.0.0",
        adapter_contract="1.0.0",
    )
    assert compare_backend_contracts(a, b) == "major_break"


def test_compare_same_backend_major_diff_adapter_is_major():
    a = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-adap-major-a"),
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
    )
    b = upgrade_to_v2(
        _v1_manifest(variant_name="cmp-adap-major-b"),
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="3.0.0",
    )
    assert compare_backend_contracts(a, b) == "major_break"


def test_compare_rejects_non_manifest():
    with pytest.raises(ManifestValidationError):
        compare_backend_contracts(AURELIUS_REFERENCE_MANIFEST, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# list_v2_manifests reads lazily from the single registry.
# ---------------------------------------------------------------------------


def test_list_v2_manifests_contains_reference():
    manifests = list_v2_manifests()
    assert AURELIUS_REFERENCE_MANIFEST in manifests
    for m in manifests:
        assert is_v2_manifest(m) is True


def test_list_v2_manifests_no_duplicate_registry():
    # Sanity: every v2 manifest returned must also be in the single
    # source-of-truth registry.
    for m in list_v2_manifests():
        assert MODEL_MANIFEST_REGISTRY[m.registry_key] is m


# ---------------------------------------------------------------------------
# load_manifest / dump_manifest behavior with v2 fields.
# ---------------------------------------------------------------------------


def test_load_manifest_accepts_payload_without_new_keys():
    payload = _v1_payload(variant_name="load-without")
    m = load_manifest(payload)
    assert m.backend_name is None
    assert m.engine_contract is None
    assert m.adapter_contract is None


def test_load_manifest_accepts_payload_with_new_keys():
    payload = _v1_payload(
        variant_name="load-with",
        backend_name="jax",
        engine_contract="1.2.3",
        adapter_contract="4.5.6",
    )
    m = load_manifest(payload)
    assert m.backend_name == "jax"
    assert m.engine_contract == "1.2.3"
    assert m.adapter_contract == "4.5.6"
    assert is_v2_manifest(m) is True


def test_load_manifest_rejects_bad_backend_name_in_payload():
    payload = _v1_payload(variant_name="load-bad", backend_name="Py Torch")
    with pytest.raises(ManifestValidationError):
        load_manifest(payload)


def test_dump_manifest_always_includes_three_keys():
    v1 = _v1_manifest(variant_name="dump-v1")
    d = dump_manifest(v1)
    assert "backend_name" in d and d["backend_name"] is None
    assert "engine_contract" in d and d["engine_contract"] is None
    assert "adapter_contract" in d and d["adapter_contract"] is None

    d2 = dump_manifest(AURELIUS_REFERENCE_MANIFEST)
    assert d2["backend_name"] == "pytorch"
    assert d2["engine_contract"] == "1.0.0"
    assert d2["adapter_contract"] == "1.0.0"


def test_dump_then_load_roundtrip_preserves_fields():
    original = upgrade_to_v2(
        _v1_manifest(variant_name="rt-preserve"),
        backend_name="vllm",
        engine_contract="2.0.0",
        adapter_contract="1.5.0",
    )
    encoded = json.dumps(dump_manifest(original))
    decoded = json.loads(encoded)
    restored = load_manifest(decoded)
    assert restored == original
    assert restored.backend_name == "vllm"
    assert restored.engine_contract == "2.0.0"
    assert restored.adapter_contract == "1.5.0"
