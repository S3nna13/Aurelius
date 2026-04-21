"""Tests for src.model.compatibility (Cycle-122, Meta-Prompt v5)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.model.compatibility import (
    CompatibilityError,
    CompatibilityVerdict,
    SemverParts,
    assert_compatible,
    check_checkpoint_compatibility,
    check_manifest_compatibility,
    parse_semver,
)
from src.model.manifest import AURELIUS_REFERENCE_MANIFEST, FamilyManifest


def _base() -> FamilyManifest:
    return AURELIUS_REFERENCE_MANIFEST


# ---------------------------------------------------------------------------
# semver parse
# ---------------------------------------------------------------------------


def test_parse_semver_valid():
    parts = parse_semver("1.2.3")
    assert isinstance(parts, SemverParts)
    assert (parts.major, parts.minor, parts.patch) == (1, 2, 3)


def test_parse_semver_invalid_raises():
    with pytest.raises(CompatibilityError):
        parse_semver("1.2")
    with pytest.raises(CompatibilityError):
        parse_semver("not-a-version")
    with pytest.raises(CompatibilityError):
        parse_semver(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# manifest compatibility
# ---------------------------------------------------------------------------


def test_manifest_exact_match():
    v = check_manifest_compatibility(_base(), _base())
    assert v.compatible is True
    assert v.severity == "exact"
    assert v.reasons == ()


def test_manifest_minor_mismatch_forward_compat():
    other = replace(_base(), compatibility_version="1.1.0")
    v = check_manifest_compatibility(_base(), other)
    assert v.compatible is True
    assert v.severity == "minor_mismatch"
    assert any("compatibility_version" in r for r in v.reasons)


def test_manifest_major_break_on_compatibility_version():
    other = replace(_base(), compatibility_version="2.0.0")
    v = check_manifest_compatibility(_base(), other)
    assert v.compatible is False
    assert v.severity == "major_break"


def test_manifest_tokenizer_name_mismatch_is_major():
    other = replace(_base(), tokenizer_name="some-other-bpe")
    v = check_manifest_compatibility(_base(), other)
    assert v.severity == "major_break"
    assert any("tokenizer_name" in r for r in v.reasons)


def test_manifest_vocab_size_mismatch_is_major():
    other = replace(_base(), vocab_size=64000)
    v = check_manifest_compatibility(_base(), other)
    assert v.severity == "major_break"
    assert any("vocab_size" in r for r in v.reasons)


def test_manifest_capability_superset_is_ok():
    required = replace(_base(), capability_tags=("base",))
    candidate = replace(_base(), capability_tags=("base", "chat"))
    v = check_manifest_compatibility(required, candidate)
    assert v.compatible is True
    assert v.severity == "exact"


def test_manifest_capability_missing_is_major():
    required = replace(_base(), capability_tags=("base", "chat"))
    candidate = replace(_base(), capability_tags=("base",))
    v = check_manifest_compatibility(required, candidate)
    assert v.severity == "major_break"
    assert any("capability_tags" in r for r in v.reasons)


def test_manifest_rope_theta_mismatch_is_major():
    other = replace(_base(), rope_config={"theta": 10000, "yarn_scale": 1.0})
    v = check_manifest_compatibility(_base(), other)
    assert v.severity == "major_break"
    assert any("theta" in r for r in v.reasons)


def test_manifest_rope_yarn_scale_diff_is_minor():
    other = replace(_base(), rope_config={"theta": 500000, "yarn_scale": 4.0})
    v = check_manifest_compatibility(_base(), other)
    assert v.severity == "minor_mismatch"
    assert v.compatible is True
    assert any("yarn_scale" in r for r in v.reasons)


def test_manifest_backbone_class_mismatch_is_major():
    other = replace(_base(), backbone_class="src.model.factory._DummyBackbone")
    v = check_manifest_compatibility(_base(), other)
    assert v.severity == "major_break"


# ---------------------------------------------------------------------------
# checkpoint compatibility
# ---------------------------------------------------------------------------


def test_checkpoint_format_major_mismatch_is_major():
    meta = {
        "checkpoint_format_version": "2.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": None,
    }
    v = check_checkpoint_compatibility(_base(), meta)
    assert v.severity == "major_break"
    assert v.compatible is False


def test_checkpoint_config_minor_drift_is_minor():
    meta = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.3.0",
        "tokenizer_hash": None,
    }
    v = check_checkpoint_compatibility(_base(), meta)
    assert v.severity == "minor_mismatch"
    assert v.compatible is True


def test_checkpoint_tokenizer_hash_mismatch_when_both_set():
    man = replace(_base(), tokenizer_hash="abc123")
    meta = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": "xyz789",
    }
    v = check_checkpoint_compatibility(man, meta)
    assert v.severity == "major_break"


def test_checkpoint_tokenizer_hash_none_is_permissive():
    man = replace(_base(), tokenizer_hash=None)
    meta = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": "anything",
    }
    v = check_checkpoint_compatibility(man, meta)
    assert v.severity == "exact"

    man = replace(_base(), tokenizer_hash="abc123")
    meta = {
        "checkpoint_format_version": "1.0.0",
        "config_version": "1.0.0",
        "tokenizer_hash": None,
    }
    v = check_checkpoint_compatibility(man, meta)
    assert v.severity == "exact"


# ---------------------------------------------------------------------------
# assert_compatible
# ---------------------------------------------------------------------------


def test_assert_compatible_raises_on_major():
    other = replace(_base(), compatibility_version="2.0.0")
    v = check_manifest_compatibility(_base(), other)
    with pytest.raises(CompatibilityError):
        assert_compatible(v)


def test_assert_compatible_silent_on_minor():
    other = replace(_base(), rope_config={"theta": 500000, "yarn_scale": 4.0})
    v = check_manifest_compatibility(_base(), other)
    assert_compatible(v)  # must not raise


def test_assert_compatible_silent_on_exact():
    v = check_manifest_compatibility(_base(), _base())
    assert_compatible(v)


def test_verdict_reasons_populated_on_failure():
    other = replace(_base(), vocab_size=64000, tokenizer_name="other-bpe")
    v = check_manifest_compatibility(_base(), other)
    assert len(v.reasons) >= 2
    assert not v.compatible


def test_verdict_construction_validates_severity():
    with pytest.raises(CompatibilityError):
        CompatibilityVerdict(
            compatible=True, reasons=(), severity="not-a-real-severity"
        )
