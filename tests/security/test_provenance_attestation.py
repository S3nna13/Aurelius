"""Tests for src/security/provenance_attestation.py — AUR-SEC-2026-supply-chain."""

import json
import tempfile

import pytest

from src.security.provenance_attestation import (
    PROVENANCE_ATTESTATION_REGISTRY,
    AttestationVerificationError,
    Material,
    ProvenanceAttestation,
    ProvenanceAttestationBuilder,
    build_file_attestation,
)


@pytest.fixture
def builder() -> ProvenanceAttestationBuilder:
    return ProvenanceAttestationBuilder(
        builder_id="https://aurelius.local/builder/v1",
        shared_key=b"test-key-1234",
    )


def test_build_returns_attestation(builder):
    att = builder.build(build_config={"command": "build foo"})
    assert isinstance(att, ProvenanceAttestation)
    assert att.builder_id == "https://aurelius.local/builder/v1"


def test_build_includes_build_config(builder):
    config = {"command": "build foo", "env": {"FOO": "bar"}}
    att = builder.build(build_config=config)
    assert att.build_config == config


def test_build_includes_materials(builder):
    materials = [Material(uri="pkg:deb/debian/curl@7.50.3", sha256="a" * 64)]
    att = builder.build(materials=materials)
    assert len(att.materials) == 1
    assert att.materials[0].uri == "pkg:deb/debian/curl@7.50.3"


def test_build_includes_recipe(builder):
    recipe = {"type": "https://slsa.dev/build.github-actions/v1"}
    att = builder.build(recipe=recipe)
    assert att.recipe == recipe


def test_build_adds_timestamps_in_metadata(builder):
    att = builder.build()
    assert "generated_at" in att.metadata
    assert "generated_at_iso" in att.metadata
    assert isinstance(att.metadata["generated_at"], float)


def test_build_merges_custom_metadata(builder):
    att = builder.build(metadata={"pipeline": "ci-v2", "runner": "macos-14"})
    assert att.metadata["pipeline"] == "ci-v2"
    assert att.metadata["runner"] == "macos-14"
    assert "generated_at" in att.metadata


def test_build_produces_non_empty_signature(builder):
    att = builder.build()
    assert att.signature
    assert len(att.signature) == 64


def test_build_different_inputs_different_signatures(builder):
    att1 = builder.build(build_config={"cmd": "a"})
    att2 = builder.build(build_config={"cmd": "b"})
    assert att1.signature != att2.signature


def test_verify_valid_attestation(builder):
    att = builder.build(build_config={"cmd": "test"})
    assert builder.verify(att) is True


def test_verify_raises_on_tampered_builder_id(builder):
    att = builder.build(build_config={"cmd": "test"})
    att.builder_id = "https://evil.local/builder"
    with pytest.raises(AttestationVerificationError, match="signature mismatch"):
        builder.verify(att)


def test_verify_raises_on_tampered_material(builder):
    att = builder.build(materials=[Material(uri="good", sha256="a" * 64)])
    att.materials[0] = Material(uri="evil", sha256="b" * 64)
    with pytest.raises(AttestationVerificationError, match="signature mismatch"):
        builder.verify(att)


def test_verify_raises_on_tampered_config(builder):
    att = builder.build(build_config={"cmd": "safe"})
    att.build_config["cmd"] = "rm -rf /"
    with pytest.raises(AttestationVerificationError, match="signature mismatch"):
        builder.verify(att)


def test_verify_raises_on_empty_key():
    no_key_builder = ProvenanceAttestationBuilder(
        builder_id="test",
        shared_key=b"",
    )
    att = no_key_builder.build()
    with pytest.raises(AttestationVerificationError, match="no shared key"):
        no_key_builder.verify(att)


def test_verify_different_key_rejects(builder):
    att = builder.build(build_config={"cmd": "test"})
    other = ProvenanceAttestationBuilder(
        builder_id="https://aurelius.local/builder/v1",
        shared_key=b"different-key",
    )
    with pytest.raises(AttestationVerificationError, match="signature mismatch"):
        other.verify(att)


def test_build_file_attestation(builder):
    with tempfile.NamedTemporaryFile(prefix="att-test-", suffix=".bin") as f:
        f.write(b"hello world")
        f.flush()
        att = build_file_attestation(
            file_path=f.name,
            builder_id=builder._builder_id,
            shared_key=builder._shared_key,
        )
    assert att.build_config["artifact_path"] == f.name
    assert len(att.materials) == 1
    assert att.materials[0].uri == f"file:{f.name}"
    assert isinstance(att.materials[0].sha256, str)
    assert len(att.materials[0].sha256) == 64


def test_attestation_defaults():
    att = ProvenanceAttestation(builder_id="test", build_config={})
    assert att.materials == []
    assert att.recipe == {}
    assert att.metadata == {}
    assert att.signature == ""


def test_attestation_material_immutable():
    m = Material(uri="pkg:npm/foo@1.0.0", sha256="d" * 64)
    assert m.uri == "pkg:npm/foo@1.0.0"
    assert m.sha256 == "d" * 64


def test_registry_contains_default():
    assert "default" in PROVENANCE_ATTESTATION_REGISTRY
    assert isinstance(
        PROVENANCE_ATTESTATION_REGISTRY["default"],
        ProvenanceAttestationBuilder,
    )


def test_payload_is_valid_json(builder):
    att = builder.build(build_config={"cmd": "test"})
    payload = builder._payload(att)
    decoded = json.loads(payload)
    assert decoded["builder_id"] == att.builder_id
    assert decoded["build_config"] == att.build_config
