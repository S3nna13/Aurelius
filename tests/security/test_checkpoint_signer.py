"""Tests for src/security/checkpoint_signer.py — AUR-SEC-2026-supply-chain."""

import tempfile

import pytest

from src.security.checkpoint_signer import (
    CHECKPOINT_SIGNER_REGISTRY,
    CheckpointIntegrityError,
    CheckpointSigner,
)


@pytest.fixture
def hmac_signer() -> CheckpointSigner:
    return CheckpointSigner(hmac_key=b"test-hmac-key-32bytes!")


@pytest.fixture
def sample_digest() -> str:
    return "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


# ---------------------------------------------------------------------------
# HMAC roundtrip
# ---------------------------------------------------------------------------


def test_sign_and_verify_hmac_roundtrip(hmac_signer, sample_digest):
    sig = hmac_signer.sign_digest(sample_digest)
    assert sig
    assert len(sig) == 64
    assert hmac_signer.verify_digest(sample_digest, sig) is True


def test_verify_raises_on_wrong_signature(hmac_signer, sample_digest):
    sig = hmac_signer.sign_digest(sample_digest)
    wrong_digest = "0" * 64
    with pytest.raises(CheckpointIntegrityError, match="signature does not match"):
        hmac_signer.verify_digest(wrong_digest, sig)


def test_verify_raises_on_tampered_signature(hmac_signer, sample_digest):
    sig = hmac_signer.sign_digest(sample_digest)
    tampered = "f" + sig[1:]
    with pytest.raises(CheckpointIntegrityError, match="signature does not match"):
        hmac_signer.verify_digest(sample_digest, tampered)


def test_sign_file_and_verify_file(hmac_signer):
    with tempfile.NamedTemporaryFile(prefix="ckpt-", suffix=".pt") as f:
        f.write(b"model checkpoint data here")
        f.flush()
        sig = hmac_signer.sign_file(f.name)
        assert sig
        assert hmac_signer.verify_file(f.name, sig) is True


def test_verify_file_raises_on_tampered_file(hmac_signer):
    with tempfile.NamedTemporaryFile(prefix="ckpt-", suffix=".pt") as f:
        f.write(b"original content")
        f.flush()
        sig = hmac_signer.sign_file(f.name)
    with open(f.name, "a") as f:
        f.write("tampered")
    with pytest.raises(CheckpointIntegrityError, match="signature does not match"):
        hmac_signer.verify_file(f.name, sig)


# ---------------------------------------------------------------------------
# Algorithm detection
# ---------------------------------------------------------------------------


def test_signer_algorithm_label_default():
    signer = CheckpointSigner(hmac_key=b"key")
    assert signer.algorithm == "hmac-sha256"


def test_signer_algorithm_fallback_when_crypto_unavailable():
    """When cryptography is not installed, Ed25519 key bytes fall back to HMAC."""
    signer = CheckpointSigner(private_key_bytes=b"k" * 32)
    assert signer.algorithm == "hmac-sha256"


# ---------------------------------------------------------------------------
# Ed25519 (conditional on cryptography library)
# ---------------------------------------------------------------------------


def test_ed25519_sign_and_verify_roundtrip():
    pytest.importorskip("cryptography")
    import os

    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key_bytes = os.urandom(32)
    signer = CheckpointSigner(private_key_bytes=private_key_bytes)
    assert signer.algorithm == "ed25519"

    digest = "ab" * 32
    sig = signer.sign_digest(digest)
    assert sig
    assert len(sig) > 0

    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    public_key_bytes = private_key.public_key().public_bytes_raw()
    verify_signer = CheckpointSigner(public_key_bytes=public_key_bytes)
    assert verify_signer.verify_digest(digest, sig) is True


def test_ed25519_rejects_wrong_digest():
    pytest.importorskip("cryptography")
    import os

    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key_bytes = os.urandom(32)
    signer = CheckpointSigner(private_key_bytes=private_key_bytes)
    digest = "ab" * 32
    sig = signer.sign_digest(digest)

    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    public_key_bytes = private_key.public_key().public_bytes_raw()
    verify_signer = CheckpointSigner(public_key_bytes=public_key_bytes)
    wrong_digest = "ba" * 32
    with pytest.raises(CheckpointIntegrityError, match="signature does not match"):
        verify_signer.verify_digest(wrong_digest, sig)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_registry_contains_entry():
    assert "default" in CHECKPOINT_SIGNER_REGISTRY
    assert isinstance(CHECKPOINT_SIGNER_REGISTRY["default"], CheckpointSigner)
