"""Tests for encryptor."""

from __future__ import annotations

import os

import pytest

# Set a valid Fernet key before importing the encryptor module.
# AURELIUS_ENCRYPTION_KEY is required by SimpleEncryptor.__post_init__.
os.environ.setdefault(
    "AURELIUS_ENCRYPTION_KEY",
    "eraA96t0Jt605u3a6it1Z58dZXraqjM22HCNv4RYb7U=",
)

from src.security.encryptor import SimpleEncryptor, _get_default_encryptor  # noqa: E402

# Detect whether cryptography is available.
_CRYPTOGAPHY_AVAILABLE = True
try:
    _get_default_encryptor()
except ImportError:
    _CRYPTOGAPHY_AVAILABLE = False


@pytest.mark.skipif(not _CRYPTOGAPHY_AVAILABLE, reason="cryptography not installed")
class TestSimpleEncryptor:
    def test_encrypt_decrypt_roundtrip(self):
        enc = SimpleEncryptor()
        ct = enc.encrypt("hello world")
        pt = enc.decrypt(ct)
        assert pt == "hello world"

    def test_different_plaintexts(self):
        enc = SimpleEncryptor()
        ct1 = enc.encrypt("abc")
        ct2 = enc.encrypt("xyz")
        assert ct1 != ct2
