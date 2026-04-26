"""Tests for encryptor."""

from __future__ import annotations

import pytest

from src.security.encryptor import SimpleEncryptor, _get_default_encryptor

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
