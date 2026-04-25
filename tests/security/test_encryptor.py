"""Tests for encryptor."""
from __future__ import annotations

import pytest

from src.security import encryptor as enc_mod


@pytest.mark.skipif(enc_mod.SIMPLE_ENCRYPTOR is None, reason="cryptography not installed")
class TestSimpleEncryptor:
    def test_encrypt_decrypt_roundtrip(self):
        enc = enc_mod.SimpleEncryptor()
        ct = enc.encrypt("hello world")
        pt = enc.decrypt(ct)
        assert pt == "hello world"

    def test_different_plaintexts(self):
        enc = enc_mod.SimpleEncryptor()
        ct1 = enc.encrypt("abc")
        ct2 = enc.encrypt("xyz")
        assert ct1 != ct2