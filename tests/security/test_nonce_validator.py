"""Tests for nonce validator."""
from __future__ import annotations

import time

import pytest

from src.security.nonce_validator import NonceValidator


class TestNonceValidator:
    def test_valid_nonce(self):
        nv = NonceValidator(max_age_seconds=60)
        assert nv.validate("abc123", time.time()) is True

    def test_replay_rejected(self):
        nv = NonceValidator(max_age_seconds=60)
        ts = time.time()
        assert nv.validate("same", ts) is True
        assert nv.validate("same", ts) is False

    def test_expired_rejected(self):
        nv = NonceValidator(max_age_seconds=1)
        assert nv.validate("old", time.time() - 10) is False

    def test_is_valid_tuple(self):
        nv = NonceValidator(max_age_seconds=60)
        ok, msg = nv.is_valid("n1", time.time())
        assert ok is True
        assert msg == "ok"

    def test_is_valid_expired(self):
        nv = NonceValidator(max_age_seconds=1)
        ok, msg = nv.is_valid("n2", time.time() - 10)
        assert ok is False
        assert "expired" in msg