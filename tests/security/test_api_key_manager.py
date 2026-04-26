"""Tests for API key manager."""

from __future__ import annotations

from src.security.api_key_manager import APIKeyManager


class TestAPIKeyManager:
    def test_generate_and_validate(self):
        mgr = APIKeyManager()
        raw, key = mgr.generate("test-key")
        assert raw.startswith("ak_")
        validated = mgr.validate(raw)
        assert validated is not None
        assert validated.name == "test-key"

    def test_validate_invalid_key(self):
        mgr = APIKeyManager()
        assert mgr.validate("invalid_key") is None

    def test_revoke(self):
        mgr = APIKeyManager()
        raw, _ = mgr.generate("temp")
        assert mgr.revoke(raw) is True
        assert mgr.validate(raw) is None

    def test_revoke_unknown(self):
        mgr = APIKeyManager()
        assert mgr.revoke("fake") is False

    def test_list_keys(self):
        mgr = APIKeyManager()
        mgr.generate("k1")
        mgr.generate("k2")
        assert len(mgr.list_keys()) == 2
