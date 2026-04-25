"""Tests for secrets scanner."""
from __future__ import annotations

import pytest
import tempfile

from src.security.secrets_scanner import SecretsScanner


class TestSecretsScanner:
    def test_detects_aws_key(self):
        scanner = SecretsScanner()
        results = scanner.scan("AKIA0123456789ABCDEF")
        assert any(r.match_type == "aws_key" for r in results)

    def test_detects_github_token(self):
        scanner = SecretsScanner()
        results = scanner.scan("ghp_" + "a" * 36)
        assert any(r.match_type == "github_token" for r in results)

    def test_detects_bearer_token(self):
        scanner = SecretsScanner()
        results = scanner.scan("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test")
        assert any(r.match_type == "bearer_token" for r in results)

    def test_mask_short_value(self):
        from src.security.secrets_scanner import SecretMatch
        m = SecretMatch(value="12345678", match_type="test", line=1, column=1)
        masked = m.masked()
        assert "***" in masked

    def test_no_false_positives_clean_text(self):
        scanner = SecretsScanner()
        results = scanner.scan("hello world this is a test")
        assert len(results) == 0

    def test_scan_file(self):
        scanner = SecretsScanner()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("password = 'supersecret12345678'")
            f.flush()
            path = f.name
        results = scanner.scan_file(path)
        assert len(results) > 0