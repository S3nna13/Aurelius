"""Tests for data masker."""

from __future__ import annotations

from src.security.data_masker import DataMasker


class TestDataMasker:
    def test_masks_email(self):
        dm = DataMasker()
        masked = dm.mask("contact me at user@example.com")
        assert "user@example.com" not in masked
        assert "[REDACTED]" in masked

    def test_masks_ssn(self):
        dm = DataMasker()
        masked = dm.mask("SSN: 123-45-6789")
        assert "123-45-6789" not in masked

    def test_masks_credit_card(self):
        dm = DataMasker()
        masked = dm.mask("card: 4111111111111111")
        assert "4111111111111111" not in masked

    def test_mask_with_type(self):
        dm = DataMasker()
        masked = dm.mask_with_type("email: test@test.com")
        assert "***@***.***" in masked

    def test_no_false_positives(self):
        dm = DataMasker()
        text = "this is clean text with no PII"
        assert dm.mask(text) == text

    def test_add_custom_pattern(self):
        dm = DataMasker()
        dm.add_pattern(r"\bSECRET-\d+\b", "***SECRET***")
        masked = dm.mask("my key is SECRET-42")
        assert "SECRET-42" not in masked
        assert "[REDACTED]" in masked
