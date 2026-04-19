"""Integration tests for the PII detector's registration on the safety surface.

Verifies that importing :mod:`src.safety` registers ``"pii"`` under
``SAFETY_FILTER_REGISTRY`` without clobbering the existing jailbreak and
prompt-injection entries, and that the instantiated detector handles a
realistic adversarial blob containing an email, SSN, and valid credit card.
"""

from __future__ import annotations

from src.safety import SAFETY_FILTER_REGISTRY
from src.safety.pii_detector import PIIDetector


def test_registry_has_pii() -> None:
    assert "pii" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["pii"] is PIIDetector


def test_existing_registry_entries_untouched() -> None:
    # The sibling filters must still be present and must not have been
    # overwritten by the PII registration.
    assert "jailbreak" in SAFETY_FILTER_REGISTRY
    assert "prompt_injection" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["jailbreak"] is not PIIDetector
    assert SAFETY_FILTER_REGISTRY["prompt_injection"] is not PIIDetector


def test_end_to_end_adversarial_redaction() -> None:
    cls = SAFETY_FILTER_REGISTRY["pii"]
    detector = cls(redaction_mode="placeholder")
    text = (
        "Hi, my name is Jane Doe. Please reach me at jane.doe@example.com. "
        "My SSN is 123-45-6789 and my card number is 4111 1111 1111 1111. "
        "Thanks!"
    )
    result = detector.redact(text)

    types_found = {m.type for m in result.matches}
    assert "email" in types_found
    assert "ssn" in types_found
    assert "credit_card" in types_found

    # All three originals must be absent from the redacted output.
    assert "jane.doe@example.com" not in result.redacted_text
    assert "123-45-6789" not in result.redacted_text
    assert "4111 1111 1111 1111" not in result.redacted_text

    # Placeholder form should appear for each type.
    assert "<EMAIL>" in result.redacted_text
    assert "<SSN>" in result.redacted_text
    assert "<CREDIT_CARD>" in result.redacted_text
