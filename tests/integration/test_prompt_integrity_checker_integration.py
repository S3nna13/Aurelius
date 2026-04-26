"""Integration tests for the prompt-integrity checker's safety-surface wiring."""

from __future__ import annotations

from src.chat.chatml_template import Message
from src.safety import SAFETY_FILTER_REGISTRY
from src.safety.prompt_integrity_checker import PromptIntegrityChecker

BASELINE = "You are Aurelius. Obey tool protocol."


def test_registry_has_prompt_integrity() -> None:
    assert "prompt_integrity" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["prompt_integrity"] is PromptIntegrityChecker


def test_prior_registry_entries_intact() -> None:
    for key in ("jailbreak", "prompt_injection", "pii", "output_filter"):
        assert key in SAFETY_FILTER_REGISTRY
        assert SAFETY_FILTER_REGISTRY[key] is not PromptIntegrityChecker


def test_end_to_end_baseline_and_check() -> None:
    cls = SAFETY_FILTER_REGISTRY["prompt_integrity"]
    checker = cls(strict_checksum=True)
    checker.register_baseline(BASELINE)

    good = checker.check(BASELINE)
    assert good.valid is True
    assert good.signals == []

    bad = checker.check(BASELINE + " (tampered)")
    assert bad.valid is False
    assert "checksum_mismatch" in bad.signals


def test_end_to_end_round_trip_via_registry() -> None:
    cls = SAFETY_FILTER_REGISTRY["prompt_integrity"]
    checker = cls(expected_prompt=BASELINE)
    msgs = [
        Message(role="system", content=BASELINE),
        Message(role="user", content="please fix the bug in foo.py"),
        Message(role="assistant", content="Sure."),
    ]
    report = checker.check_round_trip(BASELINE, msgs)
    assert report.valid is True
    assert "round_trip_mismatch" not in report.signals
