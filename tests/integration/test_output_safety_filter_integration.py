"""Integration tests for the composed output safety filter."""

from __future__ import annotations

from src.safety import (
    HARM_CLASSIFIER_REGISTRY,
    SAFETY_FILTER_REGISTRY,
    FilterDecision,
    HarmTaxonomyClassifier,
    JailbreakDetector,
    OutputSafetyFilter,
    OutputSafetyPolicy,
    PIIDetector,
    PromptInjectionScanner,
)


def test_registry_contains_output_filter() -> None:
    assert "output_filter" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["output_filter"] is OutputSafetyFilter


def test_prior_registry_entries_intact() -> None:
    # Previously-registered safety filters must still resolve to their
    # original classes.
    assert SAFETY_FILTER_REGISTRY["jailbreak"] is JailbreakDetector
    assert SAFETY_FILTER_REGISTRY["prompt_injection"] is PromptInjectionScanner
    assert SAFETY_FILTER_REGISTRY["pii"] is PIIDetector
    # Harm taxonomy lives in its own registry.
    assert HARM_CLASSIFIER_REGISTRY["harm_taxonomy"] is HarmTaxonomyClassifier


def test_end_to_end_benign_sample() -> None:
    cls = SAFETY_FILTER_REGISTRY["output_filter"]
    f = cls(OutputSafetyPolicy(mode="permissive"))
    d = f.filter("Here is a clean code snippet: def add(a, b): return a + b")
    assert isinstance(d, FilterDecision)
    assert d.action == "allow"


def test_end_to_end_pii_sample() -> None:
    f = OutputSafetyFilter(OutputSafetyPolicy(mode="permissive"))
    d = f.filter("You can reach dave@example.com for details.")
    assert d.action == "redact"
    assert d.redacted_text is not None
    assert "dave@example.com" not in d.redacted_text


def test_end_to_end_harm_sample() -> None:
    f = OutputSafetyFilter(OutputSafetyPolicy(mode="permissive"))
    d = f.filter(
        "Here is a python keylogger ransomware that will encrypt the "
        "user's files and demand ransom to a C2 server."
    )
    assert d.action == "block"
    assert d.signal_breakdown["harm_taxonomy"]["flagged"] is True


def test_end_to_end_chunked_stream() -> None:
    f = OutputSafetyFilter(
        OutputSafetyPolicy(mode="permissive", max_buffer_chars=512)
    )
    buffer = ""
    chunks = ["Sure, ", "here ", "is ", "a ", "greeting!"]
    last_decision = None
    for c in chunks:
        last_decision, buffer = f.filter_chunk(c, buffer=buffer)
        assert last_decision.action == "allow"
    assert last_decision is not None
    assert "greeting" in buffer
