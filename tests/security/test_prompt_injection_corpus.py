"""Tests for prompt_injection_corpus — regression test data integrity."""
from __future__ import annotations

import pytest

from src.security.prompt_injection_corpus import (
    InjectionPattern,
    PromptInjectionCorpus,
    CORPUS_REGISTRY,
    DEFAULT_PROMPT_INJECTION_CORPUS,
)


# ---------------------------------------------------------------------------
# Default corpus contents
# ---------------------------------------------------------------------------


def test_default_corpus_has_patterns():
    c = PromptInjectionCorpus()
    assert c.count() > 0


def test_by_category_filters():
    c = PromptInjectionCorpus()
    system_leaks = c.by_category("system_leak")
    assert all(p.category == "system_leak" for p in system_leaks)


def test_by_severity_filters():
    c = PromptInjectionCorpus()
    critical = c.by_severity("critical")
    assert all(p.severity == "critical" for p in critical)


def test_by_surface_filters():
    c = PromptInjectionCorpus()
    tool_hijacks = c.by_surface("tool_call")
    assert all(p.target_surface == "tool_call" for p in tool_hijacks)


# ---------------------------------------------------------------------------
# Additive extension
# ---------------------------------------------------------------------------


def test_add_increases_count():
    c = PromptInjectionCorpus()
    before = c.count()
    c.add(InjectionPattern("PI-999", "custom", "test", "user_input", "low"))
    assert c.count() == before + 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in CORPUS_REGISTRY
    assert isinstance(CORPUS_REGISTRY["default"], PromptInjectionCorpus)


def test_default_is_corpus():
    assert isinstance(DEFAULT_PROMPT_INJECTION_CORPUS, PromptInjectionCorpus)
