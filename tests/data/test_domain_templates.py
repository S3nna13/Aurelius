"""Tests for src.data.domain_templates."""

from __future__ import annotations

from src.data.domain_templates import (
    DOMAIN_TEMPLATES,
    DOMAIN_TEMPLATES_REGISTRY,
    DomainTemplate,
    get_template,
    list_domains,
)


def test_registry_has_default_key():
    assert "default" in DOMAIN_TEMPLATES_REGISTRY


def test_registry_default_points_to_templates():
    assert DOMAIN_TEMPLATES_REGISTRY["default"] is DOMAIN_TEMPLATES


def test_templates_has_at_least_six_entries():
    assert len(DOMAIN_TEMPLATES) >= 6


def test_general_domain_present():
    assert "general" in DOMAIN_TEMPLATES


def test_coding_domain_present():
    assert "coding" in DOMAIN_TEMPLATES


def test_reasoning_domain_present():
    assert "reasoning" in DOMAIN_TEMPLATES


def test_security_domain_present():
    assert "security" in DOMAIN_TEMPLATES


def test_data_science_domain_present():
    assert "data_science" in DOMAIN_TEMPLATES


def test_writing_domain_present():
    assert "writing" in DOMAIN_TEMPLATES


def test_get_template_returns_domain_template():
    t = get_template("coding")
    assert isinstance(t, DomainTemplate)


def test_get_template_correct_domain():
    assert get_template("coding").domain == "coding"


def test_get_template_default_for_unknown():
    assert get_template("does-not-exist").domain == "general"


def test_get_template_empty_string_default():
    assert get_template("").domain == "general"


def test_get_template_case_insensitive():
    assert get_template("CODING").domain == "coding"


def test_list_domains_sorted():
    names = list_domains()
    assert names == sorted(names)


def test_list_domains_length():
    assert len(list_domains()) == len(DOMAIN_TEMPLATES)


def test_list_domains_contains_general():
    assert "general" in list_domains()


def test_all_templates_are_domain_template():
    for t in DOMAIN_TEMPLATES.values():
        assert isinstance(t, DomainTemplate)


def test_all_prefixes_non_empty():
    for t in DOMAIN_TEMPLATES.values():
        assert t.prefix.strip()


def test_all_domains_non_empty():
    for t in DOMAIN_TEMPLATES.values():
        assert t.domain.strip()


def test_tags_are_lists():
    for t in DOMAIN_TEMPLATES.values():
        assert isinstance(t.tags, list)


def test_descriptions_are_strings():
    for t in DOMAIN_TEMPLATES.values():
        assert isinstance(t.description, str)


def test_template_frozen():
    t = get_template("coding")
    try:
        t.domain = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("DomainTemplate should be frozen")


def test_prefix_contains_conversation_language():
    for key in ("general", "coding", "reasoning"):
        assert "conversation" in DOMAIN_TEMPLATES[key].prefix.lower()


def test_coding_tags_includes_code():
    assert "code" in DOMAIN_TEMPLATES["coding"].tags


def test_security_tags_mentions_security():
    assert "security" in DOMAIN_TEMPLATES["security"].tags


def test_writing_tags_mentions_writing():
    assert "writing" in DOMAIN_TEMPLATES["writing"].tags


def test_domain_key_matches_domain_field():
    for key, t in DOMAIN_TEMPLATES.items():
        assert t.domain == key


def test_prefix_minimum_length():
    for t in DOMAIN_TEMPLATES.values():
        assert len(t.prefix) > 50


def test_get_template_whitespace_default():
    assert get_template("   ").domain == "general"
