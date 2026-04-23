"""Tests for system_prompt_priority."""

from __future__ import annotations

import pytest

from src.chat.system_prompt_priority import (
    PrincipalHierarchyConflict,
    SystemPromptFragment,
    SystemPromptPriority,
    SystemPromptPriorityEncoder,
)


def _f(priority, content, sid="s", immutable=False):
    return SystemPromptFragment(
        priority=priority, content=content, source_id=sid, immutable=immutable
    )


def test_merge_respects_priority_order():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.USER, "user-text", "u1"),
        _f(SystemPromptPriority.DEVELOPER, "dev-text", "d1"),
        _f(SystemPromptPriority.OPERATOR, "op-text", "o1"),
    ]
    out = enc.merge(frags)
    # developer appears before operator appears before user
    assert out.index("dev-text") < out.index("op-text") < out.index("user-text")


def test_each_fragment_has_source_marker():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "alpha", "dev-1"),
        _f(SystemPromptPriority.USER, "beta", "user-1"),
    ]
    out = enc.merge(frags)
    assert "[SOURCE:dev-1 PRIORITY:DEVELOPER]" in out
    assert "[SOURCE:user-1 PRIORITY:USER]" in out


def test_max_total_chars_truncation_drops_lowest_priority_first():
    enc = SystemPromptPriorityEncoder(max_total_chars=120)
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "A" * 30, "d"),
        _f(SystemPromptPriority.USER, "B" * 30, "u"),
        _f(SystemPromptPriority.MODEL_DEFAULT, "C" * 30, "m"),
    ]
    out = enc.merge(frags)
    assert "A" * 30 in out
    assert "C" * 30 not in out  # lowest priority dropped first


def test_immutable_cannot_be_truncated_raises():
    enc = SystemPromptPriorityEncoder(max_total_chars=50)
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "X" * 100, "d", immutable=True),
        _f(SystemPromptPriority.USER, "Y" * 100, "u", immutable=True),
    ]
    with pytest.raises(PrincipalHierarchyConflict):
        enc.merge(frags)


def test_separator_used_between_fragments():
    enc = SystemPromptPriorityEncoder(separator="\n###\n")
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "a", "d"),
        _f(SystemPromptPriority.USER, "b", "u"),
    ]
    out = enc.merge(frags)
    assert "\n###\n" in out


def test_empty_list_returns_empty_string():
    enc = SystemPromptPriorityEncoder()
    assert enc.merge([]) == ""


def test_single_fragment_returned():
    enc = SystemPromptPriorityEncoder()
    out = enc.merge([_f(SystemPromptPriority.DEVELOPER, "only", "d")])
    assert "only" in out
    assert "[SOURCE:d PRIORITY:DEVELOPER]" in out


def test_detect_conflicts_always_vs_never():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "always cite sources", "d"),
        _f(SystemPromptPriority.USER, "never cite sources", "u"),
    ]
    conflicts = enc.detect_conflicts(frags)
    assert len(conflicts) == 1
    assert conflicts[0][2] == "cite"


def test_resolve_drops_lower_priority_conflict():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "always cite sources", "d"),
        _f(SystemPromptPriority.USER, "never cite sources", "u"),
    ]
    out = enc.resolve(frags)
    assert "always cite sources" in out
    assert "never cite sources" not in out


def test_unicode_content_preserved():
    enc = SystemPromptPriorityEncoder()
    frags = [_f(SystemPromptPriority.DEVELOPER, "日本語 résumé 🎉", "d")]
    out = enc.merge(frags)
    assert "日本語 résumé 🎉" in out


def test_determinism():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.USER, "u", "u"),
        _f(SystemPromptPriority.DEVELOPER, "d", "d"),
    ]
    assert enc.merge(frags) == enc.merge(list(frags))


def test_custom_max_total_chars():
    enc = SystemPromptPriorityEncoder(max_total_chars=500)
    assert enc.max_total_chars == 500


def test_priority_order_stable_for_same_priority():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.USER, "first", "u1"),
        _f(SystemPromptPriority.USER, "second", "u2"),
        _f(SystemPromptPriority.USER, "third", "u3"),
    ]
    out = enc.merge(frags)
    assert out.index("first") < out.index("second") < out.index("third")


def test_source_id_uniqueness_not_required():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "alpha", "same"),
        _f(SystemPromptPriority.USER, "beta", "same"),
    ]
    out = enc.merge(frags)
    # both markers still printed (two occurrences)
    assert out.count("[SOURCE:same ") == 2


def test_empty_content_ignored_gracefully():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "", "d-empty"),
        _f(SystemPromptPriority.USER, "kept", "u"),
    ]
    out = enc.merge(frags)
    assert "kept" in out
    assert "d-empty" not in out


def test_invalid_max_total_chars_raises():
    with pytest.raises(ValueError):
        SystemPromptPriorityEncoder(max_total_chars=0)


def test_no_conflicts_returns_empty_list():
    enc = SystemPromptPriorityEncoder()
    frags = [
        _f(SystemPromptPriority.DEVELOPER, "always cite", "d"),
        _f(SystemPromptPriority.USER, "always summarize", "u"),
    ]
    assert enc.detect_conflicts(frags) == []
