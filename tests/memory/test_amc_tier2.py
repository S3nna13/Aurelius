"""Tests for the minimal AMC Tier-2 episodic hook."""

from __future__ import annotations

from src.memory.amc_tier2 import AMCTier2Config, AMCTier2Hook


def test_tier2_hook_stores_only_surprising_events_by_default() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.6))

    ignored = hook.observe("user", "routine greeting", surprise=0.2)
    stored = hook.observe("user", "project codename is atlas", surprise=0.9)

    assert ignored is None
    assert stored is not None
    assert stored.content == "project codename is atlas"
    assert hook.stats()["episodic_entries"] == 1


def test_tier2_hook_retrieves_query_matches_then_recent_fallback() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0, max_retrieved=2))
    hook.observe("user", "favorite color is blue", surprise=0.1)
    hook.observe("assistant", "stored preference", surprise=0.1)
    hook.observe("user", "favorite city is kyoto", surprise=0.1)

    color_context = hook.retrieve("color")
    assert [entry.content for entry in color_context] == ["favorite color is blue"]

    fallback_context = hook.retrieve("missing")
    assert [entry.content for entry in fallback_context] == [
        "stored preference",
        "favorite city is kyoto",
    ]


def test_tier2_hook_builds_prompt_context() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0, max_retrieved=2))
    hook.observe("user", "name is Mira", surprise=0.7, importance=0.8)

    context = hook.build_context("name")

    assert "Tier-2 episodic memory:" in context
    assert "user: name is Mira" in context
