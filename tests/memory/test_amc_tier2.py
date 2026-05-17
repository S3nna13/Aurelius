"""Tests for the minimal AMC Tier-2 episodic hook."""

from __future__ import annotations

from src.memory.amc_tier2 import AMCTier2AblationResult, AMCTier2Config, AMCTier2Hook


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


def test_tier2_hook_reports_evidence_metrics() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.5, max_retrieved=2))

    hook.observe("user", "routine greeting", surprise=0.1)
    hook.observe("user", "deployment region is eu-west-1", surprise=0.8)
    hook.retrieve("deployment")
    hook.retrieve("missing")

    stats = hook.stats()

    assert stats["observed_events"] == 2
    assert stats["stored_events"] == 1
    assert stats["skipped_events"] == 1
    assert stats["write_rate"] == 0.5
    assert stats["retrieval_calls"] == 2
    assert stats["retrieval_query_hits"] == 1
    assert stats["retrieval_fallbacks"] == 1
    assert stats["retrieved_entries"] == 2
    assert stats["average_retrieved_per_call"] == 1.0
    assert stats["retrieval_hit_rate"] == 0.5


def test_tier2_hook_runs_no_memory_ablation_with_term_recall() -> None:
    hook = AMCTier2Hook(AMCTier2Config(surprise_threshold=0.0, max_retrieved=2))
    hook.observe("user", "deployment region is eu-west-1", surprise=0.7)
    hook.observe("assistant", "release gate waits for CI", surprise=0.7)

    result = hook.run_no_memory_ablation(
        "deployment",
        expected_terms=("eu-west-1", "missing-term"),
    )

    assert isinstance(result, AMCTier2AblationResult)
    assert result.query == "deployment"
    assert result.no_memory_context == ""
    assert "eu-west-1" in result.tier2_context
    assert result.retrieved_entries == 1
    assert result.context_delta_chars == len(result.tier2_context)
    assert result.expected_terms_found == ("eu-west-1",)
    assert result.expected_term_recall == 0.5
    assert result.to_dict()["memory_available"] is True
