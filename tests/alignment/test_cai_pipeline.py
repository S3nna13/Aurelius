"""Tests for src/alignment/cai_pipeline.py — Constitutional AI pipeline."""

from __future__ import annotations

from src.alignment.cai_pipeline import (
    CAIConfig,
    CAIDataCollector,
    CAIReviser,
    build_critique_prompt,
    build_revision_prompt,
    extract_critique_score,
    score_response_safety,
)

# ---------------------------------------------------------------------------
# Shared mock generate function
# ---------------------------------------------------------------------------


def mock_generate(prompt: str) -> str:
    return "This is a helpful and appropriate response."


# ---------------------------------------------------------------------------
# 1. CAIConfig defaults: 3 principles
# ---------------------------------------------------------------------------


def test_config_default_n_principles():
    cfg = CAIConfig()
    assert cfg.principles is not None
    assert len(cfg.principles) == 3


# ---------------------------------------------------------------------------
# 2. CAIConfig default n_revision_steps
# ---------------------------------------------------------------------------


def test_config_default_n_revision_steps():
    cfg = CAIConfig()
    assert cfg.n_revision_steps == 2


# ---------------------------------------------------------------------------
# 3. build_critique_prompt contains principle and response
# ---------------------------------------------------------------------------


def test_build_critique_prompt_contains_principle_and_response():
    template = "Principle: {principle}\nResponse: {response}\nCritique:"
    result = build_critique_prompt("My response.", "Be honest.", template)
    assert "Be honest." in result
    assert "My response." in result


# ---------------------------------------------------------------------------
# 4. build_revision_prompt contains original response and critique
# ---------------------------------------------------------------------------


def test_build_revision_prompt_contains_response_and_critique():
    template = "Original: {response}\nCritique: {critique}\nRevised:"
    result = build_revision_prompt("Original text.", "It has issues.", template)
    assert "Original text." in result
    assert "It has issues." in result


# ---------------------------------------------------------------------------
# 5. extract_critique_score returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_extract_critique_score_returns_float_in_range():
    score = extract_critique_score("This response is good and helpful.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 6. extract_critique_score: negative text scores lower than positive text
# ---------------------------------------------------------------------------


def test_extract_critique_score_negative_lower_than_positive():
    negative = "This response is harmful, dangerous, and inappropriate."
    positive = "This response is good, helpful, and appropriate."
    neg_score = extract_critique_score(negative)
    pos_score = extract_critique_score(positive)
    assert neg_score < pos_score


# ---------------------------------------------------------------------------
# 7. CAIReviser.critique returns string
# ---------------------------------------------------------------------------


def test_cai_reviser_critique_returns_string():
    cfg = CAIConfig()
    reviser = CAIReviser(mock_generate, cfg)
    result = reviser.critique("Some response.", "Be honest.")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 8. CAIReviser.revise returns string
# ---------------------------------------------------------------------------


def test_cai_reviser_revise_returns_string():
    cfg = CAIConfig()
    reviser = CAIReviser(mock_generate, cfg)
    result = reviser.revise("Some response.", "It could be more honest.")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 9. run_revision_cycle returns a list
# ---------------------------------------------------------------------------


def test_run_revision_cycle_returns_list():
    cfg = CAIConfig(n_revision_steps=1)
    reviser = CAIReviser(mock_generate, cfg)
    result = reviser.run_revision_cycle("Initial response.")
    assert isinstance(result, list)
    # 1 step × 3 principles = 3 entries
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 10. run_full_pipeline returns tuple of (str, list)
# ---------------------------------------------------------------------------


def test_run_full_pipeline_returns_tuple():
    cfg = CAIConfig(n_revision_steps=1)
    reviser = CAIReviser(mock_generate, cfg)
    result = reviser.run_full_pipeline("Initial response.")
    assert isinstance(result, tuple)
    assert len(result) == 2
    final, history = result
    assert isinstance(final, str)
    assert isinstance(history, list)


# ---------------------------------------------------------------------------
# 11. score_response_safety returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_score_response_safety_returns_float_in_range():
    score = score_response_safety("This is a nice helpful response.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 12. score_response_safety: harmful text scores lower than safe text
# ---------------------------------------------------------------------------


def test_score_response_safety_harmful_lower_than_safe():
    safe = "This is a helpful and informative response."
    harmful = "You should bomb and attack and kill with toxic weapons to harm people."
    safe_score = score_response_safety(safe)
    harmful_score = score_response_safety(harmful)
    assert harmful_score < safe_score


# ---------------------------------------------------------------------------
# 13. CAIDataCollector.collect returns list of dicts
# ---------------------------------------------------------------------------


def test_cai_data_collector_collect_returns_list_of_dicts():
    cfg = CAIConfig(n_revision_steps=1)
    reviser = CAIReviser(mock_generate, cfg)
    collector = CAIDataCollector(reviser)
    results = collector.collect(["Response A.", "Response B."])
    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        assert isinstance(item, dict)
        assert "original" in item
        assert "final" in item
        assert "n_revisions" in item


# ---------------------------------------------------------------------------
# 14. get_stats returns correct keys
# ---------------------------------------------------------------------------


def test_get_stats_returns_correct_keys():
    cfg = CAIConfig(n_revision_steps=1)
    reviser = CAIReviser(mock_generate, cfg)
    collector = CAIDataCollector(reviser)
    collected = collector.collect(["Response A.", "Response B."])
    stats = collector.get_stats(collected)
    assert isinstance(stats, dict)
    assert "n_samples" in stats
    assert "mean_revisions" in stats
    assert "mean_safety_improvement" in stats


# ---------------------------------------------------------------------------
# 15. collect handles empty input
# ---------------------------------------------------------------------------


def test_collect_handles_empty_input():
    cfg = CAIConfig(n_revision_steps=1)
    reviser = CAIReviser(mock_generate, cfg)
    collector = CAIDataCollector(reviser)
    results = collector.collect([])
    assert results == []
    stats = collector.get_stats(results)
    assert stats["n_samples"] == 0.0
    assert stats["mean_revisions"] == 0.0
    assert stats["mean_safety_improvement"] == 0.0
