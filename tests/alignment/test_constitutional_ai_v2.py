"""Tests for src/alignment/constitutional_ai_v2.py.

Uses tiny configs so every test runs in milliseconds with no external deps.
"""

from __future__ import annotations

from src.alignment.constitutional_ai_v2 import (
    CAIConfig,
    CAISession,
    Principle,
    compute_principle_coverage,
    format_critique_prompt,
    format_revision_prompt,
    score_principle_adherence,
    select_worst_principle,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TINY_PRINCIPLES = ["Be helpful", "Be harmless", "Be honest"]


def _tiny_config(**kwargs) -> CAIConfig:
    """Return a CAIConfig with minimal revisions for fast tests."""
    defaults = dict(principles=TINY_PRINCIPLES, n_revisions=2)
    defaults.update(kwargs)
    return CAIConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. CAIConfig defaults
# ---------------------------------------------------------------------------


def test_caiconfig_defaults():
    cfg = CAIConfig()
    assert cfg.principles == ["Be helpful", "Be harmless", "Be honest"]
    assert cfg.n_revisions == 2
    assert cfg.critique_prefix == "Critique:"
    assert cfg.revision_prefix == "Revision:"
    assert cfg.max_critique_len == 200
    assert cfg.max_revision_len == 500


# ---------------------------------------------------------------------------
# 2. Principle dataclass fields
# ---------------------------------------------------------------------------


def test_principle_defaults():
    p = Principle(text="Do no harm")
    assert p.text == "Do no harm"
    assert p.weight == 1.0
    assert p.category == "general"


def test_principle_custom_fields():
    p = Principle(text="Be concise", weight=0.5, category="style")
    assert p.weight == 0.5
    assert p.category == "style"


# ---------------------------------------------------------------------------
# 3. format_critique_prompt contains response and principle
# ---------------------------------------------------------------------------


def test_format_critique_prompt_contains_response_and_principle():
    cfg = _tiny_config()
    response = "This is my answer."
    principle = "Be helpful"
    prompt = format_critique_prompt(response, principle, cfg)
    assert response in prompt
    assert principle in prompt


def test_format_critique_prompt_contains_critique_prefix():
    cfg = _tiny_config()
    prompt = format_critique_prompt("answer", "Be honest", cfg)
    assert cfg.critique_prefix in prompt


# ---------------------------------------------------------------------------
# 4. format_revision_prompt contains critique
# ---------------------------------------------------------------------------


def test_format_revision_prompt_contains_critique():
    cfg = _tiny_config()
    critique_text = "The answer lacks detail."
    prompt = format_revision_prompt("original", critique_text, "Be helpful", cfg)
    assert critique_text in prompt


def test_format_revision_prompt_contains_response_and_principle():
    cfg = _tiny_config()
    response = "Short answer."
    principle = "Be harmless"
    prompt = format_revision_prompt(response, "some critique", principle, cfg)
    assert response in prompt
    assert principle in prompt


# ---------------------------------------------------------------------------
# 5. score_principle_adherence in [0, 1]
# ---------------------------------------------------------------------------


def test_score_principle_adherence_range():
    score = score_principle_adherence("I will help you safely.", "Be helpful and harmless")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 6. score_principle_adherence perfect match = high score
# ---------------------------------------------------------------------------


def test_score_principle_adherence_high_on_match():
    # Response contains all key words of the principle.
    principle = "helpful harmless honest"
    response = "I am helpful and harmless and honest in everything I do."
    score = score_principle_adherence(response, principle)
    assert score > 0.5


# ---------------------------------------------------------------------------
# 7. score_principle_adherence empty response = 0
# ---------------------------------------------------------------------------


def test_score_principle_adherence_empty_response():
    score = score_principle_adherence("", "Be helpful and harmless")
    assert score == 0.0


# ---------------------------------------------------------------------------
# 8. select_worst_principle returns valid index
# ---------------------------------------------------------------------------


def test_select_worst_principle_valid_index():
    response = "I will try to assist."
    idx, score = select_worst_principle(response, TINY_PRINCIPLES)
    assert 0 <= idx < len(TINY_PRINCIPLES)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 9. select_worst_principle returns min score
# ---------------------------------------------------------------------------


def test_select_worst_principle_returns_min_score():
    response = "I will try to assist."
    principles = TINY_PRINCIPLES
    idx, score = select_worst_principle(response, principles)
    all_scores = [score_principle_adherence(response, p) for p in principles]
    assert score == min(all_scores)


# ---------------------------------------------------------------------------
# 10. CAISession.critique returns non-empty string
# ---------------------------------------------------------------------------


def test_caisession_critique_nonempty():
    session = CAISession(_tiny_config())
    result = session.critique("Some response.", "Be helpful")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 11. CAISession.revise returns non-empty string
# ---------------------------------------------------------------------------


def test_caisession_revise_nonempty():
    session = CAISession(_tiny_config())
    result = session.revise("Some response.", "It lacks detail.", "Be helpful")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 12. CAISession.run_revision_loop length = n_revisions
# ---------------------------------------------------------------------------


def test_run_revision_loop_length():
    cfg = _tiny_config(n_revisions=3)
    session = CAISession(cfg)
    history = session.run_revision_loop("Initial answer.")
    assert len(history) == 3


# ---------------------------------------------------------------------------
# 13. CAISession.run_revision_loop dict has required keys
# ---------------------------------------------------------------------------


def test_run_revision_loop_dict_keys():
    session = CAISession(_tiny_config(n_revisions=1))
    history = session.run_revision_loop("Initial answer.")
    required_keys = {
        "iteration",
        "principle",
        "critique",
        "revision",
        "score_before",
        "score_after",
    }
    assert required_keys.issubset(history[0].keys())


# ---------------------------------------------------------------------------
# 14. CAISession.get_final_response returns string
# ---------------------------------------------------------------------------


def test_get_final_response_returns_string():
    session = CAISession(_tiny_config())
    history = session.run_revision_loop("Initial answer.")
    final = session.get_final_response(history)
    assert isinstance(final, str)
    assert len(final) > 0


def test_get_final_response_matches_last_revision():
    session = CAISession(_tiny_config(n_revisions=2))
    history = session.run_revision_loop("Initial answer.")
    final = session.get_final_response(history)
    assert final == history[-1]["revision"]


# ---------------------------------------------------------------------------
# 15. compute_principle_coverage has coverage and mean_improvement keys
# ---------------------------------------------------------------------------


def test_compute_principle_coverage_keys():
    session = CAISession(_tiny_config())
    history = session.run_revision_loop("Initial answer.")
    result = compute_principle_coverage(history, TINY_PRINCIPLES)
    assert "coverage" in result
    assert "mean_improvement" in result


def test_compute_principle_coverage_range():
    session = CAISession(_tiny_config(n_revisions=3))
    history = session.run_revision_loop("Initial answer.")
    result = compute_principle_coverage(history, TINY_PRINCIPLES)
    assert 0.0 <= result["coverage"] <= 1.0


def test_compute_principle_coverage_empty_history():
    result = compute_principle_coverage([], TINY_PRINCIPLES)
    assert result["coverage"] == 0.0
    assert result["mean_improvement"] == 0.0
