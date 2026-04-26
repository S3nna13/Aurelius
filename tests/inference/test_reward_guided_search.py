"""Unit tests for src/inference/reward_guided_search.py.

Tests cover SearchConfig defaults, score_candidate logic (combined /
length-penalty), expand_beams behaviour, search termination, best_sequence
selection, diversity_score computation, and registry wiring.

Tiny vocab_size=256 is used throughout for speed.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.reward_guided_search import (
    RewardGuidedSearch,
    SearchBeam,
    SearchConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cfg(**kwargs) -> SearchConfig:
    defaults = dict(beam_width=4, max_steps=8, vocab_size=256)
    defaults.update(kwargs)
    return SearchConfig(**defaults)


def constant_value_fn(v: float):
    """Return a value_fn that always returns the same constant."""

    def fn(tokens):
        return v

    return fn


def random_logits_model_fn(vocab_size: int = 256, seed: int = 0):
    """Return a model_fn that produces deterministic random logits."""

    def fn(tokens):
        g = torch.Generator()
        g.manual_seed(seed + len(tokens))
        return torch.randn(vocab_size, generator=g)

    return fn


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SearchConfig()
    assert cfg.beam_width == 4
    assert cfg.max_steps == 64
    assert cfg.reward_weight == 0.5
    assert cfg.length_penalty == 0.6
    assert cfg.vocab_size == 128000
    assert cfg.eos_token_id == 2
    assert cfg.pad_token_id == 0


# ---------------------------------------------------------------------------
# 2. test_score_candidate_combined — λ=0.5: score = 0.5*lp + 0.5*v / len^α
# ---------------------------------------------------------------------------


def test_score_candidate_combined():
    cfg = make_cfg(reward_weight=0.5, length_penalty=0.6)
    searcher = RewardGuidedSearch(cfg)
    lp, v, length = -2.0, 1.0, 4
    sc = searcher.score_candidate(lp, v, length)
    combined = 0.5 * lp + 0.5 * v
    expected = combined / (length**0.6)
    assert sc == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. test_score_candidate_pure_logprob — λ=0: score based on log_prob only
# ---------------------------------------------------------------------------


def test_score_candidate_pure_logprob():
    cfg = make_cfg(reward_weight=0.0, length_penalty=0.6)
    searcher = RewardGuidedSearch(cfg)
    lp, v, length = -3.0, 5.0, 2
    sc = searcher.score_candidate(lp, v, length)
    expected = lp / (length**0.6)
    assert sc == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. test_score_candidate_pure_value — λ=1: score based on value only
# ---------------------------------------------------------------------------


def test_score_candidate_pure_value():
    cfg = make_cfg(reward_weight=1.0, length_penalty=0.6)
    searcher = RewardGuidedSearch(cfg)
    lp, v, length = -10.0, 3.0, 3
    sc = searcher.score_candidate(lp, v, length)
    expected = v / (length**0.6)
    assert sc == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# 5. test_score_candidate_length_penalty — longer seq → lower score
# ---------------------------------------------------------------------------


def test_score_candidate_length_penalty():
    cfg = make_cfg(reward_weight=0.5, length_penalty=0.6)
    searcher = RewardGuidedSearch(cfg)
    lp, v = 0.0, 1.0  # combined = 0.5 > 0, so longer len → lower score
    sc_short = searcher.score_candidate(lp, v, 2)
    sc_long = searcher.score_candidate(lp, v, 10)
    assert sc_short > sc_long, f"Shorter sequence should score higher: {sc_short} vs {sc_long}"


# ---------------------------------------------------------------------------
# 6. test_expand_beams_count — beam_width=2 → returns exactly 2 beams
# ---------------------------------------------------------------------------


def test_expand_beams_count():
    cfg = make_cfg(beam_width=2, vocab_size=256)
    searcher = RewardGuidedSearch(cfg)
    beams = [
        SearchBeam(token_ids=[1, 2], log_prob_sum=-1.0, value_sum=0.5),
        SearchBeam(token_ids=[1, 3], log_prob_sum=-1.5, value_sum=0.4),
    ]
    logits = torch.randn(2, 256)
    new_beams = searcher.expand_beams(beams, logits, constant_value_fn(0.5))
    assert len(new_beams) == 2


# ---------------------------------------------------------------------------
# 7. test_expand_beams_best_first — returned beams sorted by score desc
# ---------------------------------------------------------------------------


def test_expand_beams_best_first():
    cfg = make_cfg(beam_width=4, vocab_size=256)
    searcher = RewardGuidedSearch(cfg)
    beams = [SearchBeam(token_ids=[1], log_prob_sum=0.0, value_sum=0.0)]
    logits = torch.randn(1, 256)
    new_beams = searcher.expand_beams(beams, logits, constant_value_fn(0.3))
    scores = [b.score for b in new_beams]
    assert scores == sorted(scores, reverse=True), f"Beams not sorted by score desc: {scores}"


# ---------------------------------------------------------------------------
# 8. test_expand_marks_eos — token==eos_token_id → beam.is_finished=True
# ---------------------------------------------------------------------------


def test_expand_marks_eos():
    cfg = make_cfg(beam_width=4, vocab_size=256, eos_token_id=2)
    searcher = RewardGuidedSearch(cfg)

    # Build logits that put huge mass on EOS token (id=2) so it is always
    # selected as the top-1 expansion of this beam.
    logits = torch.full((1, 256), -1e9)
    logits[0, 2] = 1e9  # token 2 = EOS

    beams = [SearchBeam(token_ids=[10, 11], log_prob_sum=0.0, value_sum=0.0)]
    new_beams = searcher.expand_beams(beams, logits, constant_value_fn(0.0))

    # At least one beam must be marked finished (the EOS expansion).
    finished = [b for b in new_beams if b.is_finished]
    assert len(finished) >= 1, "Expected at least one finished beam after EOS expansion"
    for b in finished:
        assert b.token_ids[-1] == cfg.eos_token_id


# ---------------------------------------------------------------------------
# 9. test_search_returns_beams — search returns a list of SearchBeam
# ---------------------------------------------------------------------------


def test_search_returns_beams():
    cfg = make_cfg(beam_width=2, max_steps=3, vocab_size=256)
    searcher = RewardGuidedSearch(cfg)
    result = searcher.search(
        initial_tokens=[1, 2, 3],
        model_fn=random_logits_model_fn(256),
        value_fn=constant_value_fn(0.5),
    )
    assert isinstance(result, list), "search() must return a list"
    assert len(result) > 0, "search() returned an empty list"
    for b in result:
        assert isinstance(b, SearchBeam), f"Expected SearchBeam, got {type(b)}"


# ---------------------------------------------------------------------------
# 10. test_search_max_steps — stops after at most max_steps expansions
# ---------------------------------------------------------------------------


def test_search_max_steps():
    max_steps = 5
    cfg = make_cfg(beam_width=2, max_steps=max_steps, vocab_size=256)
    searcher = RewardGuidedSearch(cfg)

    call_count = {"n": 0}

    def counting_model_fn(tokens):
        call_count["n"] += 1
        return torch.randn(256)

    searcher.search(
        initial_tokens=[1],
        model_fn=counting_model_fn,
        value_fn=constant_value_fn(0.0),
    )
    # Each step calls model_fn once per active beam (2 beams, 5 steps = ≤10 calls).
    assert call_count["n"] <= max_steps * cfg.beam_width, (
        f"model_fn called {call_count['n']} times, expected <= {max_steps * cfg.beam_width}"
    )


# ---------------------------------------------------------------------------
# 11. test_best_sequence_finished — returns highest-scored finished beam
# ---------------------------------------------------------------------------


def test_best_sequence_finished():
    beams = [
        SearchBeam(
            token_ids=[1, 2, 3], log_prob_sum=-1.0, value_sum=1.0, score=0.8, is_finished=True
        ),
        SearchBeam(
            token_ids=[1, 4, 5], log_prob_sum=-0.5, value_sum=0.5, score=0.9, is_finished=True
        ),
        SearchBeam(
            token_ids=[1, 6, 7, 8, 9],
            log_prob_sum=-0.1,
            value_sum=0.1,
            score=1.5,
            is_finished=False,
        ),  # best score but NOT finished
    ]
    cfg = SearchConfig()
    searcher = RewardGuidedSearch(cfg)
    result = searcher.best_sequence(beams)
    # Should return the finished beam with score 0.9 → token_ids [1,4,5]
    assert result == [1, 4, 5], f"Expected [1,4,5], got {result}"


# ---------------------------------------------------------------------------
# 12. test_best_sequence_no_finished — returns longest active beam
# ---------------------------------------------------------------------------


def test_best_sequence_no_finished():
    beams = [
        SearchBeam(
            token_ids=[1, 2], log_prob_sum=-1.0, value_sum=0.5, score=0.9, is_finished=False
        ),
        SearchBeam(
            token_ids=[1, 2, 3, 4, 5],
            log_prob_sum=-0.5,
            value_sum=0.3,
            score=0.3,
            is_finished=False,
        ),  # longest
        SearchBeam(
            token_ids=[1, 2, 3], log_prob_sum=-0.8, value_sum=0.4, score=0.5, is_finished=False
        ),
    ]
    cfg = SearchConfig()
    searcher = RewardGuidedSearch(cfg)
    result = searcher.best_sequence(beams)
    assert result == [1, 2, 3, 4, 5], f"Expected longest beam, got {result}"


# ---------------------------------------------------------------------------
# 13. test_diversity_score_identical — same sequence → 0.0
# ---------------------------------------------------------------------------


def test_diversity_score_identical():
    beams = [
        SearchBeam(token_ids=[1, 2, 3], log_prob_sum=0.0, value_sum=0.0),
        SearchBeam(token_ids=[1, 2, 3], log_prob_sum=0.0, value_sum=0.0),
        SearchBeam(token_ids=[1, 2, 3], log_prob_sum=0.0, value_sum=0.0),
    ]
    cfg = SearchConfig()
    searcher = RewardGuidedSearch(cfg)
    d = searcher.diversity_score(beams)
    assert d == pytest.approx(0.0, abs=1e-6), f"Expected 0.0, got {d}"


# ---------------------------------------------------------------------------
# 14. test_diversity_score_different — distinct sequences → > 0.0
# ---------------------------------------------------------------------------


def test_diversity_score_different():
    beams = [
        SearchBeam(token_ids=[1, 2, 3], log_prob_sum=0.0, value_sum=0.0),
        SearchBeam(token_ids=[4, 5, 6], log_prob_sum=0.0, value_sum=0.0),
        SearchBeam(token_ids=[7, 8, 9], log_prob_sum=0.0, value_sum=0.0),
    ]
    cfg = SearchConfig()
    searcher = RewardGuidedSearch(cfg)
    d = searcher.diversity_score(beams)
    assert d > 0.0, f"Expected > 0.0 for distinct sequences, got {d}"
    assert d <= 1.0, f"Diversity score must be in [0, 1], got {d}"


# ---------------------------------------------------------------------------
# 15. test_registry — DECODER_REGISTRY["reward_guided_search"] is RewardGuidedSearch
# ---------------------------------------------------------------------------


def test_registry():
    from src.inference import DECODER_REGISTRY

    assert "reward_guided_search" in DECODER_REGISTRY, (
        "'reward_guided_search' key missing from DECODER_REGISTRY"
    )
    assert DECODER_REGISTRY["reward_guided_search"] is RewardGuidedSearch, (
        "DECODER_REGISTRY['reward_guided_search'] is not RewardGuidedSearch"
    )
