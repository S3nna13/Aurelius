"""Tests for src/data/qura_scorer.py (QuRating-style quality scorer).

Covers the 14-point rigor floor specified in the implementation spec.
Only stdlib + pure PyTorch — no scipy, sklearn, HuggingFace, etc.
"""

from __future__ import annotations

import math

import pytest
import torch

from aurelius.data.qura_scorer import (
    QuRaScorer,
    QuRatingConfig,
    QuRatingScores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokens(n: int, vocab_size: int = 128000, seed: int = 0) -> torch.Tensor:
    """Return deterministic (n,) int64 token-id tensor."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (n,), dtype=torch.long)


def _uniform_log_probs(n: int, value: float) -> torch.Tensor:
    """Return (n,) float tensor filled with *value*."""
    return torch.full((n,), value, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Shape: score_batch returns list of length B
# ---------------------------------------------------------------------------

def test_score_batch_returns_list_of_length_b():
    scorer = QuRaScorer()
    B, T = 4, 30
    ids = _make_tokens(B * T).view(B, T)
    lp = torch.full((B, T), -2.0)
    mask = torch.ones(B, T, dtype=torch.long)

    results = scorer.score_batch(ids, lp, mask)

    assert isinstance(results, list)
    assert len(results) == B


# ---------------------------------------------------------------------------
# 2. All individual scores in [0, 1]
# ---------------------------------------------------------------------------

def test_all_individual_scores_in_unit_interval():
    scorer = QuRaScorer()
    B, T = 6, 50
    ids = _make_tokens(B * T).view(B, T)
    lp = torch.randn(B, T) - 3.0  # random negative log-probs
    mask = torch.ones(B, T, dtype=torch.long)

    for s in scorer.score_batch(ids, lp, mask):
        assert 0.0 <= s.writing_quality <= 1.0
        assert 0.0 <= s.educational_value <= 1.0
        assert 0.0 <= s.facts_score <= 1.0
        assert 0.0 <= s.expertise_score <= 1.0
        assert 0.0 <= s.composite <= 1.0


# ---------------------------------------------------------------------------
# 3. Composite is the normalised weighted average
# ---------------------------------------------------------------------------

def test_composite_is_weighted_average():
    cfg = QuRatingConfig(
        writing_weight=0.35,
        educational_weight=0.30,
        facts_weight=0.20,
        expertise_weight=0.15,
    )
    scorer = QuRaScorer(cfg)
    ids = _make_tokens(40)
    lp = _uniform_log_probs(40, -2.0)

    s = scorer.score_tokens(ids, lp)

    total_w = 0.35 + 0.30 + 0.20 + 0.15
    expected = (
        0.35 * s.writing_quality
        + 0.30 * s.educational_value
        + 0.20 * s.facts_score
        + 0.15 * s.expertise_score
    ) / total_w
    assert math.isclose(s.composite, expected, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 4. Low-perplexity text → higher writing quality
# ---------------------------------------------------------------------------

def test_low_perplexity_yields_high_writing_quality():
    scorer = QuRaScorer()
    ids = _make_tokens(50)
    low_ppl_lp = _uniform_log_probs(50, -0.1)   # near 0 → ppl ≈ 1.1 → high quality
    high_ppl_lp = _uniform_log_probs(50, -5.0)  # very negative → higher ppl

    low_score = scorer.score_tokens(ids, low_ppl_lp)
    high_score = scorer.score_tokens(ids, high_ppl_lp)

    assert low_score.writing_quality > high_score.writing_quality


# ---------------------------------------------------------------------------
# 5. High-perplexity text → lower writing quality
# ---------------------------------------------------------------------------

def test_high_perplexity_yields_low_writing_quality():
    scorer = QuRaScorer()
    ids = _make_tokens(50)
    # log-prob so negative that ppl >= max_perplexity → writing_quality == 0
    extreme_lp = _uniform_log_probs(50, -100.0)

    s = scorer.score_tokens(ids, extreme_lp)

    assert s.writing_quality == 0.0


# ---------------------------------------------------------------------------
# 6. Edge case: short sequence (< min_tokens) → zero scores
# ---------------------------------------------------------------------------

def test_short_sequence_returns_zero_scores():
    cfg = QuRatingConfig(min_tokens=10)
    scorer = QuRaScorer(cfg)
    ids = _make_tokens(5)
    lp = _uniform_log_probs(5, -1.0)

    s = scorer.score_tokens(ids, lp)

    assert s.writing_quality == 0.0
    assert s.educational_value == 0.0
    assert s.facts_score == 0.0
    assert s.expertise_score == 0.0
    assert s.composite == 0.0


# ---------------------------------------------------------------------------
# 7. select_top_k returns exactly k indices
# ---------------------------------------------------------------------------

def test_select_top_k_returns_k_indices():
    scorer = QuRaScorer()
    B, T = 10, 30
    ids = _make_tokens(B * T).view(B, T)
    lp = torch.randn(B, T) - 2.0
    mask = torch.ones(B, T, dtype=torch.long)

    scores = scorer.score_batch(ids, lp, mask)
    top3 = scorer.select_top_k(scores, k=3)

    assert len(top3) == 3
    assert len(set(top3)) == 3  # distinct indices
    assert all(0 <= i < B for i in top3)


# ---------------------------------------------------------------------------
# 8. select_top_k(criterion='composite') returns highest composite scorers
# ---------------------------------------------------------------------------

def test_select_top_k_composite_returns_highest():
    scorer = QuRaScorer()
    B, T = 8, 30
    torch.manual_seed(42)
    ids = torch.randint(0, 128000, (B, T), dtype=torch.long)
    lp = torch.randn(B, T) - 2.0
    mask = torch.ones(B, T, dtype=torch.long)

    scores = scorer.score_batch(ids, lp, mask)
    k = 3
    top_k_indices = scorer.select_top_k(scores, k=k, criterion="composite")

    top_k_composites = sorted([scores[i].composite for i in top_k_indices], reverse=True)
    all_composites = sorted([s.composite for s in scores], reverse=True)

    assert top_k_composites == pytest.approx(all_composites[:k], abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Determinism: same inputs → same outputs
# ---------------------------------------------------------------------------

def test_determinism():
    scorer = QuRaScorer()
    ids = _make_tokens(40, seed=7)
    lp = _uniform_log_probs(40, -2.5)

    s1 = scorer.score_tokens(ids, lp)
    s2 = scorer.score_tokens(ids, lp)

    assert s1 == s2


# ---------------------------------------------------------------------------
# 10. Batch vs single: score_tokens and score_batch[0] agree
# ---------------------------------------------------------------------------

def test_batch_and_single_agree():
    scorer = QuRaScorer()
    T = 40
    ids = _make_tokens(T, seed=11)
    lp = _uniform_log_probs(T, -1.5)

    single = scorer.score_tokens(ids, lp)

    # Wrap in a batch of size 1, no padding
    ids_2d = ids.unsqueeze(0)
    lp_2d = lp.unsqueeze(0)
    mask = torch.ones(1, T, dtype=torch.long)
    batch_result = scorer.score_batch(ids_2d, lp_2d, mask)[0]

    assert single.writing_quality == pytest.approx(batch_result.writing_quality, abs=1e-6)
    assert single.educational_value == pytest.approx(batch_result.educational_value, abs=1e-6)
    assert single.facts_score == pytest.approx(batch_result.facts_score, abs=1e-6)
    assert single.expertise_score == pytest.approx(batch_result.expertise_score, abs=1e-6)
    assert single.composite == pytest.approx(batch_result.composite, abs=1e-6)


# ---------------------------------------------------------------------------
# 11. Numerical stability: no NaN/Inf with extreme log_probs
# ---------------------------------------------------------------------------

def test_numerical_stability_extreme_log_probs():
    scorer = QuRaScorer()
    ids = _make_tokens(50)

    for extreme_value in (-1e6, -1e4, -500.0, -0.0001, 0.0):
        lp = _uniform_log_probs(50, extreme_value)
        s = scorer.score_tokens(ids, lp)
        for field_val in s:
            assert not math.isnan(field_val), f"NaN for log_prob={extreme_value}"
            assert not math.isinf(field_val), f"Inf for log_prob={extreme_value}"


# ---------------------------------------------------------------------------
# 12. Padding mask: masked tokens do not affect score
# ---------------------------------------------------------------------------

def test_padding_mask_excluded_from_score():
    scorer = QuRaScorer()
    T_valid = 30
    T_pad = 20
    T_total = T_valid + T_pad

    torch.manual_seed(3)
    ids_valid = torch.randint(0, 128000, (T_valid,), dtype=torch.long)
    lp_valid = _uniform_log_probs(T_valid, -2.0)

    # Padding tokens are wildly different — should not change scores
    ids_padded = torch.cat([ids_valid, torch.zeros(T_pad, dtype=torch.long)])
    lp_padded = torch.cat([lp_valid, _uniform_log_probs(T_pad, -999.0)])
    mask = torch.cat([
        torch.ones(T_valid, dtype=torch.long),
        torch.zeros(T_pad, dtype=torch.long),
    ])

    # Score unpadded directly
    s_direct = scorer.score_tokens(ids_valid, lp_valid)

    # Score via batch with mask
    s_batch = scorer.score_batch(
        ids_padded.unsqueeze(0),
        lp_padded.unsqueeze(0),
        mask.unsqueeze(0),
    )[0]

    assert s_direct.writing_quality == pytest.approx(s_batch.writing_quality, abs=1e-6)
    assert s_direct.composite == pytest.approx(s_batch.composite, abs=1e-6)


# ---------------------------------------------------------------------------
# 13. Empty-ish batch: batch of size 1 works
# ---------------------------------------------------------------------------

def test_batch_of_size_one():
    scorer = QuRaScorer()
    T = 20
    ids = _make_tokens(T).unsqueeze(0)
    lp = _uniform_log_probs(T, -1.8).unsqueeze(0)
    mask = torch.ones(1, T, dtype=torch.long)

    results = scorer.score_batch(ids, lp, mask)

    assert len(results) == 1
    s = results[0]
    assert isinstance(s, QuRatingScores)
    assert 0.0 <= s.composite <= 1.0


# ---------------------------------------------------------------------------
# 14. Config weights: changing weights changes composite proportionally
# ---------------------------------------------------------------------------

def test_config_weights_affect_composite():
    """Doubling the writing weight should move the composite toward writing_quality."""
    ids = _make_tokens(50, seed=99)
    lp = _uniform_log_probs(50, -1.0)

    cfg_default = QuRatingConfig()
    cfg_high_writing = QuRatingConfig(
        writing_weight=0.70,
        educational_weight=0.10,
        facts_weight=0.10,
        expertise_weight=0.10,
    )

    s_default = QuRaScorer(cfg_default).score_tokens(ids, lp)
    s_high_w = QuRaScorer(cfg_high_writing).score_tokens(ids, lp)

    # With high writing weight, composite should be closer to writing_quality
    diff_high = abs(s_high_w.composite - s_high_w.writing_quality)
    diff_default = abs(s_default.composite - s_default.writing_quality)

    # Both scorers see the same document, so individual dimension scores are equal
    assert s_default.writing_quality == pytest.approx(s_high_w.writing_quality, abs=1e-6)
    # Composite differs because weights differ
    assert not math.isclose(s_default.composite, s_high_w.composite, abs_tol=1e-6) or \
           math.isclose(s_default.writing_quality, s_default.composite, abs_tol=1e-4)
    # Higher writing weight → composite pulled more toward writing_quality
    assert diff_high <= diff_default + 1e-6
