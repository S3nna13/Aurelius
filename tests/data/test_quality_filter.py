"""Tests for src/data/quality_filter.py.

16 unit tests + 1 integration test covering QualityFilterConfig, QualityFilter
metrics, filter logic, batch processing, and aggregate statistics.
"""

from __future__ import annotations

import math

import pytest

from src.data.quality_filter import (
    FilterResult,
    QualityFilter,
    QualityFilterConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_filter() -> QualityFilter:
    return QualityFilter(QualityFilterConfig())


# Good quality English paragraph — passes all default checks.
GOOD_PARAGRAPH = (
    "Scientists at the university recently published groundbreaking research "
    "on quantum entanglement that could revolutionize secure communications. "
    "The study involved years of careful experimentation and peer review before "
    "reaching these remarkable and thoroughly validated conclusions about physics."
)

# Repeated single word — very low entropy and repetitive.
REPETITIVE_TEXT = "the " * 200  # 800 chars, but very low entropy and low unique ngram ratio


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = QualityFilterConfig()
    assert cfg.min_chars == 100
    assert cfg.max_chars == 100_000
    assert cfg.min_words == 20
    assert cfg.min_char_entropy == pytest.approx(3.5)
    assert cfg.max_char_entropy == pytest.approx(6.5)
    assert cfg.ngram_n == 5
    assert cfg.min_unique_ngram_ratio == pytest.approx(0.2)
    assert cfg.min_alpha_ratio == pytest.approx(0.5)
    assert cfg.max_bullet_ratio == pytest.approx(0.9)
    assert cfg.max_ellipsis_ratio == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 2. test_char_entropy_uniform — uniform char distribution → high entropy
# ---------------------------------------------------------------------------

def test_char_entropy_uniform(default_filter: QualityFilter):
    # 256 distinct characters each appearing exactly once → max entropy = log2(256) = 8
    text = "".join(chr(i) for i in range(32, 288))  # 256 printable-ish chars
    entropy = default_filter.char_entropy(text)
    expected = math.log2(256)
    assert entropy == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. test_char_entropy_repetitive — single char repeated → near-0 entropy
# ---------------------------------------------------------------------------

def test_char_entropy_repetitive(default_filter: QualityFilter):
    text = "a" * 1000
    entropy = default_filter.char_entropy(text)
    assert entropy == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. test_unique_ngram_ratio_all_unique — no repeated ngrams → ratio = 1.0
# ---------------------------------------------------------------------------

def test_unique_ngram_ratio_all_unique(default_filter: QualityFilter):
    # 30 distinct words — with n=5 most 5-grams slide one word at a time,
    # all unique because no word repeats at the same position window.
    words = [f"word{i}" for i in range(30)]
    text = " ".join(words)
    ratio = default_filter.unique_ngram_ratio(text)
    assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. test_unique_ngram_ratio_all_same — same ngram repeated → low ratio
# ---------------------------------------------------------------------------

def test_unique_ngram_ratio_all_same(default_filter: QualityFilter):
    # "a b c d e" repeated many times → every 5-gram is identical
    text = "a b c d e " * 50
    ratio = default_filter.unique_ngram_ratio(text)
    # Only 1 unique 5-gram out of many total → very low ratio
    assert ratio < 0.05


# ---------------------------------------------------------------------------
# 6. test_alpha_ratio_all_letters — pure alpha text → 1.0
# ---------------------------------------------------------------------------

def test_alpha_ratio_all_letters(default_filter: QualityFilter):
    ratio = default_filter.alpha_ratio("abcdefghijklmnopqrstuvwxyz")
    assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. test_alpha_ratio_mixed — numbers and letters
# ---------------------------------------------------------------------------

def test_alpha_ratio_mixed(default_filter: QualityFilter):
    # "aaa111" → 3 alpha / 6 total = 0.5
    ratio = default_filter.alpha_ratio("aaa111")
    assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 8. test_bullet_ratio_none — no bullets → 0.0
# ---------------------------------------------------------------------------

def test_bullet_ratio_none(default_filter: QualityFilter):
    text = "This is line one.\nThis is line two.\nThis is line three."
    ratio = default_filter.bullet_ratio(text)
    assert ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. test_bullet_ratio_all — all bullet lines → 1.0
# ---------------------------------------------------------------------------

def test_bullet_ratio_all(default_filter: QualityFilter):
    text = "- item one\n* item two\n• item three\n1. item four"
    ratio = default_filter.bullet_ratio(text)
    assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. test_filter_passes_good_text — normal paragraph passes all checks
# ---------------------------------------------------------------------------

def test_filter_passes_good_text(default_filter: QualityFilter):
    result = default_filter.filter(GOOD_PARAGRAPH)
    assert isinstance(result, FilterResult)
    assert result.passed, f"Expected good text to pass. Reasons: {result.reasons}"
    assert result.reasons == []
    # Stats should be populated
    assert "char_entropy" in result.stats
    assert "unique_ngram_ratio" in result.stats
    assert "alpha_ratio" in result.stats


# ---------------------------------------------------------------------------
# 11. test_filter_fails_short — too few chars/words → fails length check
# ---------------------------------------------------------------------------

def test_filter_fails_short(default_filter: QualityFilter):
    result = default_filter.filter("Hi.")
    assert not result.passed
    # Should have at least a chars and words failure reason
    reasons_str = " ".join(result.reasons)
    assert "too_short_chars" in reasons_str or "too_few_words" in reasons_str


# ---------------------------------------------------------------------------
# 12. test_filter_fails_low_entropy — "aaa..." repeated → fails entropy check
# ---------------------------------------------------------------------------

def test_filter_fails_low_entropy():
    # Use very relaxed length/word thresholds so only entropy causes failure.
    cfg = QualityFilterConfig(
        min_chars=1,
        min_words=1,
        min_char_entropy=3.5,
        max_char_entropy=6.5,
        min_unique_ngram_ratio=0.0,
        min_alpha_ratio=0.0,
        max_bullet_ratio=1.0,
        max_ellipsis_ratio=1.0,
    )
    qf = QualityFilter(cfg)
    low_entropy_text = "a" * 500
    result = qf.filter(low_entropy_text)
    assert not result.passed
    reasons_str = " ".join(result.reasons)
    assert "low_entropy" in reasons_str


# ---------------------------------------------------------------------------
# 13. test_filter_fails_low_alpha — all-digit text → fails alpha ratio
# ---------------------------------------------------------------------------

def test_filter_fails_low_alpha():
    cfg = QualityFilterConfig(
        min_chars=1,
        min_words=1,
        min_char_entropy=0.0,
        max_char_entropy=100.0,
        min_unique_ngram_ratio=0.0,
        min_alpha_ratio=0.5,
        max_bullet_ratio=1.0,
        max_ellipsis_ratio=1.0,
    )
    qf = QualityFilter(cfg)
    # 200 digits — alpha ratio = 0
    digit_text = "1234567890 " * 20
    result = qf.filter(digit_text)
    assert not result.passed
    reasons_str = " ".join(result.reasons)
    assert "low_alpha_ratio" in reasons_str


# ---------------------------------------------------------------------------
# 14. test_filter_batch_length — returns same length as input
# ---------------------------------------------------------------------------

def test_filter_batch_length(default_filter: QualityFilter):
    texts = [GOOD_PARAGRAPH, "short", REPETITIVE_TEXT, ""]
    results = default_filter.filter_batch(texts)
    assert len(results) == len(texts)
    assert all(isinstance(r, FilterResult) for r in results)


# ---------------------------------------------------------------------------
# 15. test_statistics_pass_rate — pass_rate in [0, 1]
# ---------------------------------------------------------------------------

def test_statistics_pass_rate(default_filter: QualityFilter):
    texts = [GOOD_PARAGRAPH, "x", REPETITIVE_TEXT]
    results = default_filter.filter_batch(texts)
    stats = default_filter.statistics(results)

    assert "pass_rate" in stats
    assert "mean_char_entropy" in stats
    assert "mean_unique_ngram_ratio" in stats
    assert "mean_alpha_ratio" in stats
    assert 0.0 <= stats["pass_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 16. test_statistics_empty — empty list returns zeroed stats
# ---------------------------------------------------------------------------

def test_statistics_empty(default_filter: QualityFilter):
    stats = default_filter.statistics([])
    assert stats["pass_rate"] == pytest.approx(0.0)
    assert stats["mean_char_entropy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: mix of 5 good + 5 bad texts
# ---------------------------------------------------------------------------

def test_integration_mixed_batch():
    """Integration test: 5 good + 5 bad texts; pass_rate strictly between 0 and 1."""
    cfg = QualityFilterConfig(
        min_chars=100,
        min_words=20,
        min_char_entropy=3.5,
        max_char_entropy=6.5,
        ngram_n=5,
        min_unique_ngram_ratio=0.2,
        min_alpha_ratio=0.5,
        max_bullet_ratio=0.9,
        max_ellipsis_ratio=0.1,
    )
    qf = QualityFilter(cfg)

    good_texts = [
        (
            "Researchers at the institute published a landmark study on renewable energy "
            "that proposed new methods for storing solar power in large-scale battery arrays. "
            "The findings were validated by three independent laboratories across two continents."
        ),
        (
            "The history of ancient Rome spans many centuries and includes the rise of the "
            "Republic, the conquests of Julius Caesar, the reign of Augustus, and the eventual "
            "fall of the Western Empire in the fifth century of the common era."
        ),
        (
            "Machine learning algorithms have transformed how computers recognize speech, "
            "translate languages, and generate creative content. These advances rely on large "
            "neural networks trained on vast datasets collected from across the internet."
        ),
        (
            "Climate change poses significant risks to global food security, sea levels, and "
            "biodiversity. Scientists urge immediate reductions in carbon emissions to limit "
            "warming to manageable levels and prevent the worst projected impacts on ecosystems."
        ),
        (
            "The novel explores themes of identity, belonging, and cultural displacement through "
            "the eyes of a young immigrant navigating life in an unfamiliar city. Critics praised "
            "its vivid prose and nuanced characterization of communities often overlooked in literature."
        ),
    ]

    bad_texts = [
        "Hi.",                           # too short
        "1" * 500,                       # all digits, zero alpha ratio
        "a" * 500,                       # low entropy (single char)
        "a b c d e " * 100,             # repetitive n-grams
        "x",                             # far too short
    ]

    all_texts = good_texts + bad_texts
    results = qf.filter_batch(all_texts)

    assert len(results) == 10

    stats = qf.statistics(results)

    # At least some good texts pass and at least some bad texts fail
    assert stats["pass_rate"] > 0.0, "Expected some texts to pass"
    assert stats["pass_rate"] < 1.0, "Expected some texts to fail"

    # All required stat keys are present and finite
    for key in ("pass_rate", "mean_char_entropy", "mean_unique_ngram_ratio", "mean_alpha_ratio"):
        assert key in stats
        import math as _math
        assert _math.isfinite(stats[key]), f"{key} is not finite: {stats[key]}"

    # Sanity: every result has stats populated
    for r in results:
        assert "char_entropy" in r.stats
        assert "alpha_ratio" in r.stats
