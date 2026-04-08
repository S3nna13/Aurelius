"""Tests for src/data/quality_filter.py"""

import pytest

from src.data.quality_filter import (
    DeduplicationPipeline,
    FilterResult,
    MinHashSimilarity,
    QualityFilterConfig,
    TextQualityFilter,
    compute_ngram_hashes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_good_text(n_sentences: int = 10) -> str:
    """Return a plausible English paragraph that should pass all filters."""
    sentence = (
        "The quick brown fox jumps over the lazy dog near the riverbank "
        "on a sunny afternoon while birds chirp softly in the trees."
    )
    return " ".join([sentence] * n_sentences)


# ---------------------------------------------------------------------------
# TextQualityFilter tests
# ---------------------------------------------------------------------------

def test_filter_short_text_rejected():
    """Text with fewer than min_words should be rejected."""
    cfg = QualityFilterConfig(min_words=50, min_chars=5)
    f = TextQualityFilter(cfg)
    short_text = "This is a very short text."
    result = f.filter(short_text)
    assert not result.passed
    assert any("too_few_words" in r for r in result.reasons)


def test_filter_good_text_passes():
    """A normal English paragraph should pass all quality checks."""
    f = TextQualityFilter()
    text = _make_good_text(10)
    result = f.filter(text)
    assert result.passed, f"Expected pass, got reasons: {result.reasons}"


def test_filter_high_repetition_rejected():
    """Text that is just the same word repeated should be rejected."""
    f = TextQualityFilter()
    # "the " repeated enough to pass char/word length but fail repetition
    text = ("the " * 200).strip()
    result = f.filter(text)
    assert not result.passed
    assert any("high_word_repeat_ratio" in r or "low_unique_word_ratio" in r for r in result.reasons)


def test_filter_symbol_heavy_rejected():
    """Text with a very high symbol-to-word ratio should be rejected."""
    cfg = QualityFilterConfig(
        min_chars=10,
        min_words=5,
        max_symbol_to_word_ratio=0.1,
    )
    f = TextQualityFilter(cfg)
    # Construct text: a handful of words interspersed with many symbols
    words = "hello world foo bar baz"
    symbols = "!@#$%^&*()!@#$%^&*()!@#$%^&*()" * 10
    # Put symbols as part of the text but keep whitespace so word count stays low
    text = words + " " + " ".join(symbols)
    result = f.filter(text)
    assert not result.passed
    assert any("high_symbol_to_word_ratio" in r for r in result.reasons)


def test_filter_batch_returns_all():
    """filter_batch should return one FilterResult per input text."""
    f = TextQualityFilter()
    texts = [
        _make_good_text(10),
        "short",
        _make_good_text(8),
        "tiny text here",
        _make_good_text(12),
    ]
    results = f.filter_batch(texts)
    assert len(results) == 5
    assert all(isinstance(r, FilterResult) for r in results)


def test_stats_pass_rate():
    """A corpus of all-good texts should have a pass_rate close to 1.0."""
    f = TextQualityFilter()
    texts = [_make_good_text(10) for _ in range(20)]
    s = f.stats(texts)
    assert s["n_total"] == 20
    assert s["pass_rate"] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# compute_ngram_hashes tests
# ---------------------------------------------------------------------------

def test_compute_ngram_hashes_count():
    """For a 10-word text with n=5, there should be exactly 6 hashes."""
    text = "one two three four five six seven eight nine ten"
    hashes = compute_ngram_hashes(text, n=5)
    # words: 10, n-grams: 10 - 5 + 1 = 6
    assert len(hashes) == 6


# ---------------------------------------------------------------------------
# MinHashSimilarity tests
# ---------------------------------------------------------------------------

def test_minhash_signature_length():
    """Signature should have exactly n_hashes elements."""
    mh = MinHashSimilarity(n_hashes=64)
    sig = mh.signature(_make_good_text(5))
    assert len(sig) == 64


def test_minhash_identical_texts():
    """jaccard_estimate of a signature with itself should be 1.0."""
    mh = MinHashSimilarity(n_hashes=128)
    text = _make_good_text(5)
    sig = mh.signature(text)
    assert mh.jaccard_estimate(sig, sig) == pytest.approx(1.0)


def test_minhash_disjoint_texts():
    """Two completely different texts should have a low Jaccard estimate."""
    mh = MinHashSimilarity(n_hashes=256)
    text_a = " ".join(["alpha"] * 100)
    text_b = " ".join(["zeta"] * 100)
    sig_a = mh.signature(text_a)
    sig_b = mh.signature(text_b)
    j = mh.jaccard_estimate(sig_a, sig_b)
    assert j < 0.3, f"Expected low Jaccard for disjoint texts, got {j}"


# ---------------------------------------------------------------------------
# DeduplicationPipeline tests
# ---------------------------------------------------------------------------

def test_deduplication_removes_duplicates():
    """A corpus where one document is an exact duplicate should lose that doc."""
    unique_texts = [_make_good_text(i + 5) for i in range(4)]
    # Append an exact copy of the first document
    texts = unique_texts + [unique_texts[0]]
    pipeline = DeduplicationPipeline()
    deduped, kept_indices = pipeline.deduplicate(texts)
    assert len(deduped) == len(unique_texts), (
        f"Expected {len(unique_texts)} docs after dedup, got {len(deduped)}"
    )
    # The duplicate (last index) should not be in kept_indices
    assert len(texts) - 1 not in kept_indices


def test_deduplication_keeps_unique():
    """All-unique texts should all be kept."""
    texts = [
        "The cat sat on the mat and looked around the quiet room curiously.",
        "A programmer wrote efficient code late into the dark winter night.",
        "Mountains rise above the clouds where eagles soar through empty air.",
        "Scientists discovered a new species deep beneath the ocean floor today.",
        "Children laughed and played in the bright sunlit park all afternoon.",
    ]
    # Make each text long enough to have distinct n-gram profiles
    texts = [t * 20 for t in texts]
    pipeline = DeduplicationPipeline()
    deduped, kept_indices = pipeline.deduplicate(texts)
    assert len(deduped) == len(texts), (
        f"Expected all {len(texts)} unique docs kept, got {len(deduped)}"
    )
