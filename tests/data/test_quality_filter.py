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

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon.",
    "Scientists discovered that deep ocean trenches harbor extraordinary forms of microbial life.",
    "Modern programming languages balance expressiveness with performance in thoughtful ways.",
    "A fresh breeze carried the scent of pine needles across the mountain meadow at dawn.",
    "Historical records reveal surprising connections between seemingly unrelated civilisations.",
    "Astronomers mapped thousands of exoplanets orbiting distant stars within our galaxy.",
    "Renewable energy technology has advanced rapidly thanks to global research investment.",
    "Children who read widely tend to develop stronger critical thinking and empathy skills.",
    "Urban planners increasingly incorporate green spaces to improve residents' mental health.",
    "Classical music theorists debate the harmonic language of late Romantic composers.",
    "Geological surveys confirmed that ancient volcanic activity shaped this dramatic landscape.",
    "Machine learning models require careful evaluation to avoid perpetuating existing biases.",
    "The novelist wove themes of identity and belonging into each carefully crafted chapter.",
    "Athletes train systematically to improve endurance strength and technical precision together.",
    "Coastal communities adapt traditional fishing practices to cope with changing ocean conditions.",
    "Philosophers have long questioned whether free will is compatible with physical determinism.",
    "Software engineers refactor legacy codebases to improve maintainability and testability.",
    "Migratory birds navigate thousands of kilometres using magnetic fields and star patterns.",
    "Economic inequality shapes access to education healthcare and political representation.",
    "Botanists catalogued hundreds of previously undescribed flowering plant species in the rainforest.",
]


def _make_good_text(n_sentences: int = 10, offset: int = 0) -> str:
    """Return a plausible English paragraph that should pass all filters.

    Uses a rotating pool of distinct sentences to keep vocabulary varied,
    which prevents triggering word-repetition or unique-word-ratio filters.
    """
    chosen = [_SENTENCES[(offset + i) % len(_SENTENCES)] for i in range(n_sentences)]
    return " ".join(chosen)


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
    text = _make_good_text(n_sentences=15, offset=0)
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
    texts = [_make_good_text(n_sentences=15, offset=i) for i in range(20)]
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
    # Use offset so each block of sentences is drawn from a different part of the pool
    unique_texts = [_make_good_text(n_sentences=10, offset=i * 5) for i in range(4)]
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
