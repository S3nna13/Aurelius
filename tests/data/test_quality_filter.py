"""Tests for src/data/quality_filter.py — 12 tests covering FilterConfig,
heuristic functions, HeuristicFilter, MinHashSketch, DeduplicationIndex,
and DataQualityPipeline."""

import numpy as np
import pytest

from src.data.quality_filter import (
    DataQualityPipeline,
    DeduplicationIndex,
    FilterConfig,
    HeuristicFilter,
    MinHashSketch,
    compute_alpha_ratio,
    compute_repetition_ratio,
    compute_text_quality_score,
)


# ---------------------------------------------------------------------------
# 1. FilterConfig defaults
# ---------------------------------------------------------------------------

def test_filter_config_defaults():
    cfg = FilterConfig()
    assert cfg.min_length == 50
    assert cfg.max_length == 100000
    assert cfg.min_alpha_ratio == 0.6
    assert cfg.max_repetition_ratio == 0.3
    assert cfg.language == "en"
    assert cfg.dedup_threshold == 0.8
    assert cfg.n_minhash_perms == 64


# ---------------------------------------------------------------------------
# 2. compute_alpha_ratio — all alpha
# ---------------------------------------------------------------------------

def test_compute_alpha_ratio_all_alpha():
    result = compute_alpha_ratio("abcdefghij")
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. compute_alpha_ratio — no alpha
# ---------------------------------------------------------------------------

def test_compute_alpha_ratio_no_alpha():
    result = compute_alpha_ratio("1234567890!@#$")
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. compute_repetition_ratio — all unique n-grams → low ratio
# ---------------------------------------------------------------------------

def test_compute_repetition_ratio_unique():
    # Each consecutive 3-gram is unique in a simple ascending sequence
    words = " ".join(str(i) for i in range(20))
    ratio = compute_repetition_ratio(words, n=3)
    # All n-grams are unique, so repeated count is 0 → ratio == 0.0
    assert ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. compute_text_quality_score — in [0, 1]
# ---------------------------------------------------------------------------

def test_compute_text_quality_score_range():
    texts = [
        "Hello world. This is a sentence. Another one here.",
        "",
        "x" * 500,
        "The quick brown fox jumps over the lazy dog. " * 10,
    ]
    for t in texts:
        score = compute_text_quality_score(t)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for text: {t[:40]!r}"


# ---------------------------------------------------------------------------
# 6. HeuristicFilter — too short fails length check
# ---------------------------------------------------------------------------

def test_heuristic_filter_too_short():
    cfg = FilterConfig(min_length=100)
    hf = HeuristicFilter(cfg)
    short_text = "Hi there."
    passes, checks = hf.passes(short_text)
    assert not passes
    assert not checks["length"]


# ---------------------------------------------------------------------------
# 7. HeuristicFilter — good text passes
# ---------------------------------------------------------------------------

def test_heuristic_filter_passes_good_text():
    cfg = FilterConfig(min_length=10, min_alpha_ratio=0.5, max_repetition_ratio=0.5)
    hf = HeuristicFilter(cfg)
    # Good English prose: long enough, alpha-heavy, non-repetitive
    good_text = (
        "The scientist carefully examined the ancient fossil under a bright microscope lens. "
        "She noted its unusual spiral structure and compared it with published reference specimens. "
        "After several hours the analysis confirmed it belonged to a previously undescribed genus."
    )
    passes, checks = hf.passes(good_text)
    assert passes, f"Expected good text to pass, checks: {checks}"


# ---------------------------------------------------------------------------
# 8. MinHashSketch — signature shape is (n_perms,)
# ---------------------------------------------------------------------------

def test_minhash_sketch_shape():
    n_perms = 64
    sketch = MinHashSketch(n_perms=n_perms)
    sig = sketch.compute("some sample text for hashing purposes")
    assert isinstance(sig, np.ndarray)
    assert sig.shape == (n_perms,)


# ---------------------------------------------------------------------------
# 9. MinHashSketch — similarity of identical signatures is 1.0
# ---------------------------------------------------------------------------

def test_minhash_similarity_identical():
    sketch = MinHashSketch(n_perms=64)
    text = "The quick brown fox jumps over the lazy dog near the riverbank."
    sig = sketch.compute(text)
    sim = sketch.similarity(sig, sig)
    assert sim == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10. MinHashSketch — similarity of very different texts is < 1.0
# ---------------------------------------------------------------------------

def test_minhash_similarity_different():
    sketch = MinHashSketch(n_perms=128, seed=0)
    text_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    text_b = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    sig_a = sketch.compute(text_a)
    sig_b = sketch.compute(text_b)
    sim = sketch.similarity(sig_a, sig_b)
    assert sim < 1.0, f"Expected similarity < 1.0 for very different texts, got {sim}"


# ---------------------------------------------------------------------------
# 11. DeduplicationIndex — removes near-duplicate
# ---------------------------------------------------------------------------

def test_deduplication_index_removes_duplicate():
    cfg = FilterConfig(dedup_threshold=0.8, n_minhash_perms=64)
    idx = DeduplicationIndex(cfg)

    base = "The quick brown fox jumps over the lazy dog. " * 20
    near_dup = base  # exact copy → should be flagged

    unique_text = (
        "Astronomers discovered thousands of exoplanets orbiting distant stars in our galaxy. " * 20
    )

    texts = [base, near_dup, unique_text]
    result = idx.deduplicate_batch(texts)

    # near_dup should be dropped, so we expect 2 unique texts
    assert len(result) == 2
    assert base in result
    assert unique_text in result


# ---------------------------------------------------------------------------
# 12. DataQualityPipeline — stats keys present and consistent
# ---------------------------------------------------------------------------

def test_data_quality_pipeline_stats_keys():
    cfg = FilterConfig(min_length=10, min_alpha_ratio=0.3, max_repetition_ratio=0.9)
    pipeline = DataQualityPipeline(cfg)

    texts = [
        "The scientist examined the fossil under a bright microscope and took detailed notes.",
        "x",  # too short — filtered out
        "A programmer wrote efficient code late into the dark winter night with great care.",
        "Mountains rise above the clouds where eagles soar through the empty crisp cold air.",
    ]
    kept, stats = pipeline.process(texts)

    required_keys = {"n_original", "n_kept", "n_filtered", "n_deduped"}
    assert required_keys == set(stats.keys()), f"Missing keys: {required_keys - set(stats.keys())}"

    assert stats["n_original"] == len(texts)
    assert stats["n_kept"] + stats["n_filtered"] + stats["n_deduped"] == stats["n_original"]
    assert stats["n_kept"] == len(kept)
