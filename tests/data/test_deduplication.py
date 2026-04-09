"""Tests for src/data/deduplication.py — MinHash LSH deduplication pipeline."""

from __future__ import annotations

import sys

import pytest

from src.data.deduplication import (
    LSHIndex,
    StreamingDeduplicator,
    estimate_jaccard,
    is_duplicate,
    minhash_signature,
    shingle,
)


# ---------------------------------------------------------------------------
# shingle tests
# ---------------------------------------------------------------------------

def test_shingle_basic():
    """Shingles of length k are extracted correctly."""
    text = "abcdef"
    result = shingle(text, k=5)
    assert result == {"abcde", "bcdef"}


def test_shingle_empty_string():
    """Empty string returns empty set."""
    assert shingle("", k=5) == set()


def test_shingle_text_shorter_than_k():
    """Text shorter than k returns empty set."""
    assert shingle("abc", k=5) == set()


def test_shingle_exact_length():
    """Text of exactly length k returns a single shingle."""
    result = shingle("hello", k=5)
    assert result == {"hello"}


# ---------------------------------------------------------------------------
# minhash_signature tests
# ---------------------------------------------------------------------------

def test_minhash_signature_length():
    """Signature length equals n_hashes."""
    sig = minhash_signature({"abc", "def", "ghi"}, n_hashes=64)
    assert len(sig) == 64


def test_minhash_signature_identical_texts():
    """Identical shingle sets produce identical signatures."""
    shingles = shingle("the quick brown fox", k=5)
    sig1 = minhash_signature(shingles, n_hashes=64, seed=42)
    sig2 = minhash_signature(shingles, n_hashes=64, seed=42)
    assert sig1 == sig2


def test_minhash_signature_empty_shingles():
    """Empty shingle set returns all sys.maxsize values."""
    sig = minhash_signature(set(), n_hashes=16)
    assert sig == [sys.maxsize] * 16


def test_minhash_signature_deterministic_with_seed():
    """Same seed produces same signature for same shingles."""
    shingles = {"ab", "cd", "ef"}
    assert minhash_signature(shingles, n_hashes=32, seed=7) == minhash_signature(
        shingles, n_hashes=32, seed=7
    )


# ---------------------------------------------------------------------------
# estimate_jaccard tests
# ---------------------------------------------------------------------------

def test_estimate_jaccard_identical_sigs():
    """Identical signatures yield Jaccard == 1.0."""
    sig = [1, 2, 3, 4, 5]
    assert estimate_jaccard(sig, sig) == 1.0


def test_estimate_jaccard_no_overlap():
    """Completely different signatures yield Jaccard == 0.0."""
    sig1 = [1, 2, 3, 4]
    sig2 = [5, 6, 7, 8]
    assert estimate_jaccard(sig1, sig2) == 0.0


def test_estimate_jaccard_partial():
    """Partial overlap returns correct fraction."""
    sig1 = [1, 2, 3, 4]
    sig2 = [1, 2, 5, 6]
    assert estimate_jaccard(sig1, sig2) == 0.5


# ---------------------------------------------------------------------------
# is_duplicate tests
# ---------------------------------------------------------------------------

def test_is_duplicate_identical():
    """Identical signatures are duplicates."""
    sig = [10, 20, 30, 40]
    assert is_duplicate(sig, sig, threshold=0.8) is True


def test_is_duplicate_completely_different():
    """Completely different signatures are not duplicates."""
    sig1 = [1, 2, 3, 4]
    sig2 = [5, 6, 7, 8]
    assert is_duplicate(sig1, sig2, threshold=0.8) is False


# ---------------------------------------------------------------------------
# LSHIndex tests
# ---------------------------------------------------------------------------

def test_lsh_index_add_exact_duplicate_detected():
    """Adding an exact duplicate signature returns the original doc as a candidate."""
    index = LSHIndex(n_hashes=128, n_bands=32, threshold=0.8)

    text = "the quick brown fox jumps over the lazy dog"
    shingles_set = shingle(text, k=5)
    sig = minhash_signature(shingles_set, n_hashes=128, seed=42)

    candidates_first = index.add("doc1", sig)
    assert candidates_first == [], "First doc should have no candidates"

    candidates_second = index.add("doc2", sig)
    assert "doc1" in candidates_second, "Exact duplicate should surface doc1 as candidate"


def test_lsh_index_no_false_duplicate_for_unrelated():
    """Very different documents should not share band buckets (probabilistically)."""
    index = LSHIndex(n_hashes=128, n_bands=32, threshold=0.8)

    sig1 = minhash_signature(shingle("aaa bbb ccc ddd eee fff", k=5), n_hashes=128, seed=42)
    sig2 = minhash_signature(shingle("zzz yyy xxx www vvv uuu", k=5), n_hashes=128, seed=42)

    index.add("docA", sig1)
    candidates = index.add("docB", sig2)
    # Not a strict guarantee, but for very different texts this should hold
    # (if it fires, the band hash collision is a true false positive — acceptable)
    for cid in candidates:
        candidate_sig = index._signatures[cid]
        jaccard = estimate_jaccard(sig1, candidate_sig)
        assert jaccard < 0.8, f"Unexpected high Jaccard {jaccard} for unrelated docs"


# ---------------------------------------------------------------------------
# StreamingDeduplicator tests
# ---------------------------------------------------------------------------

def test_streaming_deduplicator_keeps_first_drops_duplicate():
    """First document is kept; an exact duplicate is dropped."""
    deduper = StreamingDeduplicator(n_hashes=128, n_bands=32, threshold=0.8, shingle_k=5)
    text = "the quick brown fox jumps over the lazy dog and runs away"
    assert deduper.process("doc1", text) is True, "First document should be kept"
    assert deduper.process("doc2", text) is False, "Exact duplicate should be dropped"


def test_streaming_deduplicator_keeps_distinct_docs():
    """Clearly distinct documents are both kept."""
    deduper = StreamingDeduplicator(n_hashes=128, n_bands=32, threshold=0.8, shingle_k=5)
    text1 = "apple banana cherry date elderberry fig grape honeydew"
    text2 = "zebra yak xenops wolf vulture turtle snake rabbit"
    assert deduper.process("doc1", text1) is True
    assert deduper.process("doc2", text2) is True


def test_streaming_deduplicator_stats_keys_present():
    """stats() dict contains all required keys."""
    deduper = StreamingDeduplicator()
    deduper.process("d1", "hello world this is a test document")
    stats = deduper.stats()
    for key in ("n_seen", "n_kept", "n_dropped", "duplicate_rate"):
        assert key in stats, f"Missing key: {key}"


def test_streaming_deduplicator_stats_counts():
    """stats() returns accurate counts after processing."""
    deduper = StreamingDeduplicator(n_hashes=128, n_bands=32, threshold=0.8, shingle_k=5)
    text = "the quick brown fox jumps over the lazy dog once more time"
    deduper.process("d1", text)
    deduper.process("d2", text)  # duplicate
    deduper.process("d3", "completely different text about zebras and antelopes running")

    stats = deduper.stats()
    assert stats["n_seen"] == 3
    assert stats["n_kept"] == 2
    assert stats["n_dropped"] == 1
    assert abs(stats["duplicate_rate"] - 1 / 3) < 1e-9
