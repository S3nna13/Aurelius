"""Tests for src/data/preprocessing.py"""
import pytest

from src.data.preprocessing import (
    PreprocessConfig,
    normalize_whitespace_fn,
    remove_html_tags,
    filter_by_length,
    is_valid_utf8,
    compute_text_hash,
    ExactDeduplicator,
    MinHashDeduplicator,
    TextCleaner,
)


# ---------------------------------------------------------------------------
# PreprocessConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = PreprocessConfig()
    assert cfg.min_length == 10
    assert cfg.max_length == 2048
    assert cfg.dedup_threshold == pytest.approx(0.9)
    assert cfg.filter_non_utf8 is True
    assert cfg.normalize_whitespace is True
    assert cfg.remove_html is True


# ---------------------------------------------------------------------------
# normalize_whitespace_fn
# ---------------------------------------------------------------------------

def test_normalize_whitespace_collapses_spaces():
    result = normalize_whitespace_fn("hello    world")
    assert result == "hello world"


def test_normalize_whitespace_collapses_tabs_and_newlines():
    result = normalize_whitespace_fn("hello\t\nworld")
    assert result == "hello world"


def test_normalize_whitespace_strips_edges():
    result = normalize_whitespace_fn("  hello  ")
    assert result == "hello"


def test_normalize_whitespace_empty():
    assert normalize_whitespace_fn("") == ""


# ---------------------------------------------------------------------------
# remove_html_tags
# ---------------------------------------------------------------------------

def test_remove_html_removes_tags():
    result = remove_html_tags("<b>bold</b> text")
    assert "<b>" not in result
    assert "</b>" not in result


def test_remove_html_preserves_content():
    result = remove_html_tags("<p>Hello world</p>")
    assert "Hello world" in result


def test_remove_html_no_tags_unchanged():
    text = "plain text here"
    assert remove_html_tags(text) == text


# ---------------------------------------------------------------------------
# filter_by_length
# ---------------------------------------------------------------------------

def test_filter_by_length_keeps_correct():
    texts = ["short", "a" * 20, "b" * 100]
    result = filter_by_length(texts, min_len=10, max_len=50)
    assert "a" * 20 in result


def test_filter_by_length_rejects_short():
    texts = ["hi", "hello world this is long enough"]
    result = filter_by_length(texts, min_len=10, max_len=100)
    assert "hi" not in result


def test_filter_by_length_rejects_long():
    texts = ["a" * 500, "normal length text here"]
    result = filter_by_length(texts, min_len=5, max_len=100)
    assert "a" * 500 not in result


# ---------------------------------------------------------------------------
# compute_text_hash
# ---------------------------------------------------------------------------

def test_compute_text_hash_same_input():
    h1 = compute_text_hash("hello")
    h2 = compute_text_hash("hello")
    assert h1 == h2


def test_compute_text_hash_different_inputs():
    assert compute_text_hash("hello") != compute_text_hash("world")


def test_compute_text_hash_returns_hex_string():
    h = compute_text_hash("test")
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex = 64 chars


# ---------------------------------------------------------------------------
# ExactDeduplicator
# ---------------------------------------------------------------------------

def test_exact_dedup_first_occurrence_true():
    d = ExactDeduplicator()
    assert d.add_and_check("hello world") is True


def test_exact_dedup_second_occurrence_false():
    d = ExactDeduplicator()
    d.add_and_check("hello world")
    assert d.add_and_check("hello world") is False


def test_exact_dedup_deduplicate_removes_dups():
    d = ExactDeduplicator()
    texts = ["apple", "banana", "apple", "cherry"]
    result = d.deduplicate(texts)
    assert result == ["apple", "banana", "cherry"]


def test_exact_dedup_len():
    d = ExactDeduplicator()
    d.add_and_check("a")
    d.add_and_check("b")
    d.add_and_check("a")
    assert len(d) == 2


# ---------------------------------------------------------------------------
# MinHashDeduplicator
# ---------------------------------------------------------------------------

def test_minhash_dedup_identical_as_duplicate():
    d = MinHashDeduplicator(n_hashes=64, threshold=0.8)
    texts = ["the quick brown fox", "the quick brown fox"]
    result = d.deduplicate(texts)
    assert len(result) == 1


def test_minhash_dedup_keeps_different():
    d = MinHashDeduplicator(n_hashes=64, threshold=0.8)
    texts = ["the quick brown fox", "completely different text about cats and dogs"]
    result = d.deduplicate(texts)
    assert len(result) == 2


def test_minhash_is_near_duplicate_same():
    d = MinHashDeduplicator(n_hashes=64, threshold=0.8)
    text = "the quick brown fox jumps over the lazy dog"
    sig = d._compute_minhash(text)
    assert d.is_near_duplicate(text, sig) is True


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

def test_text_cleaner_clean_returns_string():
    cfg = PreprocessConfig()
    cleaner = TextCleaner(cfg)
    result = cleaner.clean("<p>Hello world</p>")
    assert isinstance(result, str)


def test_text_cleaner_clean_removes_html():
    cfg = PreprocessConfig(remove_html=True)
    cleaner = TextCleaner(cfg)
    result = cleaner.clean("<b>bold</b>")
    assert "<b>" not in result


def test_text_cleaner_clean_batch_filters_by_length():
    cfg = PreprocessConfig(min_length=10, max_length=100)
    cleaner = TextCleaner(cfg)
    texts = ["short", "this is a longer string that should be kept in the result"]
    result = cleaner.clean_batch(texts)
    assert all(10 <= len(t) <= 100 for t in result)
    assert "short" not in result


def test_text_cleaner_get_stats_keys():
    cfg = PreprocessConfig()
    cleaner = TextCleaner(cfg)
    original = ["hello world", "foo bar baz"]
    cleaned = ["hello world"]
    stats = cleaner.get_stats(original, cleaned)
    for key in ["n_original", "n_kept", "filter_rate", "mean_length_before", "mean_length_after"]:
        assert key in stats


def test_text_cleaner_get_stats_filter_rate():
    cfg = PreprocessConfig()
    cleaner = TextCleaner(cfg)
    original = ["a", "b", "c", "d"]
    cleaned = ["a", "b"]
    stats = cleaner.get_stats(original, cleaned)
    assert 0.0 <= stats["filter_rate"] <= 1.0
    assert stats["filter_rate"] == pytest.approx(0.5)
