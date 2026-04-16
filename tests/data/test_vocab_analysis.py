"""Tests for src/data/vocab_analysis.py.

Pure stdlib — no PyTorch, HuggingFace, scipy, or sklearn.
Tiny configs / data to keep the suite fast.
"""

from __future__ import annotations

import math

import pytest

from src.data.vocab_analysis import (
    VocabAnalyzer,
    VocabConfig,
    compute_fertility,
    compute_token_frequencies,
    compute_token_length_distribution,
    compute_zipf_exponent,
    estimate_compression_ratio,
    find_dead_tokens,
    find_rare_tokens,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SMALL_VOCAB_SIZE = 10

# A tiny vocab: ids 0-9 with distinct string lengths
TINY_VOCAB: dict[int, str] = {
    0: "<pad>",
    1: "<eos>",
    2: "hello",
    3: "world",
    4: "##ing",
    5: "\u0120the",
    6: "42",
    7: "!",
    8: "a",
    9: "foo",
}

# Token sequence: ids 0-4 appear; 5-9 never appear.
TINY_TOKEN_IDS = [0, 0, 1, 2, 3, 2, 0, 1, 4]
# Expected frequencies for ids 0..9:
#   0->3, 1->2, 2->2, 3->1, 4->1, 5->0, 6->0, 7->0, 8->0, 9->0


# ---------------------------------------------------------------------------
# VocabConfig defaults
# ---------------------------------------------------------------------------


def test_vocab_config_defaults():
    cfg = VocabConfig()
    assert cfg.vocab_size == 50257
    assert cfg.min_freq == 1
    assert cfg.max_token_len == 20
    assert cfg.special_tokens == ["<pad>", "<eos>", "<bos>", "<unk>"]


def test_vocab_config_custom():
    cfg = VocabConfig(vocab_size=100, min_freq=3)
    assert cfg.vocab_size == 100
    assert cfg.min_freq == 3


def test_vocab_config_special_tokens_are_independent():
    """Mutable default must not be shared between instances."""
    cfg1 = VocabConfig()
    cfg2 = VocabConfig()
    cfg1.special_tokens.append("<extra>")
    assert "<extra>" not in cfg2.special_tokens


# ---------------------------------------------------------------------------
# compute_token_frequencies
# ---------------------------------------------------------------------------


def test_frequencies_length_equals_vocab_size():
    freqs = compute_token_frequencies(TINY_TOKEN_IDS, SMALL_VOCAB_SIZE)
    assert len(freqs) == SMALL_VOCAB_SIZE


def test_frequencies_correct_counts():
    freqs = compute_token_frequencies(TINY_TOKEN_IDS, SMALL_VOCAB_SIZE)
    assert freqs[0] == 3
    assert freqs[1] == 2
    assert freqs[2] == 2
    assert freqs[3] == 1
    assert freqs[4] == 1
    assert freqs[5] == 0
    assert freqs[9] == 0


def test_frequencies_out_of_range_ignored():
    freqs = compute_token_frequencies([0, 99, -1, 5], vocab_size=10)
    assert freqs[0] == 1
    assert freqs[5] == 1
    assert sum(freqs) == 2  # only in-range tokens counted


def test_frequencies_empty_input():
    freqs = compute_token_frequencies([], vocab_size=8)
    assert freqs == [0] * 8


# ---------------------------------------------------------------------------
# compute_zipf_exponent
# ---------------------------------------------------------------------------


def test_zipf_exponent_returns_positive_float():
    freqs = compute_token_frequencies(TINY_TOKEN_IDS, SMALL_VOCAB_SIZE)
    alpha = compute_zipf_exponent(freqs)
    assert isinstance(alpha, float)
    assert alpha > 0.0


def test_zipf_exponent_typical_range():
    # Natural language-like distribution: f(r) = 1/r
    freqs = [0] * 20
    for r in range(1, 21):
        freqs[r - 1] = max(1, 1000 // r)
    alpha = compute_zipf_exponent(freqs)
    # Expect alpha close to 1.0 for 1/r distribution
    assert 0.5 < alpha < 2.0


def test_zipf_exponent_raises_with_too_few_nonzero():
    with pytest.raises(ValueError):
        compute_zipf_exponent([0, 0, 5])  # only one non-zero → need at least 2


# ---------------------------------------------------------------------------
# find_rare_tokens
# ---------------------------------------------------------------------------


def test_find_rare_tokens_only_nonzero_below_min():
    freqs = [0, 1, 2, 5, 0, 3]
    # min_freq=3 → tokens with 0 < freq < 3: indices 1 (freq=1), 2 (freq=2)
    rare = find_rare_tokens(freqs, min_freq=3)
    assert rare == [1, 2]


def test_find_rare_tokens_excludes_zero_freq():
    freqs = [0, 0, 1, 2]
    rare = find_rare_tokens(freqs, min_freq=5)
    assert 0 not in rare
    assert 1 not in rare
    assert 2 in rare
    assert 3 in rare


def test_find_rare_tokens_sorted():
    freqs = [1, 0, 1, 0, 1]
    rare = find_rare_tokens(freqs, min_freq=2)
    assert rare == sorted(rare)


def test_find_rare_tokens_none_when_all_above_min():
    freqs = [10, 20, 30]
    assert find_rare_tokens(freqs, min_freq=5) == []


# ---------------------------------------------------------------------------
# find_dead_tokens
# ---------------------------------------------------------------------------


def test_find_dead_tokens_returns_zero_freq_only():
    freqs = compute_token_frequencies(TINY_TOKEN_IDS, SMALL_VOCAB_SIZE)
    dead = find_dead_tokens(freqs)
    assert all(freqs[i] == 0 for i in dead)
    assert set(dead) == {5, 6, 7, 8, 9}


def test_find_dead_tokens_empty_when_all_seen():
    freqs = [1, 2, 3, 4]
    assert find_dead_tokens(freqs) == []


# ---------------------------------------------------------------------------
# compute_token_length_distribution
# ---------------------------------------------------------------------------


def test_length_distribution_has_required_keys():
    result = compute_token_length_distribution(TINY_VOCAB)
    for key in ("mean", "std", "max", "min", "histogram"):
        assert key in result, f"Missing key: {key}"


def test_length_distribution_histogram_sums_to_vocab_size():
    result = compute_token_length_distribution(TINY_VOCAB)
    assert sum(result["histogram"]) == len(TINY_VOCAB)


def test_length_distribution_min_max_correct():
    result = compute_token_length_distribution(TINY_VOCAB)
    lengths = [len(t) for t in TINY_VOCAB.values()]
    assert result["min"] == min(lengths)
    assert result["max"] == max(lengths)


def test_length_distribution_mean_correct():
    simple_vocab = {0: "a", 1: "bb", 2: "ccc"}
    result = compute_token_length_distribution(simple_vocab)
    assert abs(result["mean"] - 2.0) < 1e-9


def test_length_distribution_empty_vocab():
    result = compute_token_length_distribution({})
    assert result["histogram"] == []
    assert result["mean"] == 0.0


# ---------------------------------------------------------------------------
# compute_fertility
# ---------------------------------------------------------------------------


def test_fertility_handles_division_by_zero():
    result = compute_fertility([], char_count=100)
    assert result == 0.0


def test_fertility_correct_value():
    # 10 chars, 5 tokens → 2.0 chars/token
    result = compute_fertility([1, 2, 3, 4, 5], char_count=10)
    assert abs(result - 2.0) < 1e-9


def test_fertility_fractional():
    result = compute_fertility([0, 1, 2], char_count=7)
    assert abs(result - 7 / 3) < 1e-9


# ---------------------------------------------------------------------------
# estimate_compression_ratio
# ---------------------------------------------------------------------------


def test_compression_ratio_positive_float():
    ratio = estimate_compression_ratio("hello world", [1, 2, 3])
    assert isinstance(ratio, float)
    assert ratio > 0.0


def test_compression_ratio_correct_value():
    text = "abc"  # 3 bytes in UTF-8
    tokens = [0, 1]  # 2 tokens → ratio = 1.5
    ratio = estimate_compression_ratio(text, tokens)
    assert abs(ratio - 1.5) < 1e-9


def test_compression_ratio_empty_tokens():
    ratio = estimate_compression_ratio("hello", [])
    assert ratio == 0.0


def test_compression_ratio_multibyte():
    # "é" is 2 bytes in UTF-8; 1 token → ratio = 2.0
    text = "\u00e9"
    ratio = estimate_compression_ratio(text, [42])
    assert abs(ratio - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# VocabAnalyzer.analyze_corpus
# ---------------------------------------------------------------------------


def test_analyze_corpus_has_required_keys():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE, min_freq=2)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.analyze_corpus(TINY_TOKEN_IDS)
    for key in ("frequencies", "zipf_exponent", "rare_tokens", "dead_tokens",
                "fertility", "coverage"):
        assert key in result, f"Missing key: {key}"


def test_analyze_corpus_frequencies_length():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.analyze_corpus(TINY_TOKEN_IDS)
    assert len(result["frequencies"]) == SMALL_VOCAB_SIZE


def test_analyze_corpus_coverage_between_0_and_1():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.analyze_corpus(TINY_TOKEN_IDS)
    assert 0.0 <= result["coverage"] <= 1.0


def test_analyze_corpus_coverage_value():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.analyze_corpus(TINY_TOKEN_IDS)
    # ids 0-4 appear; 5 seen tokens out of 10 vocab → 0.5
    assert abs(result["coverage"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# VocabAnalyzer.find_similar_tokens
# ---------------------------------------------------------------------------


def test_find_similar_tokens_length_equals_n():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.find_similar_tokens(2, n=3)  # token 2 = "hello" (5 chars)
    assert len(result) == 3


def test_find_similar_tokens_excludes_self():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.find_similar_tokens(2, n=5)
    ids = [r[0] for r in result]
    assert 2 not in ids


def test_find_similar_tokens_sorted_by_diff():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.find_similar_tokens(2, n=5)
    diffs = [r[1] for r in result]
    assert diffs == sorted(diffs)


def test_find_similar_tokens_unknown_id():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    result = analyzer.find_similar_tokens(999, n=3)
    assert result == []


# ---------------------------------------------------------------------------
# VocabAnalyzer.get_subword_stats
# ---------------------------------------------------------------------------


def test_get_subword_stats_has_required_keys():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    stats = analyzer.get_subword_stats()
    for key in ("n_prefix_tokens", "n_whole_word_tokens", "n_special_tokens",
                "n_digit_tokens", "n_punct_tokens"):
        assert key in stats, f"Missing key: {key}"


def test_get_subword_stats_prefix_tokens():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    stats = analyzer.get_subword_stats()
    # TINY_VOCAB has "##ing" (id=4) and "\u0120the" (id=5) as prefix tokens
    assert stats["n_prefix_tokens"] == 2


def test_get_subword_stats_special_tokens():
    cfg = VocabConfig(
        vocab_size=SMALL_VOCAB_SIZE,
        special_tokens=["<pad>", "<eos>"],
    )
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    stats = analyzer.get_subword_stats()
    # <pad> (0) and <eos> (1) match special_tokens list
    assert stats["n_special_tokens"] >= 2


def test_get_subword_stats_digit_tokens():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    stats = analyzer.get_subword_stats()
    # "42" (id=6) is the only digit token
    assert stats["n_digit_tokens"] == 1


def test_get_subword_stats_punct_tokens():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    stats = analyzer.get_subword_stats()
    # "!" (id=7) is the only punct token
    assert stats["n_punct_tokens"] == 1


def test_get_subword_stats_counts_non_negative():
    cfg = VocabConfig(vocab_size=SMALL_VOCAB_SIZE)
    analyzer = VocabAnalyzer(TINY_VOCAB, cfg)
    stats = analyzer.get_subword_stats()
    for key, val in stats.items():
        assert val >= 0, f"Negative count for {key}"
