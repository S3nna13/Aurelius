"""Tests for src/eval/clm_metrics.py.

All tests use tiny configs (small vocab, small B/T) and pure PyTorch-free
stdlib operations — no HuggingFace, no scipy, no sklearn.

≥ 10 tests covering:
  - CLMMetricsConfig defaults
  - compute_ngram_frequencies (unigrams + bigrams)
  - compute_distinct_n (all-unique, partial-repeat, empty/too-short)
  - compute_repetition_rate (no-repeat, with-repeat)
  - compute_self_bleu ([0, 1] range, single-sequence edge case)
  - compute_length_stats (required keys present)
  - compute_vocabulary_coverage (in (0, 1])
  - GenerationMetrics.evaluate (required keys present)
  - GenerationMetrics.compare (difference dict correctness)
"""

from __future__ import annotations

from src.eval.clm_metrics import (
    CLMMetricsConfig,
    GenerationMetrics,
    compute_distinct_n,
    compute_length_stats,
    compute_ngram_frequencies,
    compute_repetition_rate,
    compute_self_bleu,
    compute_vocabulary_coverage,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 20  # small vocabulary for all tests

# Sequences with distinct tokens — no repeated n-grams
DIVERSE_SEQS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
]

# Sequences that are identical — maximally repetitive
REPEATED_SEQS = [
    [1, 2, 3, 1, 2, 3],
    [1, 2, 3, 1, 2, 3],
    [1, 2, 3, 1, 2, 3],
]


# ===========================================================================
# 1. CLMMetricsConfig defaults
# ===========================================================================


class TestCLMMetricsConfig:
    def test_default_ngram_sizes(self):
        cfg = CLMMetricsConfig()
        assert cfg.ngram_sizes == [1, 2, 3, 4]

    def test_default_repetition_window(self):
        cfg = CLMMetricsConfig()
        assert cfg.repetition_window == 32

    def test_default_distinct_n(self):
        cfg = CLMMetricsConfig()
        assert cfg.distinct_n == 2

    def test_default_length_penalty_alpha(self):
        cfg = CLMMetricsConfig()
        assert cfg.length_penalty_alpha == 0.0

    def test_custom_values(self):
        cfg = CLMMetricsConfig(
            ngram_sizes=[1, 2], repetition_window=8, distinct_n=1, length_penalty_alpha=0.5
        )
        assert cfg.ngram_sizes == [1, 2]
        assert cfg.repetition_window == 8
        assert cfg.length_penalty_alpha == 0.5


# ===========================================================================
# 2. compute_ngram_frequencies
# ===========================================================================


class TestComputeNgramFrequencies:
    def test_unigram_counts(self):
        tokens = [1, 2, 2, 3]
        freq = compute_ngram_frequencies(tokens, 1)
        assert freq[(1,)] == 1
        assert freq[(2,)] == 2
        assert freq[(3,)] == 1

    def test_bigram_counts(self):
        tokens = [1, 2, 1, 2]
        freq = compute_ngram_frequencies(tokens, 2)
        assert freq[(1, 2)] == 2
        assert freq[(2, 1)] == 1

    def test_empty_tokens_returns_empty(self):
        freq = compute_ngram_frequencies([], 1)
        assert freq == {}

    def test_too_short_for_bigram(self):
        freq = compute_ngram_frequencies([5], 2)
        assert freq == {}


# ===========================================================================
# 3. compute_distinct_n
# ===========================================================================


class TestComputeDistinctN:
    def test_all_unique_unigrams_returns_1(self):
        tokens = [0, 1, 2, 3, 4]
        assert compute_distinct_n(tokens, 1) == 1.0

    def test_all_same_tokens_unigram_less_than_1(self):
        tokens = [7, 7, 7, 7, 7]
        assert compute_distinct_n(tokens, 1) < 1.0

    def test_partial_repeat(self):
        tokens = [1, 2, 1, 3]  # 3 unique bigrams out of 3: (1,2),(2,1),(1,3) — all unique
        result = compute_distinct_n(tokens, 1)
        # unigrams: 1,2,1,3 → 3 unique / 4 total = 0.75
        assert abs(result - 0.75) < 1e-9

    def test_empty_returns_0(self):
        assert compute_distinct_n([], 1) == 0.0

    def test_too_short_for_bigram_returns_0(self):
        assert compute_distinct_n([5], 2) == 0.0

    def test_bigram_all_unique(self):
        tokens = [1, 2, 3, 4, 5]
        # bigrams: (1,2),(2,3),(3,4),(4,5) — all distinct
        assert compute_distinct_n(tokens, 2) == 1.0


# ===========================================================================
# 4. compute_repetition_rate
# ===========================================================================


class TestComputeRepetitionRate:
    def test_all_unique_returns_0(self):
        tokens = [0, 1, 2, 3, 4, 5]
        assert compute_repetition_rate(tokens, window=32) == 0.0

    def test_all_same_returns_high_rate(self):
        tokens = [1, 1, 1, 1, 1]
        rate = compute_repetition_rate(tokens, window=32)
        assert rate > 0.0

    def test_all_same_rate_is_1(self):
        tokens = [1, 1, 1, 1]
        rate = compute_repetition_rate(tokens, window=32)
        assert abs(rate - 1.0) < 1e-9

    def test_single_token_returns_0(self):
        assert compute_repetition_rate([42], window=32) == 0.0

    def test_empty_returns_0(self):
        assert compute_repetition_rate([], window=32) == 0.0

    def test_window_limits_lookback(self):
        # Token at index 4 equals token at index 0; with window=2 it should NOT
        # be flagged, but with window=10 it SHOULD.
        tokens = [1, 2, 3, 4, 1]
        rate_narrow = compute_repetition_rate(tokens, window=2)
        rate_wide = compute_repetition_rate(tokens, window=10)
        assert rate_narrow < rate_wide


# ===========================================================================
# 5. compute_self_bleu
# ===========================================================================


class TestComputeSelfBleu:
    def test_in_range_0_to_1(self):
        seqs = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
        result = compute_self_bleu(seqs, n=2)
        assert 0.0 <= result <= 1.0

    def test_single_sequence_returns_0(self):
        result = compute_self_bleu([[1, 2, 3, 4]], n=4)
        assert result == 0.0

    def test_identical_sequences_gives_high_score(self):
        seq = [1, 2, 3, 4, 5]
        seqs = [seq[:], seq[:], seq[:]]
        result = compute_self_bleu(seqs, n=2)
        assert result > 0.5

    def test_diverse_sequences_gives_lower_than_identical(self):
        identical = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        diverse = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        score_identical = compute_self_bleu(identical, n=2)
        score_diverse = compute_self_bleu(diverse, n=2)
        assert score_identical >= score_diverse


# ===========================================================================
# 6. compute_length_stats
# ===========================================================================


class TestComputeLengthStats:
    REQUIRED_KEYS = {"mean", "std", "min", "max", "median"}

    def test_all_required_keys_present(self):
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        stats = compute_length_stats(seqs)
        assert self.REQUIRED_KEYS == set(stats.keys())

    def test_correct_mean(self):
        seqs = [[1, 2], [3, 4], [5, 6]]  # all length 2
        stats = compute_length_stats(seqs)
        assert abs(stats["mean"] - 2.0) < 1e-9

    def test_min_max_correct(self):
        seqs = [[1], [2, 3, 4], [5, 6]]
        stats = compute_length_stats(seqs)
        assert stats["min"] == 1
        assert stats["max"] == 3

    def test_empty_sequences_returns_zeros(self):
        stats = compute_length_stats([])
        assert stats["mean"] == 0.0
        assert stats["min"] == 0


# ===========================================================================
# 7. compute_vocabulary_coverage
# ===========================================================================


class TestComputeVocabularyCoverage:
    def test_coverage_in_range(self):
        tokens = list(range(10))
        cov = compute_vocabulary_coverage(tokens, VOCAB_SIZE)
        assert 0.0 < cov <= 1.0

    def test_full_coverage(self):
        tokens = list(range(VOCAB_SIZE))
        cov = compute_vocabulary_coverage(tokens, VOCAB_SIZE)
        assert abs(cov - 1.0) < 1e-9

    def test_empty_tokens_returns_0(self):
        assert compute_vocabulary_coverage([], VOCAB_SIZE) == 0.0

    def test_partial_coverage(self):
        # 5 unique tokens out of vocab_size=20 → 0.25
        tokens = [0, 1, 2, 3, 4, 0, 1]
        cov = compute_vocabulary_coverage(tokens, 20)
        assert abs(cov - 0.25) < 1e-9


# ===========================================================================
# 8. GenerationMetrics.evaluate
# ===========================================================================

EXPECTED_EVAL_KEYS = {
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "distinct_4",
    "repetition_rate",
    "self_bleu",
    "length_mean",
    "length_std",
    "vocab_coverage",
}


class TestGenerationMetricsEvaluate:
    def _make_evaluator(self):
        return GenerationMetrics(CLMMetricsConfig())

    def test_all_required_keys_present(self):
        ev = self._make_evaluator()
        result = ev.evaluate(DIVERSE_SEQS, VOCAB_SIZE)
        assert EXPECTED_EVAL_KEYS == set(result.keys())

    def test_values_are_floats(self):
        ev = self._make_evaluator()
        result = ev.evaluate(DIVERSE_SEQS, VOCAB_SIZE)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_distinct_scores_in_range(self):
        ev = self._make_evaluator()
        result = ev.evaluate(DIVERSE_SEQS, VOCAB_SIZE)
        for k in [1, 2, 3, 4]:
            assert 0.0 <= result[f"distinct_{k}"] <= 1.0

    def test_repetition_rate_in_range(self):
        ev = self._make_evaluator()
        result = ev.evaluate(REPEATED_SEQS, VOCAB_SIZE)
        assert 0.0 <= result["repetition_rate"] <= 1.0

    def test_vocab_coverage_for_diverse_seqs(self):
        ev = self._make_evaluator()
        result = ev.evaluate(DIVERSE_SEQS, VOCAB_SIZE)
        # diverse sequences use 15 out of 20 tokens → 0.75
        assert abs(result["vocab_coverage"] - 0.75) < 1e-9


# ===========================================================================
# 9. GenerationMetrics.compare
# ===========================================================================


class TestGenerationMetricsCompare:
    def _make_evaluator(self):
        return GenerationMetrics(CLMMetricsConfig())

    def test_compare_returns_all_keys(self):
        ev = self._make_evaluator()
        ma = ev.evaluate(DIVERSE_SEQS, VOCAB_SIZE)
        mb = ev.evaluate(REPEATED_SEQS, VOCAB_SIZE)
        diff = ev.compare(ma, mb)
        assert set(diff.keys()) == EXPECTED_EVAL_KEYS

    def test_compare_zero_for_identical_metrics(self):
        ev = self._make_evaluator()
        ma = ev.evaluate(DIVERSE_SEQS, VOCAB_SIZE)
        diff = ev.compare(ma, ma)
        for key, val in diff.items():
            assert abs(val) < 1e-12, f"{key} diff should be 0, got {val}"

    def test_compare_signed_difference(self):
        ev = self._make_evaluator()
        # metrics_b is metrics_a with one field bumped
        ma = {
            "distinct_1": 0.5,
            "distinct_2": 0.4,
            "distinct_3": 0.3,
            "distinct_4": 0.2,
            "repetition_rate": 0.1,
            "self_bleu": 0.2,
            "length_mean": 5.0,
            "length_std": 1.0,
            "vocab_coverage": 0.5,
        }
        mb = dict(ma)
        mb["distinct_1"] = 0.8
        diff = ev.compare(ma, mb)
        assert abs(diff["distinct_1"] - 0.3) < 1e-9
        assert abs(diff["distinct_2"]) < 1e-9
