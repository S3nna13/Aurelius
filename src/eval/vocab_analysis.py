"""Vocabulary and tokenization analysis: fertility, coverage, efficiency metrics."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# VocabStats dataclass
# ---------------------------------------------------------------------------

@dataclass
class VocabStats:
    """Aggregate statistics about a tokenizer over a corpus."""

    vocab_size: int
    n_special_tokens: int = 0
    coverage: float = 0.0        # fraction of corpus unique words present in vocab
    fertility: float = 0.0       # avg subword tokens per word
    oov_rate: float = 0.0        # fraction of words NOT in vocab
    compression_ratio: float = 0.0  # char_count / token_count


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def compute_fertility(
    texts: list[str],
    tokenize_fn: Callable[[str], list[int]],
    word_split: str = " ",
) -> float:
    """Return fertility: total subword tokens / total words.

    A perfect (word-level) tokenizer returns 1.0; higher values indicate
    more fragmentation.
    """
    total_words = 0
    total_tokens = 0
    for text in texts:
        words = text.split(word_split) if word_split else text.split()
        words = [w for w in words if w]  # drop empty strings from extra spaces
        total_words += len(words)
        total_tokens += len(tokenize_fn(text))
    if total_words == 0:
        return 0.0
    return total_tokens / total_words


def compute_coverage(
    texts: list[str],
    vocab: set[str],
    word_split: str = " ",
) -> float:
    """Return word-level coverage: fraction of unique words present in vocab.

    Returns a float in [0, 1].
    """
    unique_words: set[str] = set()
    for text in texts:
        words = text.split(word_split) if word_split else text.split()
        unique_words.update(w for w in words if w)
    if not unique_words:
        return 0.0
    covered = sum(1 for w in unique_words if w in vocab)
    return covered / len(unique_words)


def compute_compression_ratio(
    texts: list[str],
    tokenize_fn: Callable[[str], list[int]],
) -> float:
    """Return chars-per-token ratio: sum(len(text)) / sum(len(tokenize(text))).

    Higher means more compression (each token covers more characters).
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenize_fn(t)) for t in texts)
    if total_tokens == 0:
        return 0.0
    return total_chars / total_tokens


def analyze_token_frequency(
    token_ids: list[int],
    vocab_size: int,
) -> dict[str, float]:
    """Analyse a flat list of token ids.

    Returns:
        entropy        – Shannon entropy over token distribution (nats)
        top10_coverage – fraction of tokens accounted for by top-10 most frequent
        hapax_ratio    – fraction of vocabulary items appearing exactly once
    """
    if not token_ids:
        return {"entropy": 0.0, "top10_coverage": 0.0, "hapax_ratio": 0.0}

    counts = Counter(token_ids)
    total = len(token_ids)

    # Shannon entropy
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log(p)

    # Top-10 coverage
    top10 = counts.most_common(10)
    top10_coverage = sum(c for _, c in top10) / total

    # Hapax ratio: unique vocab items appearing exactly once / vocab_size
    hapax_count = sum(1 for cnt in counts.values() if cnt == 1)
    hapax_ratio = hapax_count / vocab_size if vocab_size > 0 else 0.0

    return {
        "entropy": entropy,
        "top10_coverage": top10_coverage,
        "hapax_ratio": hapax_ratio,
    }


def compute_zipf_exponent(token_freqs: list[int]) -> float:
    """Fit Zipf's law (freq ∝ rank^(-alpha)) via log-log linear regression.

    Returns the exponent alpha (typically ~1.0 for natural language).
    The slope of log(freq) ~ log(rank) equals -alpha, so we return |slope|.
    """
    if len(token_freqs) < 2:
        return 0.0

    freqs = sorted(token_freqs, reverse=True)
    # Remove zero-frequency entries to avoid log(0)
    freqs = [f for f in freqs if f > 0]
    if len(freqs) < 2:
        return 0.0

    n = len(freqs)
    log_ranks = [math.log(r + 1) for r in range(n)]  # rank 1-based → index+1
    log_freqs = [math.log(f) for f in freqs]

    # Simple OLS: slope = cov(x,y) / var(x)
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    cov_xy = sum((log_ranks[i] - mean_x) * (log_freqs[i] - mean_y) for i in range(n))
    var_x = sum((log_ranks[i] - mean_x) ** 2 for i in range(n))

    if var_x == 0.0:
        return 0.0

    slope = cov_xy / var_x
    return abs(slope)  # alpha is positive; slope is negative


# ---------------------------------------------------------------------------
# TokenizerAnalyzer class
# ---------------------------------------------------------------------------

class TokenizerAnalyzer:
    """High-level analyser wrapping a tokenize function."""

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[int]],
        vocab_size: int,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_corpus(self, texts: list[str]) -> VocabStats:
        """Compute all VocabStats fields over *texts*.

        Coverage is computed against the set of words seen in *texts* itself
        (i.e., what fraction of unique words each word appears as a whole token).
        Because we use the word set as both reference and query, coverage is
        always 1.0 unless the tokenizer splits words — in which case we cannot
        tell from token ids alone.  Following the spec: vocab = set of words
        seen in texts.
        """
        if not texts:
            return VocabStats(vocab_size=self.vocab_size)

        # Build word vocab from corpus
        word_vocab: set[str] = set()
        for text in texts:
            word_vocab.update(w for w in text.split() if w)

        fertility = compute_fertility(texts, self.tokenize_fn)
        coverage = compute_coverage(texts, word_vocab)
        oov_rate = 1.0 - coverage
        compression = compute_compression_ratio(texts, self.tokenize_fn)

        return VocabStats(
            vocab_size=self.vocab_size,
            n_special_tokens=0,
            coverage=coverage,
            fertility=fertility,
            oov_rate=oov_rate,
            compression_ratio=compression,
        )

    def token_length_distribution(self, texts: list[str]) -> dict[str, float]:
        """Distribution of per-token lengths (in characters).

        Each token id is mapped back to its byte representation when the
        tokenizer is byte-level; otherwise we use the token id value as a
        proxy length.  For a general tokenize_fn we cannot recover the
        surface form from an integer id, so we report the distribution of
        *sequence lengths* (number of tokens per text) instead, which is
        always well-defined.

        Returns: {"mean": float, "std": float, "max": float, "min": float}
        """
        lengths = [len(self.tokenize_fn(t)) for t in texts]
        return _distribution_stats(lengths)

    def identify_long_tail(
        self,
        texts: list[str],
        threshold: int = 5,
    ) -> list[int]:
        """Return token ids that appear <= *threshold* times across *texts*."""
        all_ids: list[int] = []
        for text in texts:
            all_ids.extend(self.tokenize_fn(text))
        counts = Counter(all_ids)
        return [tok_id for tok_id, cnt in counts.items() if cnt <= threshold]

    def sequence_length_stats(self, texts: list[str]) -> dict[str, float]:
        """Token counts per text.

        Returns: {"mean": float, "std": float, "p50": float, "p95": float}
        """
        lengths = [len(self.tokenize_fn(t)) for t in texts]
        if not lengths:
            return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}

        n = len(lengths)
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        std = math.sqrt(variance)

        sorted_lengths = sorted(lengths)
        p50 = _percentile(sorted_lengths, 50)
        p95 = _percentile(sorted_lengths, 95)

        return {"mean": mean, "std": std, "p50": p50, "p95": p95}


# ---------------------------------------------------------------------------
# compare_tokenizers
# ---------------------------------------------------------------------------

def compare_tokenizers(
    tokenize_a: Callable[[str], list[int]],
    tokenize_b: Callable[[str], list[int]],
    texts: list[str],
) -> dict[str, float]:
    """Compare two tokenizers over *texts*.

    Returns:
        fertility_a, fertility_b   – fertility for each tokenizer
        compression_a, compression_b – compression ratio for each tokenizer
        winner                     – "a" if compression_a > compression_b else "b"
    """
    fertility_a = compute_fertility(texts, tokenize_a)
    fertility_b = compute_fertility(texts, tokenize_b)
    compression_a = compute_compression_ratio(texts, tokenize_a)
    compression_b = compute_compression_ratio(texts, tokenize_b)
    winner = "a" if compression_a > compression_b else "b"
    return {
        "fertility_a": fertility_a,
        "fertility_b": fertility_b,
        "compression_a": compression_a,
        "compression_b": compression_b,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _distribution_stats(values: list[int | float]) -> dict[str, float]:
    """Return mean/std/max/min for a list of numeric values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "max": float(max(values)),
        "min": float(min(values)),
    }


def _percentile(sorted_values: list[int | float], p: float) -> float:
    """Return the p-th percentile of a pre-sorted list (linear interpolation)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return float(sorted_values[-1])
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
