"""CLM (Causal Language Model) evaluation metrics.

Measures generation quality beyond loss: diversity, repetition, and
factuality proxies — all implemented in pure PyTorch / stdlib.

References:
- Distinct-n: Li et al. 2016
- Self-BLEU: Zhu et al. 2018
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CLMMetricsConfig:
    """Configuration for CLM evaluation metrics."""
    ngram_sizes: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    repetition_window: int = 32
    distinct_n: int = 2
    length_penalty_alpha: float = 0.0


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[int], n: int) -> List[Tuple[int, ...]]:
    """Return list of n-gram tuples from a token list."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_ngram_frequencies(tokens: List[int], n: int) -> Dict[Tuple[int, ...], int]:
    """Return {ngram_tuple: count} frequency dict for all n-grams in *tokens*.

    Args:
        tokens: Flat list of integer token ids.
        n:      N-gram order (must be >= 1).

    Returns:
        Counter mapping each n-gram tuple to its frequency.
    """
    return dict(Counter(_ngrams(tokens, n)))


# ---------------------------------------------------------------------------
# Distinct-n
# ---------------------------------------------------------------------------

def compute_distinct_n(tokens: List[int], n: int) -> float:
    """Distinct n-grams / total n-grams — measures lexical diversity.

    Returns 0.0 when the sequence is too short to form any n-gram.

    Args:
        tokens: Flat list of integer token ids.
        n:      N-gram order.

    Returns:
        Float in [0, 1].  1.0 means every n-gram is unique.
    """
    grams = _ngrams(tokens, n)
    total = len(grams)
    if total == 0:
        return 0.0
    unique = len(set(grams))
    return unique / total


# ---------------------------------------------------------------------------
# Repetition rate
# ---------------------------------------------------------------------------

def compute_repetition_rate(tokens: List[int], window: int) -> float:
    """Fraction of tokens that already appeared in the preceding *window* tokens.

    For each position i the algorithm checks whether tokens[i] is in the
    look-back window tokens[max(0, i-window) : i].

    Args:
        tokens: Flat list of integer token ids.
        window: Number of previous tokens to look back.

    Returns:
        Float in [0, 1].  0.0 means no repetition.
    """
    if len(tokens) <= 1:
        return 0.0

    repeated = 0
    for i in range(1, len(tokens)):
        look_back = tokens[max(0, i - window) : i]
        if tokens[i] in look_back:
            repeated += 1

    return repeated / (len(tokens) - 1)


# ---------------------------------------------------------------------------
# Self-BLEU
# ---------------------------------------------------------------------------

def _ngram_precision(hypothesis: List[int], references: List[List[int]], n: int) -> float:
    """Clipped n-gram precision of *hypothesis* against *references*."""
    hyp_grams = Counter(_ngrams(hypothesis, n))
    if not hyp_grams:
        return 0.0

    # Build maximum reference counts
    ref_max: Counter = Counter()
    for ref in references:
        ref_grams = Counter(_ngrams(ref, n))
        for gram, cnt in ref_grams.items():
            ref_max[gram] = max(ref_max[gram], cnt)

    clipped = sum(min(cnt, ref_max[gram]) for gram, cnt in hyp_grams.items())
    return clipped / sum(hyp_grams.values())


def _brevity_penalty(hyp_len: int, ref_len: float) -> float:
    """Standard BLEU brevity penalty."""
    if hyp_len >= ref_len:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1.0 - ref_len / hyp_len)


def _bleu_single(hypothesis: List[int], references: List[List[int]], n: int) -> float:
    """Sentence-level BLEU up to order *n* using geometric mean of precisions."""
    if not hypothesis or not references:
        return 0.0

    # Geometric mean of n-gram precisions for orders 1..n
    log_avg = 0.0
    for order in range(1, n + 1):
        p = _ngram_precision(hypothesis, references, order)
        if p == 0.0:
            return 0.0
        log_avg += math.log(p)
    log_avg /= n

    # Brevity penalty: closest reference length
    ref_len = min(len(r) for r in references)
    bp = _brevity_penalty(len(hypothesis), ref_len)
    return bp * math.exp(log_avg)


def compute_self_bleu(sequences: List[List[int]], n: int = 4) -> float:
    """Mean BLEU of each sequence against the remaining sequences as references.

    Higher self-BLEU indicates more repetitive / less diverse output.

    Args:
        sequences: List of token-id sequences.
        n:         Maximum n-gram order for BLEU.

    Returns:
        Float in [0, 1].  Returns 0.0 when fewer than 2 sequences are given.
    """
    if len(sequences) < 2:
        return 0.0

    scores: List[float] = []
    for i, hyp in enumerate(sequences):
        refs = [seq for j, seq in enumerate(sequences) if j != i]
        scores.append(_bleu_single(hyp, refs, n))

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Length statistics
# ---------------------------------------------------------------------------

def compute_length_stats(sequences: List[List[int]]) -> Dict[str, float]:
    """Compute length statistics over a list of sequences.

    Args:
        sequences: List of token-id sequences.

    Returns:
        dict with keys "mean", "std", "min", "max", "median".
    """
    if not sequences:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "median": 0.0}

    lengths = [len(s) for s in sequences]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / n
    std = math.sqrt(variance)
    sorted_lengths = sorted(lengths)
    if n % 2 == 1:
        median = float(sorted_lengths[n // 2])
    else:
        median = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2.0

    return {
        "mean": mean,
        "std": std,
        "min": min(lengths),
        "max": max(lengths),
        "median": median,
    }


# ---------------------------------------------------------------------------
# Vocabulary coverage
# ---------------------------------------------------------------------------

def compute_vocabulary_coverage(tokens: List[int], vocab_size: int) -> float:
    """Fraction of the vocabulary that appears at least once in *tokens*.

    Args:
        tokens:     Flat list of integer token ids.
        vocab_size: Total vocabulary size.

    Returns:
        Float in (0, 1].  Returns 0.0 for an empty token list.
    """
    if not tokens or vocab_size <= 0:
        return 0.0
    unique = len(set(tokens))
    return unique / vocab_size


# ---------------------------------------------------------------------------
# GenerationMetrics
# ---------------------------------------------------------------------------

class GenerationMetrics:
    """High-level evaluation wrapper that aggregates all CLM metrics."""

    def __init__(self, config: CLMMetricsConfig) -> None:
        self.config = config

    def evaluate(self, sequences: List[List[int]], vocab_size: int) -> Dict[str, float]:
        """Compute all metrics for *sequences*.

        Args:
            sequences:  List of generated token-id sequences.
            vocab_size: Vocabulary size for coverage computation.

        Returns:
            dict with keys:
                "distinct_1", "distinct_2", "distinct_3", "distinct_4",
                "repetition_rate", "self_bleu",
                "length_mean", "length_std", "vocab_coverage"
        """
        # Pool all tokens for sequence-level metrics
        all_tokens: List[int] = [tok for seq in sequences for tok in seq]

        results: Dict[str, float] = {}

        # Distinct-n for orders 1..4
        for k in [1, 2, 3, 4]:
            results[f"distinct_{k}"] = compute_distinct_n(all_tokens, k)

        results["repetition_rate"] = compute_repetition_rate(
            all_tokens, self.config.repetition_window
        )
        results["self_bleu"] = compute_self_bleu(sequences, n=4)

        length_stats = compute_length_stats(sequences)
        results["length_mean"] = length_stats["mean"]
        results["length_std"] = length_stats["std"]

        results["vocab_coverage"] = compute_vocabulary_coverage(all_tokens, vocab_size)

        return results

    def compare(
        self, metrics_a: Dict[str, float], metrics_b: Dict[str, float]
    ) -> Dict[str, float]:
        """Return {key: b - a} differences for every key present in *metrics_a*.

        Args:
            metrics_a: Baseline metric dict (e.g., from evaluate()).
            metrics_b: Comparison metric dict.

        Returns:
            Dict mapping each key to the signed difference (b minus a).
        """
        return {key: metrics_b[key] - metrics_a[key] for key in metrics_a}
