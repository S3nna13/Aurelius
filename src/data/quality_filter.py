"""Data quality filtering and deduplication: heuristic filters, MinHash deduplication, and quality scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    min_length: int = 50
    max_length: int = 100000
    min_alpha_ratio: float = 0.6
    max_repetition_ratio: float = 0.3
    language: str = "en"
    dedup_threshold: float = 0.8
    n_minhash_perms: int = 64


# ---------------------------------------------------------------------------
# Standalone heuristic functions
# ---------------------------------------------------------------------------

def compute_alpha_ratio(text: str) -> float:
    """Fraction of alphabetic characters in text. Returns float in [0, 1]."""
    if not text:
        return 0.0
    n_alpha = sum(1 for ch in text if ch.isalpha())
    return n_alpha / len(text)


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of text covered by repeated n-grams.

    Count n-gram frequencies; sum counts of n-grams appearing >1 times divided
    by total n-grams. Returns float in [0, 1].
    """
    words = text.split()
    if len(words) < n:
        return 0.0

    ngram_counts: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    total = sum(ngram_counts.values())
    if total == 0:
        return 0.0

    repeated = sum(count for count in ngram_counts.values() if count > 1)
    return repeated / total


def compute_text_quality_score(text: str) -> float:
    """Composite quality score based on multiple heuristics.

    alpha_ratio (40%) + sentence_count_score (30%) + vocabulary_richness (30%)
    sentence_count_score: min(n_sentences / 5, 1.0)
    vocabulary_richness: unique_words / total_words (type-token ratio)
    Returns float in [0, 1].
    """
    if not text:
        return 0.0

    alpha_ratio = compute_alpha_ratio(text)

    # Sentence count: split on . ! ?
    sentences = re.split(r'[.!?]+', text)
    n_sentences = sum(1 for s in sentences if s.strip())
    sentence_count_score = min(n_sentences / 5.0, 1.0)

    # Vocabulary richness (type-token ratio)
    words = text.lower().split()
    if words:
        vocabulary_richness = len(set(words)) / len(words)
    else:
        vocabulary_richness = 0.0

    score = (
        0.4 * alpha_ratio
        + 0.3 * sentence_count_score
        + 0.3 * vocabulary_richness
    )
    return float(min(max(score, 0.0), 1.0))


# ---------------------------------------------------------------------------
# HeuristicFilter
# ---------------------------------------------------------------------------

class HeuristicFilter:
    """Apply length and quality heuristics."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def passes(self, text: str) -> tuple[bool, dict[str, bool]]:
        """Check length, alpha_ratio, and repetition_ratio.

        Returns (passes_all, {check_name: passed}).
        """
        cfg = self.config
        checks: dict[str, bool] = {}

        n = len(text)
        checks["length"] = cfg.min_length <= n <= cfg.max_length

        ar = compute_alpha_ratio(text)
        checks["alpha_ratio"] = ar >= cfg.min_alpha_ratio

        rr = compute_repetition_ratio(text)
        checks["repetition_ratio"] = rr <= cfg.max_repetition_ratio

        passes_all = all(checks.values())
        return passes_all, checks

    def filter_batch(self, texts: list[str]) -> list[str]:
        """Return texts that pass all checks."""
        return [t for t in texts if self.passes(t)[0]]


# ---------------------------------------------------------------------------
# MinHashSketch
# ---------------------------------------------------------------------------

class MinHashSketch:
    """MinHash signature for approximate deduplication."""

    _PRIME: int = (1 << 31) - 1  # 2^31 - 1

    def __init__(self, n_perms: int = 64, seed: int = 42) -> None:
        self.n_perms = n_perms
        rng = np.random.RandomState(seed)
        self._a = rng.randint(1, self._PRIME, size=n_perms, dtype=np.int64)
        self._b = rng.randint(0, self._PRIME, size=n_perms, dtype=np.int64)
        self._prime = self._PRIME

    def compute(self, text: str, shingle_size: int = 5) -> np.ndarray:
        """Tokenize text into character shingles and compute MinHash signature.

        Returns (n_perms,) int array.
        """
        # Character shingles
        shingles: set[int] = set()
        if len(text) >= shingle_size:
            for i in range(len(text) - shingle_size + 1):
                s = text[i : i + shingle_size]
                shingles.add(hash(s) & 0x7FFFFFFF)  # keep positive
        else:
            shingles.add(hash(text) & 0x7FFFFFFF)

        shingle_arr = np.array(list(shingles), dtype=np.int64)

        # For each hash function: min((a*x + b) % prime) over shingles
        # Shape: (n_perms, n_shingles)
        hashed = (
            self._a[:, None] * shingle_arr[None, :] + self._b[:, None]
        ) % self._prime

        signature = hashed.min(axis=1)
        return signature.astype(np.int64)

    def similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Jaccard similarity estimate: fraction of equal values. Returns float in [0, 1]."""
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0
        return float(np.mean(sig1 == sig2))


# ---------------------------------------------------------------------------
# DeduplicationIndex
# ---------------------------------------------------------------------------

class DeduplicationIndex:
    """Near-duplicate detection using MinHash."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self.hasher = MinHashSketch(n_perms=config.n_minhash_perms)
        self._signatures: list[tuple[str, np.ndarray]] = []

    def add(self, text_id: str, text: str) -> None:
        """Compute and store signature for text_id."""
        sig = self.hasher.compute(text)
        self._signatures.append((text_id, sig))

    def is_duplicate(self, text: str) -> bool:
        """Check if text is similar to any stored signature."""
        if not self._signatures:
            return False
        sig = self.hasher.compute(text)
        threshold = self.config.dedup_threshold
        for _, stored_sig in self._signatures:
            if self.hasher.similarity(sig, stored_sig) >= threshold:
                return True
        return False

    def deduplicate_batch(self, texts: list[str]) -> list[str]:
        """Remove near-duplicates, return unique texts."""
        unique: list[str] = []
        seen_sigs: list[np.ndarray] = []
        threshold = self.config.dedup_threshold

        for text in texts:
            sig = self.hasher.compute(text)
            is_dup = any(
                self.hasher.similarity(sig, s) >= threshold for s in seen_sigs
            )
            if not is_dup:
                unique.append(text)
                seen_sigs.append(sig)

        return unique


# ---------------------------------------------------------------------------
# DataQualityPipeline
# ---------------------------------------------------------------------------

class DataQualityPipeline:
    """Full pipeline: heuristic filter + deduplication."""

    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self.heuristic_filter = HeuristicFilter(config)
        self.dedup_index = DeduplicationIndex(config)

    def process(self, texts: list[str]) -> tuple[list[str], dict[str, int]]:
        """Apply heuristic filter, then dedup.

        Returns (kept_texts, {"n_original", "n_kept", "n_filtered", "n_deduped"}).
        """
        n_original = len(texts)

        filtered = self.heuristic_filter.filter_batch(texts)
        n_filtered = n_original - len(filtered)

        deduped = self.dedup_index.deduplicate_batch(filtered)
        n_deduped = len(filtered) - len(deduped)

        stats = {
            "n_original": n_original,
            "n_kept": len(deduped),
            "n_filtered": n_filtered,
            "n_deduped": n_deduped,
        }
        return deduped, stats

    def reset(self) -> None:
        """Clear the dedup index."""
        self.dedup_index = DeduplicationIndex(self.config)
