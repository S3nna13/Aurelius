"""Training data quality filtering and near-deduplication utilities.

Implements heuristics from C4, CC-Net, Gopher, and RedPajama pipelines.
Pure Python — no NLTK, langdetect, or datasketch required.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------

@dataclass
class QualityFilterConfig:
    min_chars: int = 200
    max_chars: int = 100_000
    min_words: int = 50
    max_word_repeat_ratio: float = 0.2
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 15.0
    max_symbol_to_word_ratio: float = 0.1
    min_unique_word_ratio: float = 0.1
    lang: str = "en"  # placeholder — no external model used


@dataclass
class FilterResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Symbol set (string.punctuation + extra unicode currency / trademark chars)
# ---------------------------------------------------------------------------

_SYMBOL_CHARS: frozenset[str] = frozenset(string.punctuation + "©®™€£¥")


# ---------------------------------------------------------------------------
# TextQualityFilter
# ---------------------------------------------------------------------------

class TextQualityFilter:
    """Apply quality heuristics to filter low-quality text.

    Based on heuristics from C4, CC-Net, Gopher, and RedPajama.
    """

    def __init__(self, config: QualityFilterConfig | None = None) -> None:
        self.config = config or QualityFilterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, text: str) -> FilterResult:
        """Apply all quality checks and return a FilterResult."""
        cfg = self.config
        reasons: list[str] = []

        # --- basic tokenisation ----------------------------------------
        n_chars = len(text)
        words = text.split()
        n_words = len(words)

        # 1. Length checks -----------------------------------------------
        if n_chars < cfg.min_chars:
            reasons.append(f"too_short_chars:{n_chars}<{cfg.min_chars}")
        if n_chars > cfg.max_chars:
            reasons.append(f"too_long_chars:{n_chars}>{cfg.max_chars}")
        if n_words < cfg.min_words:
            reasons.append(f"too_few_words:{n_words}<{cfg.min_words}")

        # Avoid division-by-zero for the remaining checks
        if n_words == 0:
            stats = {
                "n_chars": n_chars,
                "n_words": 0,
                "word_repeat_ratio": 0.0,
                "mean_word_len": 0.0,
                "symbol_to_word_ratio": 0.0,
                "unique_word_ratio": 0.0,
            }
            return FilterResult(passed=False, reasons=reasons, stats=stats)

        # 2. Word repetition ratio ----------------------------------------
        # spec formula: (len(words) - len(set(words))) / len(words)
        word_repeat_ratio = (n_words - len(set(words))) / n_words
        if word_repeat_ratio > cfg.max_word_repeat_ratio:
            reasons.append(
                f"high_word_repeat_ratio:{word_repeat_ratio:.3f}>{cfg.max_word_repeat_ratio}"
            )

        # 3. Mean word length ---------------------------------------------
        mean_word_len = sum(len(w) for w in words) / n_words
        if mean_word_len < cfg.min_mean_word_len:
            reasons.append(
                f"mean_word_len_too_short:{mean_word_len:.2f}<{cfg.min_mean_word_len}"
            )
        if mean_word_len > cfg.max_mean_word_len:
            reasons.append(
                f"mean_word_len_too_long:{mean_word_len:.2f}>{cfg.max_mean_word_len}"
            )

        # 4. Symbol-to-word ratio -----------------------------------------
        n_symbols = sum(1 for ch in text if ch in _SYMBOL_CHARS)
        symbol_to_word_ratio = n_symbols / n_words
        if symbol_to_word_ratio > cfg.max_symbol_to_word_ratio:
            reasons.append(
                f"high_symbol_to_word_ratio:{symbol_to_word_ratio:.3f}>{cfg.max_symbol_to_word_ratio}"
            )

        # 5. Unique word ratio --------------------------------------------
        unique_word_ratio = len(set(words)) / n_words
        if unique_word_ratio < cfg.min_unique_word_ratio:
            reasons.append(
                f"low_unique_word_ratio:{unique_word_ratio:.3f}<{cfg.min_unique_word_ratio}"
            )

        stats = {
            "n_chars": n_chars,
            "n_words": n_words,
            "word_repeat_ratio": word_repeat_ratio,
            "mean_word_len": mean_word_len,
            "symbol_to_word_ratio": symbol_to_word_ratio,
            "unique_word_ratio": unique_word_ratio,
        }

        return FilterResult(passed=len(reasons) == 0, reasons=reasons, stats=stats)

    def filter_batch(self, texts: list[str]) -> list[FilterResult]:
        """Filter a batch of texts. Returns one FilterResult per text."""
        return [self.filter(t) for t in texts]

    def stats(self, texts: list[str]) -> dict:
        """Return aggregate statistics about a corpus.

        Returns:
            {
                'n_total': int,
                'n_passed': int,
                'pass_rate': float,
                'rejection_reasons': dict[str, int],  # reason key → count
            }
        """
        results = self.filter_batch(texts)
        n_total = len(results)
        n_passed = sum(1 for r in results if r.passed)
        rejection_reasons: dict[str, int] = {}
        for r in results:
            for reason in r.reasons:
                # Use the reason prefix as the key (everything before the first ':')
                key = reason.split(":")[0]
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
        return {
            "n_total": n_total,
            "n_passed": n_passed,
            "pass_rate": n_passed / n_total if n_total > 0 else 0.0,
            "rejection_reasons": rejection_reasons,
        }


# ---------------------------------------------------------------------------
# N-gram hashing helpers
# ---------------------------------------------------------------------------

def compute_ngram_hashes(text: str, n: int = 5) -> set[int]:
    """Compute a set of n-gram hashes for MinHash deduplication.

    Tokenises by whitespace split, forms n-grams as space-joined strings,
    and hashes each using a simple polynomial hash:
        h = sum(ord(c) * 31**i for i, c in enumerate(ngram_str))
    """
    words = text.split()
    if len(words) < n:
        # Return a hash of the whole text if it is shorter than n words
        ngram_str = " ".join(words)
        return {_poly_hash(ngram_str)}

    hashes: set[int] = set()
    for i in range(len(words) - n + 1):
        ngram_str = " ".join(words[i : i + n])
        hashes.add(_poly_hash(ngram_str))
    return hashes


def _poly_hash(s: str) -> int:
    """Polynomial hash: sum(ord(c) * 31^i for i, c in enumerate(s))."""
    h = 0
    power = 1
    for ch in s:
        h += ord(ch) * power
        power *= 31
    return h


# ---------------------------------------------------------------------------
# MinHash similarity
# ---------------------------------------------------------------------------

class MinHashSimilarity:
    """MinHash-based near-duplicate detection.

    Uses k universal hash functions to estimate Jaccard similarity between
    documents.  Signature is a (k,) list of minimum hash values.

    Args:
        n_hashes: number of hash functions (default 128)
        n_gram:   n-gram size for shingling (default 5)
        threshold: Jaccard threshold for "duplicate" (default 0.8)
    """

    def __init__(
        self,
        n_hashes: int = 128,
        n_gram: int = 5,
        threshold: float = 0.8,
    ) -> None:
        self.n_hashes = n_hashes
        self.n_gram = n_gram
        self.threshold = threshold

        rng = random.Random(42)
        self._large_prime: int = (1 << 61) - 1
        self._a: list[int] = [
            rng.randint(1, self._large_prime - 1) for _ in range(n_hashes)
        ]
        self._b: list[int] = [
            rng.randint(0, self._large_prime - 1) for _ in range(n_hashes)
        ]

    # ------------------------------------------------------------------

    def signature(self, text: str) -> list[int]:
        """Compute MinHash signature for *text*.

        signature[i] = min over all n-grams x: (a[i]*hash(x) + b[i]) % large_prime
        """
        ngram_hashes = compute_ngram_hashes(text, self.n_gram)
        if not ngram_hashes:
            # Edge case: empty text — return max values
            return [self._large_prime] * self.n_hashes

        p = self._large_prime
        sig: list[int] = []
        for i in range(self.n_hashes):
            a = self._a[i]
            b = self._b[i]
            min_val = min((a * h + b) % p for h in ngram_hashes)
            sig.append(min_val)
        return sig

    def jaccard_estimate(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures.

        J ≈ fraction of positions where sig1[i] == sig2[i]
        """
        if not sig1 or not sig2:
            return 0.0
        matches = sum(a == b for a, b in zip(sig1, sig2))
        return matches / len(sig1)

    def is_near_duplicate(self, sig1: list[int], sig2: list[int]) -> bool:
        """Return True if estimated Jaccard similarity exceeds the threshold."""
        return self.jaccard_estimate(sig1, sig2) >= self.threshold


# ---------------------------------------------------------------------------
# Deduplication pipeline
# ---------------------------------------------------------------------------

class DeduplicationPipeline:
    """Deduplicate a corpus using MinHash.

    Uses a simple O(n²) pairwise comparison suitable for corpora < ~100k docs.
    For large scale, band-based LSH grouping should be applied.

    Args:
        minhash: MinHashSimilarity instance (created with defaults if None)
    """

    def __init__(self, minhash: MinHashSimilarity | None = None) -> None:
        self.minhash = minhash or MinHashSimilarity()

    # ------------------------------------------------------------------

    def deduplicate(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Remove near-duplicate documents from *texts*.

        For each text in order, keep it unless it is a near-duplicate of an
        already-kept text.

        Returns:
            (deduplicated_texts, kept_indices)
        """
        kept_texts: list[str] = []
        kept_indices: list[int] = []
        kept_sigs: list[list[int]] = []

        for idx, text in enumerate(texts):
            sig = self.minhash.signature(text)
            is_dup = any(
                self.minhash.is_near_duplicate(sig, ks) for ks in kept_sigs
            )
            if not is_dup:
                kept_texts.append(text)
                kept_indices.append(idx)
                kept_sigs.append(sig)

        return kept_texts, kept_indices

    def duplicate_stats(self, texts: list[str]) -> dict:
        """Return deduplication statistics without modifying *texts*.

        Returns:
            {
                'n_total': int,
                'n_unique': int,
                'duplicate_rate': float,
                'mean_cluster_size': float,
            }
        """
        n_total = len(texts)
        if n_total == 0:
            return {
                "n_total": 0,
                "n_unique": 0,
                "duplicate_rate": 0.0,
                "mean_cluster_size": 0.0,
            }

        sigs = [self.minhash.signature(t) for t in texts]

        # Union-Find to cluster near-duplicates
        parent = list(range(n_total))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n_total):
            for j in range(i + 1, n_total):
                if self.minhash.is_near_duplicate(sigs[i], sigs[j]):
                    union(i, j)

        # Count unique clusters
        cluster_roots: dict[int, int] = {}  # root → size
        for i in range(n_total):
            root = find(i)
            cluster_roots[root] = cluster_roots.get(root, 0) + 1

        n_unique = len(cluster_roots)
        duplicate_rate = 1.0 - n_unique / n_total
        mean_cluster_size = n_total / n_unique if n_unique > 0 else 0.0

        return {
            "n_total": n_total,
            "n_unique": n_unique,
            "duplicate_rate": duplicate_rate,
            "mean_cluster_size": mean_cluster_size,
        }
