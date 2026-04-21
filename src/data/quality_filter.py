"""Pretraining data quality filter — perplexity proxy, n-gram dedup, and heuristic rules.

Inspired by Dolma, RedPajama, and FineWeb quality filtering pipelines.
Pure Python builtins for string processing; no external ML libraries.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QualityFilterConfig:
    # Length filters
    min_chars: int = 100
    max_chars: int = 100_000
    min_words: int = 20

    # Perplexity proxy (character entropy)
    min_char_entropy: float = 3.5   # bits; below = repetitive
    max_char_entropy: float = 6.5   # bits; above = random/noise

    # N-gram dedup
    ngram_n: int = 5
    min_unique_ngram_ratio: float = 0.2   # below = too repetitive

    # Heuristic rules
    min_alpha_ratio: float = 0.5          # fraction of alpha chars
    max_bullet_ratio: float = 0.9         # fraction of lines starting with bullet
    max_ellipsis_ratio: float = 0.1       # fraction of lines ending with "..."


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)   # failure reasons; empty = passed
    stats: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QualityFilter
# ---------------------------------------------------------------------------

# Regex for bullet-point line starts: -, *, •, or digit followed by ./)
_BULLET_RE = re.compile(r"^\s*([-*\u2022]|\d+[.)]\s)")


class QualityFilter:
    """Three-pass quality filter for pretraining text corpora."""

    def __init__(self, config: QualityFilterConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def char_entropy(self, text: str) -> float:
        """Character-level Shannon entropy in bits.

        H = -sum(p_c * log2(p_c)) for each unique character c.
        Returns 0.0 for empty text.
        """
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def unique_ngram_ratio(self, text: str) -> float:
        """Fraction of unique word n-grams out of all n-grams.

        Tokenises on whitespace. Returns 1.0 when there are fewer words than n
        (no repeatable ngram window), and 0.0 for empty text.
        """
        cfg = self.config
        words = text.split()
        n = cfg.ngram_n
        if not words:
            return 0.0
        if len(words) < n:
            return 1.0  # can't form any n-gram → vacuously all unique
        ngrams = [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        return unique / max(total, 1)

    def alpha_ratio(self, text: str) -> float:
        """Fraction of characters that are alphabetic. Returns 0.0 for empty text."""
        if not text:
            return 0.0
        n_alpha = sum(1 for ch in text if ch.isalpha())
        return n_alpha / len(text)

    def bullet_ratio(self, text: str) -> float:
        """Fraction of lines that start with a bullet character (-, *, •, digit.)."""
        lines = text.splitlines()
        if not lines:
            return 0.0
        bullet_count = sum(1 for line in lines if _BULLET_RE.match(line))
        return bullet_count / len(lines)

    def ellipsis_ratio(self, text: str) -> float:
        """Fraction of lines ending with '...'."""
        lines = text.splitlines()
        if not lines:
            return 0.0
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("..."))
        return ellipsis_count / len(lines)

    def word_count(self, text: str) -> int:
        """Word count by whitespace splitting."""
        return len(text.split())

    # ------------------------------------------------------------------
    # Core filter
    # ------------------------------------------------------------------

    def filter(self, text: str) -> FilterResult:  # noqa: A003
        """Run all quality checks on a single text.

        Returns a FilterResult with passed=True only when all checks pass.
        Failure reasons are collected even after the first failure so the
        caller can see the full picture.
        """
        cfg = self.config
        reasons: list[str] = []

        # --- Length checks ---
        n_chars = len(text)
        if n_chars < cfg.min_chars:
            reasons.append(f"too_short_chars:{n_chars}<{cfg.min_chars}")
        if n_chars > cfg.max_chars:
            reasons.append(f"too_long_chars:{n_chars}>{cfg.max_chars}")

        n_words = self.word_count(text)
        if n_words < cfg.min_words:
            reasons.append(f"too_few_words:{n_words}<{cfg.min_words}")

        # --- Perplexity proxy (character entropy) ---
        entropy = self.char_entropy(text)
        if entropy < cfg.min_char_entropy:
            reasons.append(f"low_entropy:{entropy:.3f}<{cfg.min_char_entropy}")
        if entropy > cfg.max_char_entropy:
            reasons.append(f"high_entropy:{entropy:.3f}>{cfg.max_char_entropy}")

        # --- N-gram dedup ---
        unique_ngram = self.unique_ngram_ratio(text)
        if unique_ngram < cfg.min_unique_ngram_ratio:
            reasons.append(
                f"low_unique_ngram_ratio:{unique_ngram:.3f}<{cfg.min_unique_ngram_ratio}"
            )

        # --- Heuristic rules ---
        a_ratio = self.alpha_ratio(text)
        if a_ratio < cfg.min_alpha_ratio:
            reasons.append(f"low_alpha_ratio:{a_ratio:.3f}<{cfg.min_alpha_ratio}")

        b_ratio = self.bullet_ratio(text)
        if b_ratio > cfg.max_bullet_ratio:
            reasons.append(f"high_bullet_ratio:{b_ratio:.3f}>{cfg.max_bullet_ratio}")

        e_ratio = self.ellipsis_ratio(text)
        if e_ratio > cfg.max_ellipsis_ratio:
            reasons.append(f"high_ellipsis_ratio:{e_ratio:.3f}>{cfg.max_ellipsis_ratio}")

        stats: dict[str, float] = {
            "char_count": float(n_chars),
            "word_count": float(n_words),
            "char_entropy": entropy,
            "unique_ngram_ratio": unique_ngram,
            "alpha_ratio": a_ratio,
            "bullet_ratio": b_ratio,
            "ellipsis_ratio": e_ratio,
        }

        return FilterResult(passed=len(reasons) == 0, reasons=reasons, stats=stats)

    def filter_batch(self, texts: list[str]) -> list[FilterResult]:
        """Apply filter to each text in the list. Returns one FilterResult per text."""
        return [self.filter(t) for t in texts]

    def statistics(self, results: list[FilterResult]) -> dict[str, float]:
        """Aggregate statistics across a list of FilterResults.

        Returns pass_rate, mean_char_entropy, mean_unique_ngram_ratio,
        mean_alpha_ratio.
        """
        if not results:
            return {
                "pass_rate": 0.0,
                "mean_char_entropy": 0.0,
                "mean_unique_ngram_ratio": 0.0,
                "mean_alpha_ratio": 0.0,
            }

        n = len(results)
        pass_count = sum(1 for r in results if r.passed)
        mean_entropy = sum(r.stats.get("char_entropy", 0.0) for r in results) / n
        mean_ngram = sum(r.stats.get("unique_ngram_ratio", 0.0) for r in results) / n
        mean_alpha = sum(r.stats.get("alpha_ratio", 0.0) for r in results) / n

        return {
            "pass_rate": pass_count / n,
            "mean_char_entropy": mean_entropy,
            "mean_unique_ngram_ratio": mean_ngram,
            "mean_alpha_ratio": mean_alpha,
        }


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

from src.data import DATA_REGISTRY  # noqa: E402  (import after class definition)

DATA_REGISTRY["quality_filter"] = QualityFilter
