"""
preprocessing.py — Text preprocessing utilities for the Aurelius LLM project.

Pure Python stdlib only (no PyTorch, no HuggingFace, no nltk).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    min_length: int = 10
    max_length: int = 2048
    dedup_threshold: float = 0.9
    filter_non_utf8: bool = True
    normalize_whitespace: bool = True
    remove_html: bool = True


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def normalize_whitespace_fn(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into single spaces; strip edges."""
    # Replace all whitespace sequences (spaces, tabs, newlines, etc.) with a single space
    text = re.sub(r"[ \t\r\n\f\v]+", " ", text)
    return text.strip()


def remove_html_tags(text: str) -> str:
    """Remove <tag> and </tag> patterns; preserve content between tags."""
    return re.sub(r"<[^>]+>", "", text)


def filter_by_length(texts: List[str], min_len: int, max_len: int) -> List[str]:
    """Keep texts whose character count is within [min_len, max_len] inclusive."""
    return [t for t in texts if min_len <= len(t) <= max_len]


def is_valid_utf8(text: str) -> bool:
    """
    Python str is always valid Unicode, so we check for corruption markers.
    Returns False if the text contains a significant proportion of Unicode
    replacement characters (\\ufffd), indicating garbled data.
    """
    if not text:
        return True
    replacement_count = text.count("\ufffd")
    ratio = replacement_count / len(text)
    # Treat as invalid if >5% of characters are replacement chars
    return ratio <= 0.05


def compute_text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of the UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Exact deduplication
# ---------------------------------------------------------------------------

class ExactDeduplicator:
    """Deduplicates texts by exact SHA-256 hash comparison."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def add_and_check(self, text: str) -> bool:
        """Return True if text is NEW (not seen before) and add it; False if duplicate."""
        h = compute_text_hash(text)
        if h in self._seen:
            return False
        self._seen.add(h)
        return True

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Return deduplicated list preserving original order."""
        seen: set[str] = set()
        result: List[str] = []
        for t in texts:
            h = compute_text_hash(t)
            if h not in seen:
                seen.add(h)
                result.append(t)
        return result

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# MinHash approximate deduplication
# ---------------------------------------------------------------------------

class MinHashDeduplicator:
    """Approximate near-duplicate detection via MinHash on character tri-grams."""

    def __init__(
        self,
        n_hashes: int = 128,
        threshold: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.n_hashes = n_hashes
        self.threshold = threshold
        self.seed = seed
        self._signatures: List[List[int]] = []

    def _compute_minhash(self, text: str) -> List[int]:
        """Compute n_hashes MinHash signatures using character 3-grams."""
        ngrams: set[str] = set()
        n = 3
        if len(text) < n:
            # Fall back to unigrams if text is very short
            ngrams = set(text)
        else:
            for i in range(len(text) - n + 1):
                ngrams.add(text[i : i + n])

        if not ngrams:
            # Empty text → all zeros
            return [0] * self.n_hashes

        sig: List[int] = []
        mod = 2 ** 32
        for i in range(self.n_hashes):
            min_val = mod  # sentinel larger than any hash % mod
            for gram in ngrams:
                h = hash(gram + str(i) + str(self.seed)) % mod
                if h < min_val:
                    min_val = h
            sig.append(min_val)
        return sig

    def is_near_duplicate(self, text: str, candidate_sig: List[int]) -> bool:
        """
        Estimate Jaccard similarity via MinHash: fraction of matching hash values.
        Returns True if estimated similarity >= threshold.
        """
        sig = self._compute_minhash(text)
        matches = sum(a == b for a, b in zip(sig, candidate_sig))
        estimated_jaccard = matches / self.n_hashes
        return estimated_jaccard >= self.threshold

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Streaming dedup: keep first occurrence, drop near-duplicates."""
        kept_sigs: List[List[int]] = []
        result: List[str] = []

        for text in texts:
            sig = self._compute_minhash(text)
            is_dup = False
            for kept_sig in kept_sigs:
                matches = sum(a == b for a, b in zip(sig, kept_sig))
                if matches / self.n_hashes >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept_sigs.append(sig)
                result.append(text)

        self._signatures = kept_sigs
        return result


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """Apply a configured sequence of cleaning steps to texts."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    def clean(self, text: str) -> str:
        """Apply configured cleaning steps in order."""
        if self.config.remove_html:
            text = remove_html_tags(text)
        if self.config.normalize_whitespace:
            text = normalize_whitespace_fn(text)
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Apply clean() to each text, then filter by length and (optionally)
        UTF-8 validity.  Returns only texts that pass all filters.
        """
        cleaned: List[str] = []
        for t in texts:
            c = self.clean(t)
            # Length filter
            if not (self.config.min_length <= len(c) <= self.config.max_length):
                continue
            # UTF-8 / replacement-char filter
            if self.config.filter_non_utf8 and not is_valid_utf8(c):
                continue
            cleaned.append(c)
        return cleaned

    def get_stats(
        self, original: List[str], cleaned: List[str]
    ) -> Dict[str, float]:
        """Return summary statistics comparing original vs cleaned lists."""
        n_original = len(original)
        n_kept = len(cleaned)
        filter_rate = 1.0 - (n_kept / n_original) if n_original > 0 else 0.0
        mean_length_before = (
            sum(len(t) for t in original) / n_original if n_original > 0 else 0.0
        )
        mean_length_after = (
            sum(len(t) for t in cleaned) / n_kept if n_kept > 0 else 0.0
        )
        return {
            "n_original": float(n_original),
            "n_kept": float(n_kept),
            "filter_rate": filter_rate,
            "mean_length_before": mean_length_before,
            "mean_length_after": mean_length_after,
        }
