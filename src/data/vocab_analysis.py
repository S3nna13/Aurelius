"""Vocabulary analysis tools for understanding tokenizer behavior and
token frequency distributions.

Pure Python stdlib only — no PyTorch, HuggingFace, scipy, or sklearn.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VocabConfig:
    """Configuration for vocabulary analysis."""

    vocab_size: int = 50257
    min_freq: int = 1
    max_token_len: int = 20
    special_tokens: List[str] = field(
        default_factory=lambda: ["<pad>", "<eos>", "<bos>", "<unk>"]
    )


# ---------------------------------------------------------------------------
# Standalone analysis functions
# ---------------------------------------------------------------------------


def compute_token_frequencies(token_ids: List[int], vocab_size: int) -> List[int]:
    """Return a list of length *vocab_size* where index i = count of token i.

    Runs in O(n) time.  Token ids outside [0, vocab_size) are silently ignored.
    """
    freqs: List[int] = [0] * vocab_size
    for tid in token_ids:
        if 0 <= tid < vocab_size:
            freqs[tid] += 1
    return freqs


def compute_zipf_exponent(frequencies: List[int]) -> float:
    """Fit Zipf's law to *frequencies* and return the exponent alpha.

    Method: linear regression on log(rank) vs log(freq) for all non-zero
    frequencies.  Zipf's law states freq ~ rank^{-alpha}, so we regress
      log(freq) = -alpha * log(rank) + const
    and return alpha (typically ~1.0 for natural language).

    Raises ValueError if there are fewer than two non-zero frequencies.
    """
    # Sort descending so rank 1 = most frequent
    nonzero = sorted([f for f in frequencies if f > 0], reverse=True)
    if len(nonzero) < 2:
        raise ValueError("Need at least 2 non-zero frequencies to fit Zipf exponent.")

    log_ranks = [math.log(r + 1) for r in range(len(nonzero))]  # rank is 1-indexed
    log_freqs = [math.log(f) for f in nonzero]

    n = len(log_ranks)
    mean_lr = sum(log_ranks) / n
    mean_lf = sum(log_freqs) / n

    cov = sum((log_ranks[i] - mean_lr) * (log_freqs[i] - mean_lf) for i in range(n))
    var = sum((lr - mean_lr) ** 2 for lr in log_ranks)

    if var == 0.0:
        return 0.0

    slope = cov / var  # slope is -alpha
    return -slope  # return alpha (positive)


def find_rare_tokens(frequencies: List[int], min_freq: int) -> List[int]:
    """Return sorted list of token indices with 0 < freq < min_freq.

    These are tokens that appear in the corpus but are rare.
    """
    return sorted(i for i, f in enumerate(frequencies) if 0 < f < min_freq)


def find_dead_tokens(frequencies: List[int]) -> List[int]:
    """Return sorted list of token indices that never appeared (freq == 0)."""
    return [i for i, f in enumerate(frequencies) if f == 0]


def compute_token_length_distribution(vocab: Dict[int, str]) -> dict:
    """Analyse the string lengths of tokens in *vocab*.

    Returns a dict with keys:
      "mean"      – mean token length (float)
      "std"       – standard deviation (float, 0.0 for single-token vocabs)
      "max"       – maximum token length (int)
      "min"       – minimum token length (int)
      "histogram" – list where histogram[k] = count of tokens with length k
                    (0-indexed; length 0 at index 0, …, up to max length)
    """
    if not vocab:
        return {"mean": 0.0, "std": 0.0, "max": 0, "min": 0, "histogram": []}

    lengths = [len(tok) for tok in vocab.values()]
    max_len = max(lengths)
    min_len = min(lengths)
    mean_len = sum(lengths) / len(lengths)
    std_len = statistics.pstdev(lengths)  # population std (pure stdlib)

    histogram = [0] * (max_len + 1)
    for ln in lengths:
        histogram[ln] += 1

    return {
        "mean": mean_len,
        "std": std_len,
        "max": max_len,
        "min": min_len,
        "histogram": histogram,
    }


def compute_fertility(text_tokens: List[int], char_count: int) -> float:
    """Return characters-per-token ratio (char_count / len(text_tokens)).

    Returns 0.0 on division-by-zero (empty token list).
    """
    if not text_tokens:
        return 0.0
    return char_count / len(text_tokens)


def estimate_compression_ratio(text: str, token_ids: List[int]) -> float:
    """Return bytes-per-token ratio: len(text.encode()) / len(token_ids).

    Returns 0.0 when token_ids is empty.
    """
    if not token_ids:
        return 0.0
    return len(text.encode()) / len(token_ids)


# ---------------------------------------------------------------------------
# VocabAnalyzer class
# ---------------------------------------------------------------------------

# Prefixes that indicate sub-word / continuation tokens in common tokenizers.
_SUBWORD_PREFIXES = ("##", "\u0120", "\u010a")  # ##, Ġ, Ċ
_PUNCT_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


class VocabAnalyzer:
    """High-level vocabulary analysis over a {id: token_string} vocab dict."""

    def __init__(self, vocab: Dict[int, str], config: VocabConfig) -> None:
        self.vocab = vocab
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_corpus(self, token_ids: List[int]) -> dict:
        """Run all analyses on *token_ids* and return a summary dict.

        Keys returned:
          "frequencies"    – list of ints (length = vocab_size)
          "zipf_exponent"  – float (alpha) or None if not enough data
          "rare_tokens"    – list of int indices
          "dead_tokens"    – list of int indices
          "fertility"      – float (chars per token; uses token strings)
          "coverage"       – float, fraction of vocab seen at least once
        """
        freqs = compute_token_frequencies(token_ids, self.config.vocab_size)

        # Zipf exponent — may fail with too-small corpora
        try:
            zipf = compute_zipf_exponent(freqs)
        except ValueError:
            zipf = None

        rare = find_rare_tokens(freqs, self.config.min_freq)
        dead = find_dead_tokens(freqs)

        # Fertility: count characters contributed by the token strings
        total_chars = sum(len(self.vocab.get(tid, "")) for tid in token_ids)
        fertility = compute_fertility(token_ids, total_chars)

        n_seen = sum(1 for f in freqs if f > 0)
        coverage = n_seen / self.config.vocab_size if self.config.vocab_size else 0.0

        return {
            "frequencies": freqs,
            "zipf_exponent": zipf,
            "rare_tokens": rare,
            "dead_tokens": dead,
            "fertility": fertility,
            "coverage": coverage,
        }

    def find_similar_tokens(
        self, token_id: int, n: int = 5
    ) -> List[Tuple[int, float]]:
        """Find *n* vocab tokens whose string length is closest to *token_id*'s.

        Returns list of (id, length_diff) sorted by length_diff ascending,
        excluding *token_id* itself.
        """
        if token_id not in self.vocab:
            return []

        target_len = len(self.vocab[token_id])
        candidates: List[Tuple[int, float]] = []
        for tid, tok in self.vocab.items():
            if tid == token_id:
                continue
            diff = abs(len(tok) - target_len)
            candidates.append((tid, float(diff)))

        candidates.sort(key=lambda x: x[1])
        return candidates[:n]

    def get_subword_stats(self) -> dict:
        """Return heuristic subword statistics over the vocabulary.

        Keys:
          "n_prefix_tokens"     – tokens starting with ##, Ġ, or Ċ
          "n_whole_word_tokens" – tokens that are not prefixed sub-words and
                                  contain at least one alphabetic character
          "n_special_tokens"    – tokens that match config.special_tokens or
                                  are surrounded by angle-brackets (e.g. <pad>)
          "n_digit_tokens"      – tokens whose stripped form is fully numeric
          "n_punct_tokens"      – tokens composed entirely of punctuation chars
        """
        special_set = set(self.config.special_tokens)
        n_prefix = 0
        n_whole = 0
        n_special = 0
        n_digit = 0
        n_punct = 0

        for tok in self.vocab.values():
            stripped = tok.strip()

            # Special tokens
            if tok in special_set or (
                stripped.startswith("<") and stripped.endswith(">")
            ):
                n_special += 1
                continue

            # Sub-word prefix tokens
            if any(tok.startswith(p) for p in _SUBWORD_PREFIXES):
                n_prefix += 1
                continue

            # Digit tokens
            if stripped.isdigit():
                n_digit += 1
                continue

            # Punctuation-only tokens
            if stripped and all(c in _PUNCT_CHARS for c in stripped):
                n_punct += 1
                continue

            # Whole-word tokens (contains at least one alpha char)
            if any(c.isalpha() for c in tok):
                n_whole += 1

        return {
            "n_prefix_tokens": n_prefix,
            "n_whole_word_tokens": n_whole,
            "n_special_tokens": n_special,
            "n_digit_tokens": n_digit,
            "n_punct_tokens": n_punct,
        }
