"""Heuristic context-quality scorer for compaction decisions.

Scores individual text chunks on density, query relevance, structure,
and redundancy.  No external models — pure stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4
_OPTIMAL_TOKENS = 512
_OPTIMAL_CHARS = _OPTIMAL_TOKENS * _CHARS_PER_TOKEN

_WEIGHT_DENSITY = 0.4
_WEIGHT_OVERLAP = 0.4
_WEIGHT_STRUCTURE = 0.2

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _token_density(chunk: str) -> float:
    estimated = len(chunk) / _CHARS_PER_TOKEN
    return min(estimated, _OPTIMAL_TOKENS) / _OPTIMAL_TOKENS


def _query_overlap(chunk: str, query: str) -> float:
    chunk_words = set(_words(chunk))
    query_words = set(_words(query))
    if not chunk_words and not query_words:
        return 0.0
    shared = chunk_words & query_words
    all_unique = chunk_words | query_words
    return len(shared) / len(all_unique)


def _structure_bonus(chunk: str) -> float:
    bonus = 0.0
    if "```" in chunk:
        bonus += 0.10
    if re.search(r"(?m)^\s*[-*]\s", chunk) or re.search(r"(?m)^\s*\d+\.\s", chunk):
        bonus += 0.05
    if "|" in chunk and "\n" in chunk:
        bonus += 0.05
    return min(bonus, 0.2)


def _redundancy_factor(chunk: str) -> float:
    words = _words(chunk)
    total = len(words)
    if total == 0:
        return 1.0
    unique = len(set(words))
    uniqueness = unique / total
    return 0.5 + 0.5 * uniqueness


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompactionRecommendation:
    """Result of a compaction recommendation pass."""

    keep_indices: list[int]
    drop_indices: list[int]
    avg_score: float


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ContextQualityScorer:
    """Heuristic scorer for context chunk quality.

    Combines four signals:

    * **Token density** — longer chunks score higher, saturating at
      ~512 tokens (estimated as 4 characters per token).
    * **Query overlap** — Jaccard-like shared-unique word ratio when a
      query is supplied.
    * **Structure bonus** — small boost for code blocks, lists, or tables.
    * **Redundancy penalty** — high repetition of the same words drags
      the score down via a multiplicative factor.

    All methods validate inputs and raise ``TypeError`` or ``ValueError``
    on bad data.
    """

    def score_chunk(self, chunk: str, query: str | None = None) -> float:
        """Return a quality score in ``[0.0, 1.0]`` for *chunk*."""
        if not isinstance(chunk, str):
            raise TypeError(f"chunk must be str, got {type(chunk).__name__}")
        if query is not None and not isinstance(query, str):
            raise TypeError(f"query must be str or None, got {type(query).__name__}")

        if not chunk.strip():
            return 0.0

        density = _token_density(chunk)
        overlap = _query_overlap(chunk, query) if query is not None else 0.0
        structure = _structure_bonus(chunk)
        penalty = _redundancy_factor(chunk)

        raw = density * _WEIGHT_DENSITY + overlap * _WEIGHT_OVERLAP + structure * _WEIGHT_STRUCTURE
        return min(raw * penalty, 1.0)

    def score_chunks(
        self,
        chunks: list[str],
        query: str | None = None,
    ) -> list[float]:
        """Score a batch of chunks, preserving order."""
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be list[str], got {type(chunks).__name__}")
        for i, c in enumerate(chunks):
            if not isinstance(c, str):
                raise TypeError(f"chunks[{i}] must be str, got {type(c).__name__}")
        return [self.score_chunk(c, query) for c in chunks]

    def rank_chunks(
        self,
        chunks: list[str],
        query: str | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` tuples sorted descending by score."""
        scores = self.score_chunks(chunks, query)
        indexed: list[tuple[int, float]] = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed

    def recommend(
        self,
        chunks: list[str],
        threshold: float = 0.3,
    ) -> CompactionRecommendation:
        """Partition *chunks* into *keep* and *drop* sets based on *threshold*."""
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be list[str], got {type(chunks).__name__}")
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        scores = self.score_chunks(chunks)
        keep_indices = [i for i, s in enumerate(scores) if s >= threshold]
        drop_indices = [i for i, s in enumerate(scores) if s < threshold]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return CompactionRecommendation(
            keep_indices=keep_indices,
            drop_indices=drop_indices,
            avg_score=avg_score,
        )
