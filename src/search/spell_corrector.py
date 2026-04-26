"""Aurelius search – edit-distance based spell correction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Correction:
    """Result of a spell-correction attempt for a single word."""

    original: str
    corrected: str
    distance: int
    confidence: float


# 100 common English words used as the default dictionary.
_DEFAULT_DICTIONARY: list[str] = [
    # Provided by spec (38 words)
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "any",
    "can",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "man",
    "new",
    "now",
    "old",
    "see",
    "two",
    "way",
    "who",
    "boy",
    "did",
    "its",
    "let",
    "put",
    "say",
    "she",
    "too",
    "use",
    # 62 additional common English words
    "about",
    "above",
    "after",
    "again",
    "also",
    "back",
    "been",
    "both",
    "call",
    "came",
    "come",
    "could",
    "down",
    "each",
    "even",
    "find",
    "first",
    "from",
    "give",
    "good",
    "hand",
    "have",
    "here",
    "high",
    "home",
    "into",
    "just",
    "keep",
    "know",
    "large",
    "last",
    "like",
    "long",
    "look",
    "made",
    "make",
    "many",
    "more",
    "most",
    "move",
    "much",
    "must",
    "name",
    "need",
    "next",
    "only",
    "open",
    "over",
    "part",
    "people",
    "place",
    "same",
    "show",
    "side",
    "some",
    "such",
    "take",
    "than",
    "that",
    "them",
    "then",
    "there",
    "they",
    "this",
    "time",
    "turn",
    "under",
    "very",
    "want",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "word",
    "work",
    "would",
    "year",
    "your",
]


class SpellCorrector:
    """Simple edit-distance spell corrector backed by a word dictionary."""

    def __init__(self, dictionary: list[str] | None = None) -> None:
        self._dictionary: list[str] = list(
            dictionary if dictionary is not None else _DEFAULT_DICTIONARY
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _edit_distance(self, a: str, b: str) -> int:
        """Standard DP Levenshtein distance between strings *a* and *b*."""
        m, n = len(a), len(b)
        # dp[i][j] = edit distance between a[:i] and b[:j]
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def _confidence(self, word: str, distance: int) -> float:
        """Compute confidence score for a correction."""
        if distance == 0:
            return 1.0
        if not word:
            return 0.0
        return max(0.0, 1.0 - distance / len(word))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct(self, word: str) -> Correction:
        """Return the best :class:`Correction` for *word*.

        Exact match → confidence=1.0, distance=0.
        Otherwise the closest dictionary word by Levenshtein distance.
        """
        if not self._dictionary:
            return Correction(original=word, corrected=word, distance=0, confidence=1.0)

        word_lower = word.lower()

        # Fast path: exact match
        if word_lower in self._dictionary:
            return Correction(original=word, corrected=word_lower, distance=0, confidence=1.0)

        best_word = self._dictionary[0]
        best_dist = self._edit_distance(word_lower, best_word)

        for candidate in self._dictionary[1:]:
            d = self._edit_distance(word_lower, candidate)
            if d < best_dist:
                best_dist = d
                best_word = candidate

        return Correction(
            original=word,
            corrected=best_word,
            distance=best_dist,
            confidence=self._confidence(word_lower, best_dist),
        )

    def correct_query(self, query: str) -> str:
        """Correct each whitespace-separated word in *query* and rejoin."""
        words = query.split()
        return " ".join(self.correct(w).corrected for w in words)

    def add_word(self, word: str) -> None:
        """Add *word* to the dictionary (lowercased, no duplicates)."""
        w = word.lower()
        if w not in self._dictionary:
            self._dictionary.append(w)

    def suggestions(self, word: str, n: int = 5) -> list[Correction]:
        """Return the top-*n* corrections sorted by distance asc, confidence desc."""
        word_lower = word.lower()
        [
            Correction(
                original=word,
                corrected=candidate,
                distance=self._edit_distance(word_lower, candidate),
                confidence=self._confidence(word_lower, self._edit_distance(word_lower, candidate)),
            )
            for candidate in self._dictionary
        ]
        # Precompute to avoid double computation; rebuild with stored distances
        scored: list[Correction] = []
        for candidate in self._dictionary:
            d = self._edit_distance(word_lower, candidate)
            scored.append(
                Correction(
                    original=word,
                    corrected=candidate,
                    distance=d,
                    confidence=self._confidence(word_lower, d),
                )
            )
        scored.sort(key=lambda c: (c.distance, -c.confidence))
        return scored[:n]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPELL_CORRECTOR_REGISTRY: dict[str, type] = {"default": SpellCorrector}
