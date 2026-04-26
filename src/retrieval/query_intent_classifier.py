"""Heuristic query-intent classifier for sparse retrieval pipelines.

Classifies natural-language queries into high-level intents using
keyword-based rules.  No external ML models or transformers — pure
Python stdlib so it runs safely on untrusted inputs in sandboxed
environments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.ASCII)


@dataclass(frozen=True)
class IntentRule:
    """A single heuristic rule that maps keywords to an intent."""

    intent: str
    keywords: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self):
        if not self.intent or not isinstance(self.intent, str):
            raise ValueError("intent must be a non-empty str")
        if not self.keywords:
            raise ValueError("keywords must not be empty")
        if any(not isinstance(k, str) or not k for k in self.keywords):
            raise ValueError("every keyword must be a non-empty str")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")


_DEFAULT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        intent="factual",
        keywords=("who", "what", "when", "where", "is", "did"),
        weight=0.9,
    ),
    IntentRule(
        intent="analytical",
        keywords=("why", "how", "compare", "versus", "vs", "analysis", "analyze"),
        weight=1.2,
    ),
    IntentRule(
        intent="analytical",
        keywords=("difference", "differences"),
        weight=2.0,
    ),
    IntentRule(
        intent="creative",
        keywords=("write", "create", "generate", "story", "poem", "draft", "compose", "imagine"),
    ),
    IntentRule(
        intent="procedural",
        keywords=(
            "steps",
            "how to",
            "guide",
            "tutorial",
            "instructions",
            "walkthrough",
            "procedure",
        ),
    ),
    IntentRule(
        intent="navigational",
        keywords=(
            "find",
            "locate",
            "access",
            "download",
            "open",
            "navigate",
            "url",
            "link",
            "site",
        ),
    ),
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class QueryIntentClassifier:
    """Keyword/heuristic classifier for query intents.

    Supports registering custom :class:`IntentRule` instances at runtime
    for domain-specific extensions.
    """

    def __init__(self) -> None:
        self._rules: list[IntentRule] = list(_DEFAULT_RULES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> str:
        """Return the intent with the highest heuristic score.

        Args:
            query: Raw search query (treated as untrusted).

        Returns:
            One of the registered intent strings.  If no keyword matches,
            returns ``"factual"`` as the conservative fallback.
        """
        self._validate_query(query)
        scores = self._score(query)
        if not scores:
            return "factual"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def confidence(self, query: str) -> float:
        """Return a normalised confidence in the range ``[0.0, 1.0]``.

        Confidence is computed as the ratio of the top intent score to the
        sum of *all* intent scores.  A single strong match therefore yields
        a value close to ``1.0``, whereas a tie yields a value close to
        ``1.0 / num_intents``.
        """
        self._validate_query(query)
        scores = self._score(query)
        if not scores:
            return 0.0
        total = sum(scores.values())
        top = max(scores.values())
        return top / total

    def batch_classify(self, queries: list[str]) -> list[str]:
        """Classify a list of queries, preserving order."""
        if not isinstance(queries, list):
            raise TypeError("queries must be a list of str")
        return [self.classify(q) for q in queries]

    def register_rule(self, rule: IntentRule) -> None:
        """Append a custom heuristic rule.

        Custom rules are evaluated *after* the built-in defaults, so they
        can be used to refine or override behaviour when scores tie.
        """
        if not isinstance(rule, IntentRule):
            raise TypeError("rule must be an IntentRule instance")
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("query must be str")
        # Reject extremely long inputs to avoid DoS via token explosion.
        if len(query) > 50_000:
            raise ValueError("query exceeds maximum length of 50000")

    def _score(self, query: str) -> dict[str, float]:
        """Map intent -> raw heuristic score."""
        toks = set(_tokens(query))
        q_lower = query.lower()
        scores: dict[str, float] = {}
        for rule in self._rules:
            hits = 0
            for kw in rule.keywords:
                if " " in kw and kw in q_lower:
                    # Multi-word phrase matches count for their word length
                    # so they outweigh overlapping single-word hits.
                    hits += max(1, kw.count(" ") + 1)
                elif kw in toks:
                    hits += 1
            if hits:
                scores[rule.intent] = scores.get(rule.intent, 0.0) + hits * rule.weight
        return scores


__all__ = ["IntentRule", "QueryIntentClassifier"]
