"""Web search stub: query normalization, result dataclass, mock results."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 1.0
    source: str = "web"
    id: str = field(default_factory=_new_id)


@dataclass
class SearchQuery:
    query: str
    max_results: int = 10
    language: str = "en"
    safe_search: bool = True


_DEFAULT_STUB_RESULTS: list[SearchResult] = [
    SearchResult(
        title="Python Programming Language - Official Site",
        url="https://www.python.org",
        snippet="Python is a high-level, general-purpose programming language emphasizing code readability.",  # noqa: E501
        score=0.95,
    ),
    SearchResult(
        title="Machine Learning: An Introduction",
        url="https://example.com/machine-learning-intro",
        snippet="Machine learning is a branch of artificial intelligence enabling systems to learn from data.",  # noqa: E501
        score=0.90,
    ),
    SearchResult(
        title="Transformers: Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        snippet="Transformers introduced the attention mechanism that revolutionized NLP and deep learning.",  # noqa: E501
        score=0.88,
    ),
    SearchResult(
        title="Neural Networks Explained",
        url="https://example.com/neural-networks",
        snippet="Neural networks are computational models loosely inspired by biological neural networks in brains.",  # noqa: E501
        score=0.85,
    ),
    SearchResult(
        title="Large Language Models Overview",
        url="https://example.com/language-models",
        snippet="Language models predict the probability of token sequences, powering modern AI assistants.",  # noqa: E501
        score=0.82,
    ),
]


class WebSearchStub:
    """Stub web search that filters and ranks seed results."""

    def __init__(self, seed_results: list[SearchResult] | None = None) -> None:
        if seed_results is None:
            self._seeds = list(_DEFAULT_STUB_RESULTS)
        else:
            self._seeds = list(seed_results)

    def normalize_query(self, query: str) -> str:
        """Lowercase, strip, collapse multiple spaces."""
        return re.sub(r"\s+", " ", query.strip().lower())

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Filter seed results by query tokens, return up to max_results sorted by score desc."""
        normalized = self.normalize_query(query.query)
        tokens = normalized.split()

        matched: list[SearchResult] = []
        for result in self._seeds:
            haystack = (result.title + " " + result.snippet).lower()
            if any(tok in haystack for tok in tokens):
                matched.append(result)

        if not matched:
            matched = list(self._seeds)

        matched.sort(key=lambda r: r.score, reverse=True)
        return matched[: query.max_results]

    def top_k(self, results: list[SearchResult], k: int) -> list[SearchResult]:
        """Return first k results."""
        return results[:k]


WEB_SEARCH = WebSearchStub()
