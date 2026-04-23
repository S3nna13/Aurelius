"""Aurelius search surface: web search stub, semantic search, result ranking."""

__all__ = [
    "SearchResult", "SearchQuery",
    "WebSearchStub", "WEB_SEARCH",
    "SemanticSearch", "SEMANTIC_SEARCH",
    "ResultRanker", "RESULT_RANKER",
    "SEARCH_REGISTRY",
]

from .web_search import SearchResult, SearchQuery, WebSearchStub, WEB_SEARCH
from .semantic_search import SemanticSearch, SEMANTIC_SEARCH
from .result_ranker import ResultRanker, RESULT_RANKER

SEARCH_REGISTRY: dict[str, object] = {
    "web": WEB_SEARCH,
    "semantic": SEMANTIC_SEARCH,
    "ranker": RESULT_RANKER,
}
