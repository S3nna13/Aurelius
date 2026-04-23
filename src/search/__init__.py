"""Aurelius search surface: web search, semantic (TF-IDF), BM25, hybrid (RRF), ranker."""

__all__ = [
    "SearchResult", "SearchQuery",
    "WebSearchStub", "WEB_SEARCH",
    "SemanticSearch", "SEMANTIC_SEARCH",
    "BM25Document", "BM25Index", "BM25_INDEX",
    "HybridSearchIndex", "HYBRID_INDEX",
    "ResultRanker", "RESULT_RANKER",
    "SEARCH_REGISTRY",
]

from .web_search import SearchResult, SearchQuery, WebSearchStub, WEB_SEARCH
from .semantic_search import SemanticSearch, SEMANTIC_SEARCH
from .bm25_index import BM25Document, BM25Index, BM25_INDEX
from .hybrid_search_index import HybridSearchIndex, HYBRID_INDEX
from .result_ranker import ResultRanker, RESULT_RANKER

SEARCH_REGISTRY: dict[str, object] = {
    "web": WEB_SEARCH,
    "semantic": SEMANTIC_SEARCH,
    "bm25": BM25_INDEX,
    "hybrid": HYBRID_INDEX,
    "ranker": RESULT_RANKER,
}
