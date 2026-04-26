"""Aurelius search surface: web search, semantic (TF-IDF), BM25, hybrid (RRF), ranker."""

__all__ = [
    "SearchResult",
    "SearchQuery",
    "WebSearchStub",
    "WEB_SEARCH",
    "SemanticSearch",
    "SEMANTIC_SEARCH",
    "BM25Document",
    "BM25Index",
    "BM25_INDEX",
    "HybridSearchIndex",
    "HYBRID_INDEX",
    "ResultRanker",
    "RESULT_RANKER",
    "SEARCH_REGISTRY",
    # Cycle-146 code search + query expansion
    "CodeSymbol",
    "CodeFile",
    "CodeSearchIndex",
    "CODE_SEARCH_INDEX",
    "ExpandedQuery",
    "QueryExpander",
    "QUERY_EXPANDER",
    # Cycle-147 inverted index, search cache, query parser
    "Posting",
    "InvertedIndex",
    "INVERTED_INDEX_REGISTRY",
    "CacheEntry",
    "SearchCache",
    "SEARCH_CACHE_REGISTRY",
    "TokenType",
    "QueryToken",
    "QueryParser",
    "QUERY_PARSER_REGISTRY",
    # Cross-encoder reranker + dense embedder
    "CrossEncoderConfig",
    "CrossEncoderModel",
    "CrossEncoderReranker",
    "RankedResult",
    "CROSS_ENCODER_RERANKER",
    "EmbedderConfig",
    "DenseEmbedder",
    "DENSE_EMBEDDER",
]

from .bm25_index import BM25_INDEX, BM25Document, BM25Index
from .hybrid_search_index import HYBRID_INDEX, HybridSearchIndex
from .result_ranker import RESULT_RANKER, ResultRanker
from .semantic_search import SEMANTIC_SEARCH, SemanticSearch
from .web_search import WEB_SEARCH, SearchQuery, SearchResult, WebSearchStub

SEARCH_REGISTRY: dict[str, object] = {
    "web": WEB_SEARCH,
    "semantic": SEMANTIC_SEARCH,
    "bm25": BM25_INDEX,
    "hybrid": HYBRID_INDEX,
    "ranker": RESULT_RANKER,
}

# --- Cycle-146 code search + query expansion ----------------------------------
from .code_search import CODE_SEARCH_INDEX, CodeFile, CodeSearchIndex, CodeSymbol  # noqa: F401
from .query_expander import QUERY_EXPANDER, ExpandedQuery, QueryExpander  # noqa: F401

SEARCH_REGISTRY.update({"code": CODE_SEARCH_INDEX, "expander": QUERY_EXPANDER})

# --- Cycle-147 inverted index, search cache, query parser -----------------
from .inverted_index import INVERTED_INDEX_REGISTRY, InvertedIndex, Posting  # noqa: F401
from .query_parser import QUERY_PARSER_REGISTRY, QueryParser, QueryToken, TokenType  # noqa: F401
from .search_cache import SEARCH_CACHE_REGISTRY, CacheEntry, SearchCache  # noqa: F401

SEARCH_REGISTRY.update(
    {
        "inverted_index": InvertedIndex,
        "search_cache": SearchCache,
        "query_parser": QueryParser,
    }
)

# --- Cross-encoder reranker + dense embedder ----------------------------------
from .cross_encoder_reranker import (  # noqa: F401
    CROSS_ENCODER_RERANKER,
    CrossEncoderConfig,
    CrossEncoderModel,
    CrossEncoderReranker,
    RankedResult,
)
from .dense_embedder import DENSE_EMBEDDER, DenseEmbedder, EmbedderConfig  # noqa: F401

SEARCH_REGISTRY.update({"reranker": CROSS_ENCODER_RERANKER, "dense_embedder": DENSE_EMBEDDER})
