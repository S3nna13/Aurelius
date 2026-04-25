"""Aurelius search surface: web search, semantic (TF-IDF), BM25, hybrid (RRF), ranker."""

__all__ = [
    "SearchResult", "SearchQuery",
    "WebSearchStub", "WEB_SEARCH",
    "SemanticSearch", "SEMANTIC_SEARCH",
    "BM25Document", "BM25Index", "BM25_INDEX",
    "HybridSearchIndex", "HYBRID_INDEX",
    "ResultRanker", "RESULT_RANKER",
    "SEARCH_REGISTRY",
    # Cycle-146 code search + query expansion
    "CodeSymbol", "CodeFile", "CodeSearchIndex", "CODE_SEARCH_INDEX",
    "ExpandedQuery", "QueryExpander", "QUERY_EXPANDER",
    # Cycle-147 inverted index, search cache, query parser
    "Posting", "InvertedIndex", "INVERTED_INDEX_REGISTRY",
    "CacheEntry", "SearchCache", "SEARCH_CACHE_REGISTRY",
    "TokenType", "QueryToken", "QueryParser", "QUERY_PARSER_REGISTRY",
    # Cross-encoder reranker + dense embedder
    "CrossEncoderConfig", "CrossEncoderModel", "CrossEncoderReranker", "RankedResult", "CROSS_ENCODER_RERANKER",
    "EmbedderConfig", "DenseEmbedder", "DENSE_EMBEDDER",
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

# --- Cycle-146 code search + query expansion ----------------------------------
from .code_search import CodeSymbol, CodeFile, CodeSearchIndex, CODE_SEARCH_INDEX  # noqa: F401
from .query_expander import ExpandedQuery, QueryExpander, QUERY_EXPANDER  # noqa: F401
SEARCH_REGISTRY.update({"code": CODE_SEARCH_INDEX, "expander": QUERY_EXPANDER})

# --- Cycle-147 inverted index, search cache, query parser -----------------
from .inverted_index import Posting, InvertedIndex, INVERTED_INDEX_REGISTRY  # noqa: F401
from .search_cache import CacheEntry, SearchCache, SEARCH_CACHE_REGISTRY  # noqa: F401
from .query_parser import TokenType, QueryToken, QueryParser, QUERY_PARSER_REGISTRY  # noqa: F401
SEARCH_REGISTRY.update({
    "inverted_index": InvertedIndex,
    "search_cache": SearchCache,
    "query_parser": QueryParser,
})

# --- Cross-encoder reranker + dense embedder ----------------------------------
from .cross_encoder_reranker import CrossEncoderConfig, CrossEncoderModel, CrossEncoderReranker, RankedResult, CROSS_ENCODER_RERANKER  # noqa: F401
from .dense_embedder import EmbedderConfig, DenseEmbedder, DENSE_EMBEDDER  # noqa: F401
SEARCH_REGISTRY.update({"reranker": CROSS_ENCODER_RERANKER, "dense_embedder": DENSE_EMBEDDER})
