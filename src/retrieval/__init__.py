"""Retrieval surface for Aurelius.

Exposes registries for retrievers, embedding models, and rerankers so that
higher-level components (RAG pipelines, tool-use scaffolding, long-context
routing) can resolve implementations by string name without importing the
concrete modules eagerly.

Registered retrievers:
    - "bm25": Classical Okapi BM25 sparse retriever (pure Python stdlib).

Embedding and reranker registries are intentionally left empty here; they are
populated by sibling modules as those surfaces land.
"""

from __future__ import annotations

from .bm25_retriever import BM25Retriever

RETRIEVER_REGISTRY: dict[str, type] = {}
EMBEDDING_REGISTRY: dict[str, type] = {}
RERANKER_REGISTRY: dict[str, type] = {}

RETRIEVER_REGISTRY["bm25"] = BM25Retriever

from .hybrid_retriever import HybridRetriever  # noqa: E402
RETRIEVER_REGISTRY["hybrid_rrf"] = HybridRetriever

from .reciprocal_rank_fusion import (  # noqa: E402
    FUSION_REGISTRY,
    borda_count,
    comb_mnz,
    comb_sum,
    fuse,
    reciprocal_rank_fusion,
)

from .cross_encoder_reranker import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderReranker,
)

RERANKER_REGISTRY["cross_encoder"] = CrossEncoderReranker

from .dense_embedding_trainer import (  # noqa: E402
    DenseEmbedder,
    EmbedderConfig,
    EmbeddingTrainer,
    InfoNCELoss,
)

EMBEDDING_REGISTRY["dense"] = DenseEmbedder

from .code_aware_tokenizer import (  # noqa: E402
    KEYWORDS,
    SUPPORTED_LANGUAGES,
    CodeAwareTokenizer,
)

from .instruction_prefix_embedder import (  # noqa: E402
    INSTRUCTION_PREFIXES,
    InstructionPrefixEmbedder,
)

from .diversity_reranker import (  # noqa: E402
    JaccardDiversityReranker,
    MMRReranker,
    cosine_similarity,
    jaccard_similarity,
)

RERANKER_REGISTRY["mmr"] = MMRReranker
RERANKER_REGISTRY["jaccard_mmr"] = JaccardDiversityReranker

from .corpus_indexer import Chunk, CorpusIndexer  # noqa: E402

__all__ = [
    "BM25Retriever",
    "Chunk",
    "CorpusIndexer",
    "HybridRetriever",
    "RETRIEVER_REGISTRY",
    "EMBEDDING_REGISTRY",
    "RERANKER_REGISTRY",
    "reciprocal_rank_fusion",
    "borda_count",
    "comb_sum",
    "comb_mnz",
    "fuse",
    "FUSION_REGISTRY",
    "DenseEmbedder",
    "EmbedderConfig",
    "EmbeddingTrainer",
    "InfoNCELoss",
    "CodeAwareTokenizer",
    "KEYWORDS",
    "SUPPORTED_LANGUAGES",
    "INSTRUCTION_PREFIXES",
    "InstructionPrefixEmbedder",
    "MMRReranker",
    "JaccardDiversityReranker",
    "cosine_similarity",
    "jaccard_similarity",
]
