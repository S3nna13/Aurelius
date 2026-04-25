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

from .hard_negative_miner import (  # noqa: E402
    STRATEGIES as HARD_NEGATIVE_STRATEGIES,
    HardNegative,
    HardNegativeMiner,
)

from .colbert_late_interaction import (  # noqa: E402
    ColBERTConfig,
    ColBERTScorer,
)

RERANKER_REGISTRY["colbert"] = ColBERTScorer

from .prf_query_expander import PRFExpansionResult, PRFQueryExpander  # noqa: E402

QUERY_EXPANDER_REGISTRY: dict[str, type] = {
    "prf": PRFQueryExpander,
}

from .code_aware_embedder import (  # noqa: E402
    CodeAwareEmbedder,
    CodeFeatures,
    split_identifier,
    stub_token_embed,
)

EMBEDDING_REGISTRY["code_aware"] = CodeAwareEmbedder

from .citation_tracker import (  # noqa: E402
    CitationReport,
    CitationSpan,
    CitationTracker,
    Source as CitationSource,
)

CITATION_REGISTRY: dict[str, type] = {
    "basic": CitationTracker,
}

from .neural_reranker import (  # noqa: E402
    CrossEncoderReranker as NeuralCrossEncoderReranker,
    ListwiseReranker,
    RerankScore,
)

RERANKER_REGISTRY["neural_cross_encoder"] = NeuralCrossEncoderReranker
RERANKER_REGISTRY["listwise"] = ListwiseReranker

from .dense_retriever import DenseDocument, DenseRetriever  # noqa: E402

RETRIEVER_REGISTRY["dense"] = DenseRetriever

from .hybrid_fusion_v2 import (  # noqa: E402
    FusionResult,
    FusionStrategy,
    HYBRID_FUSION_REGISTRY,
    HybridFusionV2,
)

from .passage_segmenter import (  # noqa: E402
    PassageSegmenter,
    Segment,
    SegmentStrategy,
)

SEGMENTER_REGISTRY: dict = {
    "passage_segmenter": PassageSegmenter(),
}

from .multi_hop_retriever import (  # noqa: E402
    HopResult,
    MultiHopConfig,
    MultiHopRetriever,
)

RETRIEVER_REGISTRY["multi_hop"] = MultiHopRetriever

from .citation_extractor import (  # noqa: E402
    Citation,
    CitationExtractor,
    CitationType,
)

CITATION_REGISTRY["citation_extractor"] = CitationExtractor

__all__ = [
    "BM25Retriever",
    "Chunk",
    "CorpusIndexer",
    "HardNegative",
    "HardNegativeMiner",
    "HARD_NEGATIVE_STRATEGIES",
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
    "PRFExpansionResult",
    "PRFQueryExpander",
    "QUERY_EXPANDER_REGISTRY",
    "CitationReport",
    "CitationSpan",
    "CitationTracker",
    "CitationSource",
    "CITATION_REGISTRY",
    "CodeAwareEmbedder",
    "CodeFeatures",
    "split_identifier",
    "stub_token_embed",
    # neural_reranker
    "NeuralCrossEncoderReranker",
    "ListwiseReranker",
    "RerankScore",
    # dense_retriever
    "DenseDocument",
    "DenseRetriever",
    # hybrid_fusion_v2
    "FusionResult",
    "FusionStrategy",
    "HYBRID_FUSION_REGISTRY",
    "HybridFusionV2",
    # passage_segmenter
    "PassageSegmenter",
    "Segment",
    "SegmentStrategy",
    "SEGMENTER_REGISTRY",
    # multi_hop_retriever
    "HopResult",
    "MultiHopConfig",
    "MultiHopRetriever",
    # citation_extractor
    "Citation",
    "CitationExtractor",
    "CitationType",
]
