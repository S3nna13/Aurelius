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

__all__ = [
    "BM25Retriever",
    "RETRIEVER_REGISTRY",
    "EMBEDDING_REGISTRY",
    "RERANKER_REGISTRY",
]
