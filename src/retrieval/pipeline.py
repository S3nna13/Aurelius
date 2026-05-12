"""End-to-end retrieval pipeline.

Sequences five standalone retrieval components into a single, fault-tolerant
RAG pipeline:

  query_rewriter  -> hybrid_retriever -> reranker -> citation_tracker -> compressor

Each step is optional (``None`` = skip that step). If any step raises an
exception, the error is logged and the pipeline falls back to the previous
step's output. This is intentional: a partial RAG result is almost always
better than a hard failure on the user-facing path.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured RAG output usable by downstream generation."""

    query: str
    rewritten_query: str
    chunks: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    compressed_context: str = ""


class RetrievalPipeline:
    """Compose retrieval components into a single ``run(query)`` call.

    Parameters
    ----------
    query_rewriter:
        Optional. Object exposing ``.rewrite(str) -> str | list[str]``.
    retriever:
        Optional. Object exposing ``.query(str, k=int) -> list[(doc_id, score)]``.
        Documents must already be indexed via ``.add_documents`` before
        ``run()`` is called. If a ``corpus`` attribute (list[str]) is set on
        the pipeline (or supplied separately), chunks will be looked up.
    reranker:
        Optional. Object exposing
        ``.rerank(query, documents, doc_ids=None) -> list[(idx, score)]``
        or ``.rerank_with_ids(query, [(id, doc), ...]) -> list[(id, score)]``.
    citation_tracker:
        Optional. Object exposing ``.track(output_text, sources)``.
    compressor:
        Optional. Object exposing
        ``.compress(query, documents, scores=None) -> list[str]``.
    corpus:
        Optional. Sequence of source documents indexed in the retriever, in
        the same order; required to map ``doc_id -> chunk text`` after the
        retriever has run.
    """

    def __init__(
        self,
        query_rewriter: Any | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        citation_tracker: Any | None = None,
        compressor: Any | None = None,
        corpus: Sequence[str] | None = None,
    ) -> None:
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.citation_tracker = citation_tracker
        self.compressor = compressor
        self.corpus: list[str] = list(corpus) if corpus is not None else []

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #

    def set_corpus(self, corpus: Sequence[str]) -> None:
        """Replace the document lookup table used to resolve doc_ids."""
        self.corpus = list(corpus)

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    def _safe(self, label: str, fn, fallback):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RetrievalPipeline step %r failed: %s", label, exc)
            return fallback

    def _rewrite(self, query: str) -> str:
        if self.query_rewriter is None:
            return query

        def _do() -> str:
            out = self.query_rewriter.rewrite(query)
            if isinstance(out, list):
                # Decompose / multi-query: join with spaces so a single-shot
                # retriever still benefits from the expansion.
                return " ".join(str(x) for x in out if x)
            return str(out) if out else query

        return self._safe("query_rewriter", _do, query)

    def _retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.retriever is None:
            return []

        def _do() -> list[tuple[int, float]]:
            hits = self.retriever.query(query, k=top_k)
            return [(int(doc_id), float(score)) for doc_id, score in hits]

        return self._safe("retriever", _do, [])

    def _resolve_chunks(self, hits: list[tuple[int, float]]) -> tuple[list[str], list[float], list[int]]:
        chunks: list[str] = []
        scores: list[float] = []
        ids: list[int] = []
        for doc_id, score in hits:
            if 0 <= doc_id < len(self.corpus):
                chunks.append(self.corpus[doc_id])
                scores.append(score)
                ids.append(doc_id)
        return chunks, scores, ids

    def _rerank(
        self,
        query: str,
        chunks: list[str],
        scores: list[float],
        ids: list[int],
    ) -> tuple[list[str], list[float], list[int]]:
        if self.reranker is None or not chunks:
            return chunks, scores, ids

        def _do() -> tuple[list[str], list[float], list[int]]:
            ranked = self.reranker.rerank(query, chunks)
            new_chunks: list[str] = []
            new_scores: list[float] = []
            new_ids: list[int] = []
            for idx, score in ranked:
                if 0 <= idx < len(chunks):
                    new_chunks.append(chunks[idx])
                    new_scores.append(float(score))
                    new_ids.append(ids[idx] if idx < len(ids) else idx)
            return new_chunks, new_scores, new_ids

        return self._safe("reranker", _do, (chunks, scores, ids))

    def _track_citations(
        self,
        chunks: list[str],
        ids: list[int],
    ) -> list[dict]:
        if self.citation_tracker is None or not chunks:
            return []

        def _do() -> list[dict]:
            # Use a generic shape so the pipeline does not require importing
            # CitationTracker's Source dataclass at module import time.
            try:
                from .citation_tracker import Source  # local import, optional
            except Exception:  # pragma: no cover
                return []
            sources = [
                Source(id=str(ids[i] if i < len(ids) else i), text=c, origin="retrieval", retrieved_at="")
                for i, c in enumerate(chunks)
            ]
            # We don't have the model output at retrieval time, so we record
            # the chunk metadata only; downstream code can call the tracker
            # again post-generation. This step records a "potential citation"
            # list for transparency.
            return [
                {
                    "source_id": s.id,
                    "text": s.text,
                    "origin": s.origin,
                }
                for s in sources
            ]

        return self._safe("citation_tracker", _do, [])

    def _compress(self, query: str, chunks: list[str], scores: list[float]) -> str:
        if self.compressor is None or not chunks:
            return "\n\n".join(chunks)

        def _do() -> str:
            compressed = self.compressor.compress(query, chunks, scores=scores)
            return "\n\n".join(compressed)

        return self._safe("compressor", _do, "\n\n".join(chunks))

    def run(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Run the full pipeline. Returns a :class:`RetrievalResult`.

        Always returns a result; never raises on component failures.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}")

        rewritten = self._rewrite(query)
        hits = self._retrieve(rewritten, top_k)
        chunks, scores, ids = self._resolve_chunks(hits)
        chunks, scores, ids = self._rerank(rewritten, chunks, scores, ids)
        citations = self._track_citations(chunks, ids)
        compressed = self._compress(rewritten, chunks, scores)

        return RetrievalResult(
            query=query,
            rewritten_query=rewritten,
            chunks=chunks,
            scores=scores,
            citations=citations,
            compressed_context=compressed,
        )

    # ------------------------------------------------------------------ #
    # Defaults factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_defaults(cls, corpus: Sequence[str] | None = None) -> RetrievalPipeline:
        """Build a pipeline with sensible stdlib+torch defaults.

        - QueryRewriter: ``strategy="none"`` (no-op; expand needs a callable).
        - HybridRetriever: BM25 sparse-only via weighted fusion ``(1, 0)``.
        - CrossEncoderReranker: default token-overlap scorer.
        - CitationTracker: default thresholds.
        - ContextCompressor: ``strategy="extractive"``.
        """
        try:
            from .query_rewriter import QueryRewriter
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("QueryRewriter unavailable: %s", exc)
            QueryRewriter = None  # type: ignore[assignment]

        try:
            from .bm25_retriever import BM25Retriever
            from .hybrid_retriever import HybridRetriever
        except Exception as exc:  # pragma: no cover
            logger.warning("HybridRetriever unavailable: %s", exc)
            BM25Retriever = None  # type: ignore[assignment]
            HybridRetriever = None  # type: ignore[assignment]

        try:
            from .reranker import CrossEncoderReranker
        except Exception as exc:  # pragma: no cover
            logger.warning("CrossEncoderReranker unavailable: %s", exc)
            CrossEncoderReranker = None  # type: ignore[assignment]

        try:
            from .citation_tracker import CitationTracker
        except Exception as exc:  # pragma: no cover
            logger.warning("CitationTracker unavailable: %s", exc)
            CitationTracker = None  # type: ignore[assignment]

        try:
            from .compressor import ContextCompressor
        except Exception as exc:  # pragma: no cover
            logger.warning("ContextCompressor unavailable: %s", exc)
            ContextCompressor = None  # type: ignore[assignment]

        rewriter = QueryRewriter(strategy="none") if QueryRewriter is not None else None

        retriever = None
        if BM25Retriever is not None and HybridRetriever is not None:
            try:
                bm25 = BM25Retriever()
                retriever = HybridRetriever(
                    sparse_retriever=bm25,
                    fusion="weighted",
                    weights=(1.0, 0.0),
                )
                if corpus:
                    retriever.add_documents(list(corpus))
            except Exception as exc:  # pragma: no cover
                logger.warning("Default retriever construction failed: %s", exc)
                retriever = None

        reranker = CrossEncoderReranker(top_k=10) if CrossEncoderReranker is not None else None
        tracker = CitationTracker() if CitationTracker is not None else None
        compressor = (
            ContextCompressor(strategy="extractive", max_chars=2000, max_docs=5)
            if ContextCompressor is not None
            else None
        )

        return cls(
            query_rewriter=rewriter,
            retriever=retriever,
            reranker=reranker,
            citation_tracker=tracker,
            compressor=compressor,
            corpus=corpus,
        )


__all__ = ["RetrievalPipeline", "RetrievalResult"]
