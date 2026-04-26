"""Multi-hop retrieval for Aurelius.

Iteratively expands a query using retrieved document snippets to follow
reasoning chains across multiple retrieval hops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

RETRIEVAL_REGISTRY: dict = {}


class _RetrieverProtocol(Protocol):
    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]: ...


@dataclass
class HopResult:
    hop: int
    query: str
    retrieved_ids: list[str]
    scores: list[float]


@dataclass
class MultiHopConfig:
    max_hops: int = 3
    top_k_per_hop: int = 5
    expansion_threshold: float = 0.5


class MultiHopRetriever:
    """Wraps a dense retriever and BM25 index for iterative multi-hop retrieval.

    Parameters
    ----------
    dense_retriever:
        Any object with a ``retrieve(query, k) -> list[tuple[str, float]]`` method.
    bm25_index:
        Any object with a ``retrieve(query, k) -> list[tuple[str, float]]`` method.
        If *None*, only the dense retriever is used.
    """

    def __init__(
        self,
        dense_retriever: Any,
        bm25_index: Any = None,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_index

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        initial_query: str,
        config: MultiHopConfig | None = None,
    ) -> list[HopResult]:
        """Run multi-hop retrieval and return one HopResult per hop."""
        if config is None:
            config = MultiHopConfig()

        hop_results: list[HopResult] = []
        query = initial_query

        for hop in range(1, config.max_hops + 1):
            pairs = self._fetch(query, config.top_k_per_hop)
            if not pairs:
                break
            ids = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            hop_results.append(HopResult(hop=hop, query=query, retrieved_ids=ids, scores=scores))

            top_score = scores[0] if scores else 0.0
            if top_score < config.expansion_threshold:
                break
            if hop == config.max_hops:
                break

            # Expand query with top-scored doc snippet (first 100 chars of id used
            # as a proxy when no doc store is available; callers can sub-class to
            # override _snippet).
            snippet = self._snippet(ids[0])
            query = f"{query} {snippet}"

        return hop_results

    def flatten(self, hop_results: list[HopResult]) -> list[str]:
        """Return unique doc ids across all hops, ordered by first-hop score."""
        if not hop_results:
            return []

        # Build order from first hop, then append later hops if not seen
        seen: set[str] = set()
        ordered: list[str] = []

        # First pass: respect first-hop ordering
        for hr in hop_results:
            for doc_id in hr.retrieved_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    ordered.append(doc_id)

        return ordered

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _fetch(self, query: str, k: int) -> list[tuple[str, float]]:
        """Retrieve from dense (and optionally BM25), merge by max score."""
        results: dict[str, float] = {}
        for pair in self._dense.retrieve(query, k):
            doc_id, score = pair[0], pair[1]
            results[doc_id] = max(results.get(doc_id, float("-inf")), score)
        if self._bm25 is not None:
            for pair in self._bm25.retrieve(query, k):
                doc_id, score = pair[0], pair[1]
                results[doc_id] = max(results.get(doc_id, float("-inf")), score)
        return sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]

    def _snippet(self, doc_id: str) -> str:
        """Return first 100 chars of *doc_id* as an expansion snippet.

        Override this in a subclass that has access to a document store.
        """
        return doc_id[:100]


RETRIEVAL_REGISTRY["multi_hop"] = MultiHopRetriever
