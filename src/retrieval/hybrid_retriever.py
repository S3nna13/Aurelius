"""Hybrid sparse+dense retriever with RRF / weighted score fusion.

Combines a sparse retriever (typically :class:`BM25Retriever`) with a dense
vector retriever to produce a fused ranking. Two fusion modes are supported:

* ``"rrf"`` -- Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR
  2009, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
  Learning Methods"). For each candidate document ``d``, the fused score is

      score(d) = sum_i  1 / (k_rrf + rank_i(d))

  where ``rank_i(d)`` is the 1-based rank of ``d`` in the ``i``-th ranked
  list (sparse, dense). Documents not appearing in a list contribute 0 from
  that list. ``k_rrf`` defaults to 60 per the reference paper and damps the
  contribution of documents that are merely top-lists noise.

* ``"weighted"`` -- Min-max normalize each ranked list's scores into
  ``[0, 1]`` independently, then take the convex combination

      score(d) = w_s * norm_sparse(d) + w_d * norm_dense(d)

  Missing entries from either list contribute 0 (equivalent to the minimum
  normalized score). The weights must be non-negative and are renormalized
  to sum to 1.

Dense backend:
    A dense retriever can be supplied either as
      * an object with a compatible ``.add_documents(docs)`` + ``.query(q, k)``
        surface (same contract as BM25Retriever), or
      * an ``embed_fn: str -> list[float]`` callable. In the latter case, a
        minimal in-process cosine-similarity scorer is built: document
        embeddings are stacked into a torch tensor and queries are scored by
        L2-normalized dot product. This keeps the surface self-contained for
        tests and small production indices while still allowing a proper
        ANN-backed retriever to be plugged in without code changes.

Design constraints for Aurelius:
  * No numpy, no faiss/annoy/hnswlib/sklearn. Dense math uses ``torch``.
  * No silent fallbacks. An invalid fusion name, a missing dense backend when
    the weight on the dense channel is non-zero, or a non-callable
    ``embed_fn`` all raise at construction or at the first offending call.
  * Determinism. Ties are broken by ascending ``doc_id`` so that rankings are
    reproducible across runs for caching and eval harnesses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

__all__ = ["HybridRetriever", "_CosineDenseRetriever"]


class _CosineDenseRetriever:
    """Minimal dense retriever built from an ``embed_fn``.

    Stores L2-normalized document embeddings in a single ``(N, D)`` torch
    tensor so that ``query`` is a single matmul. This is deliberately not
    exposed via the top-level package; it is an implementation detail of
    :class:`HybridRetriever` used when the caller passes an ``embed_fn``
    instead of a fully-formed dense retriever object.
    """

    __slots__ = ("_embed_fn", "_matrix", "_dim", "_n", "_indexed")

    def __init__(self, embed_fn: Callable[[str], Sequence[float]]) -> None:
        if not callable(embed_fn):
            raise TypeError("embed_fn must be callable")
        self._embed_fn = embed_fn
        self._matrix: torch.Tensor | None = None
        self._dim: int = 0
        self._n: int = 0
        self._indexed: bool = False

    @staticmethod
    def _to_vec(v: Sequence[float]) -> torch.Tensor:
        t = torch.as_tensor(list(v), dtype=torch.float32)
        if t.ndim != 1:
            raise ValueError(
                f"embed_fn must return a 1-D sequence of floats, got shape {tuple(t.shape)}"
            )
        if t.numel() == 0:
            raise ValueError("embed_fn returned an empty vector")
        return t

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    def add_documents(self, docs: Sequence[str]) -> None:
        if self._indexed:
            raise RuntimeError("_CosineDenseRetriever.add_documents may only be called once.")
        if not isinstance(docs, (list, tuple)):
            docs = list(docs)
        if len(docs) == 0:
            raise ValueError("add_documents requires a non-empty corpus")
        vecs: list[torch.Tensor] = []
        for i, d in enumerate(docs):
            if not isinstance(d, str):
                raise TypeError(f"docs[{i}] must be str, got {type(d).__name__}")
            v = self._to_vec(self._embed_fn(d))
            vecs.append(v)
        dim = vecs[0].shape[0]
        for i, v in enumerate(vecs):
            if v.shape[0] != dim:
                raise ValueError(
                    f"embed_fn produced inconsistent dim: docs[0]={dim}, docs[{i}]={v.shape[0]}"
                )
        mat = torch.stack(vecs, dim=0)
        self._matrix = self._l2_normalize(mat)
        self._dim = dim
        self._n = mat.shape[0]
        self._indexed = True

    def query(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        if not self._indexed or self._matrix is None:
            raise RuntimeError("_CosineDenseRetriever has no corpus; call add_documents() first.")
        if not isinstance(q, str):
            raise TypeError(f"query expects str, got {type(q).__name__}")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if not q:
            return []
        qv = self._to_vec(self._embed_fn(q))
        if qv.shape[0] != self._dim:
            raise ValueError(f"embed_fn query dim {qv.shape[0]} != corpus dim {self._dim}")
        qv = self._l2_normalize(qv.unsqueeze(0)).squeeze(0)
        # Cosine similarity in [-1, 1]; corpus rows are already L2-normalized.
        scores = (self._matrix @ qv).tolist()
        k_eff = min(k, self._n)
        # Deterministic: -score ASC, doc_id ASC.
        ranked = sorted(enumerate(scores), key=lambda kv: (-kv[1], kv[0]))
        return [(i, float(s)) for i, s in ranked[:k_eff]]

    @property
    def num_documents(self) -> int:
        return self._n


_ALLOWED_FUSIONS = ("rrf", "weighted")


class HybridRetriever:
    """Hybrid sparse+dense retriever.

    Parameters
    ----------
    sparse_retriever:
        Retriever with a ``.query(q, k) -> list[(doc_id, score)]`` method and
        an ``.add_documents(docs)`` method. Typically a
        :class:`BM25Retriever`.
    dense_retriever:
        Optional. Object implementing the same informal protocol. Mutually
        exclusive with ``embed_fn`` (exactly one may be provided when the
        fusion makes use of the dense channel).
    embed_fn:
        Optional. ``str -> 1D sequence of floats``. When supplied, a
        cosine-similarity dense retriever is built internally.
    fusion:
        ``"rrf"`` (default) or ``"weighted"``.
    k_rrf:
        The RRF constant ``k`` from Cormack et al. 2009. Must be > 0.
        Default 60.
    weights:
        ``(w_sparse, w_dense)`` non-negative weights for the weighted fusion
        mode. Renormalized to sum to 1. For pure sparse, pass ``(1, 0)``; in
        that case the dense backend is optional.
    candidate_multiplier:
        Each sub-retriever is queried for ``k * candidate_multiplier``
        candidates so that fusion sees enough overlap. Default 4 matches
        common RRF deployments.
    """

    __slots__ = (
        "sparse",
        "dense",
        "fusion",
        "k_rrf",
        "w_sparse",
        "w_dense",
        "candidate_multiplier",
        "_indexed",
        "_n_docs",
    )

    def __init__(
        self,
        sparse_retriever,
        dense_retriever=None,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        fusion: str = "rrf",
        k_rrf: int = 60,
        weights: tuple[float, float] = (0.5, 0.5),
        candidate_multiplier: int = 4,
    ) -> None:
        if sparse_retriever is None:
            raise ValueError("sparse_retriever is required")
        if not hasattr(sparse_retriever, "query") or not hasattr(sparse_retriever, "add_documents"):
            raise TypeError("sparse_retriever must expose .add_documents and .query")

        if fusion not in _ALLOWED_FUSIONS:
            raise ValueError(f"fusion must be one of {_ALLOWED_FUSIONS}, got {fusion!r}")

        if not isinstance(k_rrf, int) or isinstance(k_rrf, bool) or k_rrf <= 0:
            raise ValueError(f"k_rrf must be a positive int, got {k_rrf!r}")

        if not isinstance(weights, (tuple, list)) or len(weights) != 2:
            raise ValueError("weights must be a 2-tuple (w_sparse, w_dense)")
        w_s, w_d = float(weights[0]), float(weights[1])
        if w_s < 0.0 or w_d < 0.0:
            raise ValueError(f"weights must be non-negative, got {weights!r}")
        if (w_s + w_d) == 0.0:
            raise ValueError("weights must not both be zero")

        if dense_retriever is not None and embed_fn is not None:
            raise ValueError("pass exactly one of dense_retriever or embed_fn, not both")

        if embed_fn is not None and not callable(embed_fn):
            raise TypeError("embed_fn must be callable")

        # Resolve dense backend. A dense backend is REQUIRED whenever the
        # dense channel can affect the output:
        #   * fusion == "rrf" always mixes both lists,
        #   * fusion == "weighted" uses dense iff w_d > 0.
        dense_needed = fusion == "rrf" or w_d > 0.0
        dense = None
        if dense_retriever is not None:
            if not hasattr(dense_retriever, "query") or not hasattr(
                dense_retriever, "add_documents"
            ):
                raise TypeError("dense_retriever must expose .add_documents and .query")
            dense = dense_retriever
        elif embed_fn is not None:
            dense = _CosineDenseRetriever(embed_fn)
        elif dense_needed:
            raise ValueError(
                f"fusion={fusion!r} with weights={weights!r} requires a dense "
                "backend; supply dense_retriever= or embed_fn=, or set "
                "weights=(1, 0) with fusion='weighted' for sparse-only."
            )

        if not isinstance(candidate_multiplier, int) or candidate_multiplier < 1:
            raise ValueError(
                f"candidate_multiplier must be an int >= 1, got {candidate_multiplier!r}"
            )

        self.sparse = sparse_retriever
        self.dense = dense
        self.fusion: str = fusion
        self.k_rrf: int = int(k_rrf)
        # Store renormalized weights so score magnitudes are comparable across
        # different absolute weight choices.
        total = w_s + w_d
        self.w_sparse: float = w_s / total
        self.w_dense: float = w_d / total
        self.candidate_multiplier: int = int(candidate_multiplier)
        self._indexed: bool = False
        self._n_docs: int = 0

    # ------------------------------------------------------------------ #
    # Indexing                                                            #
    # ------------------------------------------------------------------ #

    def add_documents(self, docs: Sequence[str]) -> None:
        """Index ``docs`` into both sparse and dense backends.

        Raises ``ValueError`` on empty input and ``RuntimeError`` if called
        more than once (matches the BM25 contract and prevents split-brain
        sparse/dense indices).
        """
        if self._indexed:
            raise RuntimeError("HybridRetriever.add_documents may only be called once.")
        if not isinstance(docs, (list, tuple)):
            docs = list(docs)
        if len(docs) == 0:
            raise ValueError("add_documents requires a non-empty corpus")

        self.sparse.add_documents(docs)
        if self.dense is not None:
            self.dense.add_documents(docs)
        self._n_docs = len(docs)
        self._indexed = True

    # ------------------------------------------------------------------ #
    # Query                                                               #
    # ------------------------------------------------------------------ #

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError("HybridRetriever has no corpus; call add_documents() first.")

    def _candidate_k(self, k: int) -> int:
        # Pull more candidates from each sub-retriever than requested so that
        # fusion has meaningful overlap to work with.
        return max(k * self.candidate_multiplier, k)

    @staticmethod
    def _rrf(
        sparse_hits: list[tuple[int, float]],
        dense_hits: list[tuple[int, float]],
        k_rrf: int,
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion from two ranked lists.

        ``rank`` is 1-based. Per Cormack et al. 2009, the contribution from
        each list is ``1 / (k_rrf + rank)``; absent documents contribute 0.
        """
        scores: dict[int, float] = {}
        for rank, (doc_id, _s) in enumerate(sparse_hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        for rank, (doc_id, _s) in enumerate(dense_hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
        return scores

    @staticmethod
    def _minmax(hits: list[tuple[int, float]]) -> dict[int, float]:
        """Min-max normalize scores from a ranked list into [0, 1].

        Degenerate cases:
          * empty list -> empty dict
          * single element -> {doc: 1.0}
          * all equal -> all 1.0 (they're tied at the top)
        """
        if not hits:
            return {}
        scores = [s for _d, s in hits]
        lo = min(scores)
        hi = max(scores)
        if hi == lo:
            return {d: 1.0 for d, _s in hits}
        span = hi - lo
        return {d: (s - lo) / span for d, s in hits}

    def query(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        """Return the top-``k`` fused documents for query ``q``.

        Results are ``(doc_id, fused_score)`` sorted by descending fused
        score with ties broken by ascending ``doc_id``.
        """
        self._require_indexed()
        if not isinstance(q, str):
            raise TypeError(f"query expects str, got {type(q).__name__}")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if not q:
            return []

        cand_k = self._candidate_k(k)
        sparse_hits = self.sparse.query(q, k=cand_k)

        dense_hits: list[tuple[int, float]] = []
        if self.dense is not None:
            # Skip dense for pure sparse weighted mode -- it cannot affect the
            # result and avoids unnecessary embedding calls.
            if not (self.fusion == "weighted" and self.w_dense == 0.0):
                dense_hits = self.dense.query(q, k=cand_k)

        if not sparse_hits and not dense_hits:
            return []

        if self.fusion == "rrf":
            fused = self._rrf(sparse_hits, dense_hits, self.k_rrf)
        elif self.fusion == "weighted":
            ns = self._minmax(sparse_hits)
            nd = self._minmax(dense_hits)
            fused = {}
            for d, s in ns.items():
                fused[d] = fused.get(d, 0.0) + self.w_sparse * s
            for d, s in nd.items():
                fused[d] = fused.get(d, 0.0) + self.w_dense * s
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unknown fusion {self.fusion!r}")

        if not fused:
            return []

        k_eff = min(k, len(fused))
        items = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(doc_id, float(score)) for doc_id, score in items[:k_eff]]

    @property
    def num_documents(self) -> int:
        return self._n_docs

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"HybridRetriever(fusion={self.fusion!r}, k_rrf={self.k_rrf}, "
            f"weights=({self.w_sparse:.3f}, {self.w_dense:.3f}), "
            f"n_docs={self._n_docs})"
        )
