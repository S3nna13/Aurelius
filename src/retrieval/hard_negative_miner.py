"""Hard-negative mining for contrastive retrieval training.

Given a ``(query, positive_doc, corpus)`` triple, selects the top-``k``
*hard* negatives: documents that are similar to the positive (or to the
query) but are labeled irrelevant. Hard negatives are the main lever for
improving dense retrievers over naive random-negative training; random
negatives are almost always trivially separable and give near-zero
gradient once the encoder has warmed up.

Three strategies are supported:

- ``"bm25_hard"`` -- score the corpus with a :class:`BM25Retriever`
  against the *query* and return the top-``k`` non-positive hits. This
  is the "hard sparse negative" regime used in, e.g., DPR and the
  Contriever warm-start. Requires a ``retriever`` argument.

- ``"embedding_hard"`` -- score the corpus via cosine similarity
  between dense embeddings of the *positive* document and every other
  document. This is the ANCE-style regime: negatives are picked to be
  *embedding-space* close to the positive, which is exactly the
  direction the encoder most needs to separate. Requires an
  ``embedder`` exposing ``.encode(list[str]) -> tensor`` where rows
  are unit-normalized or simply proportional embeddings (we normalize
  defensively).

- ``"in_batch"`` (:meth:`HardNegativeMiner.mine_in_batch`) -- use the
  other examples' positives in the same mini-batch as negatives. This
  is the InfoNCE / SimCSE recipe and requires no extra compute at
  mining time.

References
----------
- Xiong et al. (2021). "Approximate Nearest Neighbor Negative Contrastive
  Learning for Dense Text Retrieval." arXiv:2007.00808 (ANCE).
- Izacard et al. (2021). "Unsupervised Dense Information Retrieval with
  Contrastive Learning." arXiv:2112.09118 (Contriever).

Design notes
------------
The miner deliberately identifies documents by *content string* rather
than by integer position. The surface contract takes ``corpus: list[str]``
and ``positive_doc_id: str``, so the most natural identity is the document
string itself. Duplicate strings in the corpus are treated as the same
logical document -- a hard-negative miner that returned a copy of the
positive because it happened to appear twice in the corpus would be a
silent correctness bug at training time. We de-duplicate defensively.

No foreign imports. Pure Python + PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from .bm25_retriever import BM25Retriever

__all__ = ["HardNegative", "HardNegativeMiner", "STRATEGIES"]


STRATEGIES: tuple[str, ...] = ("bm25_hard", "embedding_hard", "in_batch")


@dataclass(frozen=True)
class HardNegative:
    """A single mined hard negative.

    Attributes
    ----------
    doc_id:
        The document string itself. We use content as identity so callers
        can plug the result directly into a contrastive loss without a
        separate id -> text lookup.
    score:
        Strategy-dependent similarity score. Higher means harder (more
        similar to query / positive). Used for diagnostics and for
        sorting stability; not consumed by the loss itself.
    reason:
        Short tag identifying which strategy produced this negative.
        One of ``"bm25_hard"``, ``"embedding_hard"``, ``"in_batch"``.
    """

    doc_id: str
    score: float
    reason: str


class _Embedder(Protocol):
    """Structural type for the pluggable dense embedder.

    We only require an ``encode`` method taking a list of strings and
    returning a 2-D tensor (``[N, D]``). This matches both the real
    :class:`DenseEmbedder` once it grows a text-in interface and the
    stub embedders used in tests.
    """

    def encode(self, texts: list[str]) -> torch.Tensor:  # pragma: no cover - protocol
        ...


class HardNegativeMiner:
    """Hard-negative miner over a fixed corpus.

    Parameters
    ----------
    retriever:
        A :class:`BM25Retriever` (or compatible object exposing
        ``.query(str, k=int) -> list[(int, float)]`` and a
        ``num_documents`` property). Required for ``"bm25_hard"``.
    embedder:
        An object exposing ``.encode(list[str]) -> torch.Tensor``.
        Required for ``"embedding_hard"``.
    strategy:
        One of :data:`STRATEGIES`. Unknown strategies raise
        ``ValueError``.
    k:
        Number of hard negatives to return per query. Must be a positive
        integer. If the eligible corpus (corpus minus positive) is
        smaller than ``k``, all eligible docs are returned.
    """

    def __init__(
        self,
        retriever: BM25Retriever | None = None,
        embedder: _Embedder | None = None,
        strategy: str = "bm25_hard",
        k: int = 4,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")

        self.retriever = retriever
        self.embedder = embedder
        self.strategy = strategy
        self.k = k

    # ------------------------------------------------------------------ #
    # Validation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _validate_corpus(self, corpus: Sequence[str]) -> list[str]:
        if not isinstance(corpus, (list, tuple)):
            corpus = list(corpus)
        if len(corpus) == 0:
            raise ValueError("corpus must be non-empty")
        for i, d in enumerate(corpus):
            if not isinstance(d, str):
                raise TypeError(f"corpus[{i}] must be str, got {type(d).__name__}")
        return list(corpus)

    def _require_positive_in_corpus(self, positive_doc_id: str, corpus: list[str]) -> None:
        if not isinstance(positive_doc_id, str):
            raise TypeError(f"positive_doc_id must be str, got {type(positive_doc_id).__name__}")
        if positive_doc_id not in corpus:
            raise ValueError(
                "positive_doc_id is not present in corpus; refusing to mine "
                "negatives against an unknown positive"
            )

    # ------------------------------------------------------------------ #
    # Strategies                                                          #
    # ------------------------------------------------------------------ #

    def _mine_bm25(self, query: str, positive_doc_id: str, corpus: list[str]) -> list[HardNegative]:
        if self.retriever is None:
            raise ValueError("strategy 'bm25_hard' requires a retriever; got None")
        # Allow callers to pass either a pre-indexed retriever whose
        # document ordering matches `corpus`, or an un-indexed retriever
        # we index now. We index a *fresh* retriever only if it has no
        # corpus yet, to avoid silently clobbering a real index.
        retr = self.retriever
        if not getattr(retr, "_indexed", False):
            retr.add_documents(corpus)
        # Over-fetch so that after filtering out the positive we still
        # have at least k candidates when possible.
        want = min(self.k + corpus.count(positive_doc_id) + 1, len(corpus))
        hits = retr.query(query, k=want)
        negatives: list[HardNegative] = []
        seen: set[str] = set()
        for doc_id, score in hits:
            text = corpus[doc_id]
            if text == positive_doc_id:
                continue
            if text in seen:
                continue
            seen.add(text)
            negatives.append(HardNegative(doc_id=text, score=float(score), reason="bm25_hard"))
            if len(negatives) >= self.k:
                break
        return negatives

    def _mine_embedding(
        self, query: str, positive_doc_id: str, corpus: list[str]
    ) -> list[HardNegative]:
        if self.embedder is None:
            raise ValueError("strategy 'embedding_hard' requires an embedder; got None")
        # Encode the positive together with the corpus in a single pass
        # so callers with a learned encoder amortize the forward cost.
        texts = [positive_doc_id] + corpus
        with torch.no_grad():
            emb = self.embedder.encode(texts)
        if not isinstance(emb, torch.Tensor):
            raise TypeError(f"embedder.encode must return torch.Tensor, got {type(emb).__name__}")
        if emb.dim() != 2 or emb.shape[0] != len(texts):
            raise ValueError(
                f"embedder.encode returned shape {tuple(emb.shape)}, expected [{len(texts)}, D]"
            )
        emb = emb.to(dtype=torch.float32)
        # Defensive L2 normalization so cosine == dot product regardless
        # of whether the embedder already returns unit vectors.
        norms = emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        emb = emb / norms
        pos_vec = emb[0]
        corpus_vecs = emb[1:]
        sims = (corpus_vecs @ pos_vec).tolist()

        # Rank docs by descending similarity, ties broken by ascending
        # index for determinism. Skip the positive itself.
        ranked = sorted(
            range(len(corpus)),
            key=lambda i: (-sims[i], i),
        )
        negatives: list[HardNegative] = []
        seen: set[str] = set()
        for i in ranked:
            text = corpus[i]
            if text == positive_doc_id:
                continue
            if text in seen:
                continue
            seen.add(text)
            negatives.append(
                HardNegative(doc_id=text, score=float(sims[i]), reason="embedding_hard")
            )
            if len(negatives) >= self.k:
                break
        return negatives

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def mine(self, query: str, positive_doc_id: str, corpus: Sequence[str]) -> list[HardNegative]:
        """Mine hard negatives for a single ``(query, positive)`` pair."""
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        corpus_list = self._validate_corpus(corpus)
        self._require_positive_in_corpus(positive_doc_id, corpus_list)

        if self.strategy == "bm25_hard":
            return self._mine_bm25(query, positive_doc_id, corpus_list)
        if self.strategy == "embedding_hard":
            return self._mine_embedding(query, positive_doc_id, corpus_list)
        if self.strategy == "in_batch":
            raise ValueError(
                "strategy 'in_batch' is only valid via mine_in_batch(); "
                "in-batch mining is defined over a batch of pairs, not a "
                "single (query, positive) example"
            )
        # Unreachable given __init__ validation, but guard anyway.
        raise ValueError(f"unknown strategy {self.strategy!r}")

    def mine_batch(
        self,
        queries: Sequence[str],
        positive_doc_ids: Sequence[str],
        corpus: Sequence[str],
    ) -> list[list[HardNegative]]:
        """Mine hard negatives for a batch of ``(query, positive)`` pairs.

        Returns a list parallel to ``queries``; each element is the
        per-example hard-negative list.
        """
        if not isinstance(queries, (list, tuple)):
            queries = list(queries)
        if not isinstance(positive_doc_ids, (list, tuple)):
            positive_doc_ids = list(positive_doc_ids)
        if len(queries) != len(positive_doc_ids):
            raise ValueError(
                f"queries and positive_doc_ids length mismatch: "
                f"{len(queries)} vs {len(positive_doc_ids)}"
            )
        # Validate corpus once; each mine() call revalidates cheaply but
        # avoids re-checking per-element types here.
        self._validate_corpus(corpus)
        return [self.mine(q, pos, corpus) for q, pos in zip(queries, positive_doc_ids)]

    def mine_in_batch(
        self, query_positive_pairs: Sequence[tuple[str, str]]
    ) -> list[list[HardNegative]]:
        """In-batch negatives: each example's negatives are every *other*
        example's positive.

        For a batch of ``N`` pairs this returns ``N`` lists of
        ``N - 1`` :class:`HardNegative` entries each. The ``k`` parameter
        is *not* applied here: in-batch size is the batch itself, which
        is the InfoNCE convention (Gao et al. 2021, Karpukhin et al.
        2020). If you want a smaller slice, take a prefix downstream.
        """
        if not isinstance(query_positive_pairs, (list, tuple)):
            query_positive_pairs = list(query_positive_pairs)
        pairs: list[tuple[str, str]] = []
        for i, pair in enumerate(query_positive_pairs):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not isinstance(pair[1], str)
            ):
                raise TypeError(f"query_positive_pairs[{i}] must be a (str, str) tuple")
            pairs.append((pair[0], pair[1]))
        if len(pairs) < 2:
            raise ValueError(
                "in-batch mining requires at least 2 pairs to produce "
                f"any negatives; got {len(pairs)}"
            )

        out: list[list[HardNegative]] = []
        for i, (_q, pos_i) in enumerate(pairs):
            negs: list[HardNegative] = []
            for j, (_qj, pos_j) in enumerate(pairs):
                if j == i:
                    continue
                # The positive of example j serves as a negative for
                # example i. Score 0.0 is a placeholder: in-batch
                # negatives are unordered, and downstream InfoNCE does
                # not consume this score.
                negs.append(HardNegative(doc_id=pos_j, score=0.0, reason="in_batch"))
            out.append(negs)
        return out

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"HardNegativeMiner(strategy={self.strategy!r}, k={self.k}, "
            f"retriever={'set' if self.retriever is not None else 'None'}, "
            f"embedder={'set' if self.embedder is not None else 'None'})"
        )
