"""Pure-Python Okapi BM25 sparse retriever.

Implementation of BM25 following Robertson & Zaragoza (2009),
"The Probabilistic Relevance Framework: BM25 and Beyond".

Score of document ``d`` for query ``q`` with terms ``t_1..t_n``:

    score(d, q) = sum_{t in q} idf(t) * ( tf(t, d) * (k1 + 1) )
                               / ( tf(t, d) + k1 * (1 - b + b * |d| / avgdl) )

where

    idf(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

This is the Robertson-Sparck Jones smoothed form with the "+1" additive
guard that keeps the IDF non-negative even for terms appearing in more
than half the collection, matching the behavior described in the
reference and used by Lucene/ElasticSearch-class implementations.

The implementation deliberately uses only the Python standard library
(``math`` + built-in containers). No numpy, no rank_bm25, no sklearn,
no whoosh, no lucene. Query-time scoring is performed incrementally by
walking the postings lists of query terms; no per-doc dense vector is
ever materialized, which keeps memory usage proportional to the query
footprint rather than the corpus size.

Thread-safety: ``add_documents`` is not re-entrant. Query-time methods
(``query``, ``query_batch``, ``score``) only read precomputed state and
are safe to call concurrently from multiple threads once indexing is
complete.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

__all__ = ["BM25Retriever", "default_tokenizer"]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def default_tokenizer(text: str) -> list[str]:
    """Lowercase + Unicode-aware ``\\w+`` regex tokenizer.

    Matches any Unicode alphanumeric run (including letters with
    diacritics, CJK, digits, and ``_``). The lowercase step uses
    Python's full Unicode case-folding semantics via ``str.lower``.

    This is intentionally minimal: BM25Retriever accepts any callable
    ``str -> list[str]``, so production deployments are expected to
    plug in the real subword / code-aware tokenizer.
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenizer expects str, got {type(text).__name__}")
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 sparse retriever.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter. Must be > 0. Typical
        range is [1.2, 2.0]; default 1.5 matches common practice and
        the Lucene default family.
    b:
        Length-normalization parameter in [0, 1]. 0 disables length
        normalization; 1 fully normalizes by document length. Default
        0.75 is the canonical value from Robertson et al.
    tokenizer:
        Callable mapping a string to a list of string tokens. If
        ``None`` (default), :func:`default_tokenizer` is used.

    Notes
    -----
    Call :meth:`add_documents` exactly once with the full corpus before
    issuing queries. Re-calling ``add_documents`` will raise to avoid
    silently producing an inconsistent index; construct a new retriever
    instead. (Incremental indexing is intentionally out of scope for
    this surface -- BM25 IDF is corpus-wide and naive incremental
    updates would invalidate previously-returned scores.)
    """

    __slots__ = (
        "k1",
        "b",
        "_tokenizer",
        "_doc_count",
        "doc_lengths",
        "avg_doc_length",
        "token_df",
        "inverted_index",
        "_idf",
        "_indexed",
    )

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if not isinstance(k1, (int, float)) or math.isnan(k1) or k1 <= 0:
            raise ValueError(f"k1 must be a positive finite float, got {k1!r}")
        if not isinstance(b, (int, float)) or math.isnan(b) or b < 0.0 or b > 1.0:
            raise ValueError(f"b must be a float in [0, 1], got {b!r}")
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable or None")

        self.k1: float = float(k1)
        self.b: float = float(b)
        self._tokenizer: Callable[[str], list[str]] = (
            tokenizer if tokenizer is not None else default_tokenizer
        )

        # Index state. Populated by add_documents.
        self._doc_count: int = 0
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        # token -> number of documents containing token
        self.token_df: dict[str, int] = {}
        # token -> list of (doc_id, tf). Postings are kept sorted by doc_id.
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        # token -> precomputed IDF
        self._idf: dict[str, float] = {}
        self._indexed: bool = False

    # ------------------------------------------------------------------ #
    # Indexing                                                            #
    # ------------------------------------------------------------------ #

    def add_documents(self, docs: Sequence[str]) -> None:
        """Tokenize ``docs`` and build the inverted index + IDF table.

        ``docs[i]`` becomes document id ``i``. Duplicate document
        strings are allowed and will simply receive identical scores
        (they occupy distinct doc_ids).

        Raises
        ------
        RuntimeError
            If called more than once on the same retriever.
        TypeError
            If ``docs`` is not a sequence of strings.
        ValueError
            If ``docs`` is empty.
        """
        if self._indexed:
            raise RuntimeError(
                "BM25Retriever.add_documents may only be called once; "
                "construct a new retriever for a new corpus."
            )
        if not isinstance(docs, (list, tuple)):
            # Materialize once so we can validate and iterate deterministically.
            docs = list(docs)
        if len(docs) == 0:
            raise ValueError("add_documents requires a non-empty corpus")

        n = len(docs)
        doc_lengths: list[int] = [0] * n
        # Temporary: token -> list of (doc_id, tf). We build term frequency
        # per document first, then append into the final postings list.
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: dict[str, int] = defaultdict(int)

        for doc_id, doc in enumerate(docs):
            if not isinstance(doc, str):
                raise TypeError(f"docs[{doc_id}] must be str, got {type(doc).__name__}")
            tokens = self._tokenizer(doc)
            if not isinstance(tokens, list):
                # Accept any iterable, but normalize to list for consistency.
                tokens = list(tokens)
            doc_lengths[doc_id] = len(tokens)
            if not tokens:
                continue
            # Term frequencies within this doc.
            tf_local: dict[str, int] = {}
            for tok in tokens:
                tf_local[tok] = tf_local.get(tok, 0) + 1
            for tok, tf in tf_local.items():
                postings[tok].append((doc_id, tf))
                df[tok] += 1

        total_len = sum(doc_lengths)
        # avgdl: average document length. Guard the all-empty-docs case
        # by falling back to 1.0 so that the length-normalization factor
        # (1 - b + b * |d|/avgdl) remains well-defined; all |d| are 0 in
        # that case anyway so the numerical value does not matter.
        avgdl = (total_len / n) if total_len > 0 else 1.0

        # Precompute IDF using the Robertson-Sparck Jones smoothed form
        # with the additive "+1" guard:   log((N - df + 0.5)/(df + 0.5) + 1)
        idf: dict[str, float] = {}
        for tok, dfi in df.items():
            idf[tok] = math.log((n - dfi + 0.5) / (dfi + 0.5) + 1.0)

        # Freeze state.
        self._doc_count = n
        self.doc_lengths = doc_lengths
        self.avg_doc_length = avgdl
        self.token_df = dict(df)
        # Postings are already sorted by doc_id because we enumerate in order.
        self.inverted_index = {tok: list(pl) for tok, pl in postings.items()}
        self._idf = idf
        self._indexed = True

    # ------------------------------------------------------------------ #
    # Query                                                               #
    # ------------------------------------------------------------------ #

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError("BM25Retriever has no corpus; call add_documents() first.")

    def _score_tokens(self, q_tokens: Iterable[str]) -> dict[int, float]:
        """Accumulate BM25 scores for the given query tokens.

        Returns a ``{doc_id: score}`` mapping containing only documents
        that share at least one in-vocabulary token with the query.
        """
        k1 = self.k1
        b = self.b
        avgdl = self.avg_doc_length
        doc_lens = self.doc_lengths
        index = self.inverted_index
        idf_tab = self._idf

        scores: dict[int, float] = {}

        # Query terms are scored with their query-side multiplicity, which
        # is the standard BM25 treatment (each occurrence of a repeated
        # query term contributes once). We *don't* collapse duplicate query
        # terms, matching Robertson & Zaragoza section 3.
        for tok in q_tokens:
            postings = index.get(tok)
            if not postings:
                continue
            idf = idf_tab[tok]
            if idf == 0.0:
                continue
            # Inlined inner loop for speed; this is the hot path.
            for doc_id, tf in postings:
                dl = doc_lens[doc_id]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                # denom > 0 always: k1>0, b in [0,1], dl>=0, avgdl>0.
                contrib = idf * tf * (k1 + 1.0) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + contrib
        return scores

    def query(self, q: str, k: int = 10) -> list[tuple[int, float]]:
        """Return the top-``k`` documents for query ``q``.

        Results are ``(doc_id, score)`` tuples sorted by descending
        score, with ties broken by ascending doc_id for determinism.
        An empty query or a query containing only out-of-vocabulary
        tokens returns ``[]``.

        Raises
        ------
        RuntimeError
            If :meth:`add_documents` has not been called.
        ValueError
            If ``k`` is not a positive integer.
        TypeError
            If ``q`` is not a string.
        """
        self._require_indexed()
        if not isinstance(q, str):
            raise TypeError(f"query expects str, got {type(q).__name__}")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")

        if not q:
            return []
        q_tokens = self._tokenizer(q)
        if not q_tokens:
            return []

        scores = self._score_tokens(q_tokens)
        if not scores:
            return []

        # Cap k at the number of scored documents; we do not pad with
        # zero-score docs because that would be a silent fallback that
        # hides the "no relevant docs" signal.
        k_eff = min(k, len(scores))
        # Deterministic ordering: -score ASC, doc_id ASC.
        items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(doc_id, score) for doc_id, score in items[:k_eff]]

    def query_batch(self, qs: Sequence[str], k: int = 10) -> list[list[tuple[int, float]]]:
        """Vectorized query interface for throughput.

        Each element of ``qs`` is scored independently; the result is
        parallel to ``qs``. This wraps :meth:`query` and therefore
        preserves the exact same scoring / ordering / edge-case
        semantics, which is relied on by tests.
        """
        self._require_indexed()
        if not isinstance(qs, (list, tuple)):
            qs = list(qs)
        return [self.query(q, k=k) for q in qs]

    # ------------------------------------------------------------------ #
    # Introspection helpers (useful for tests and diagnostics)            #
    # ------------------------------------------------------------------ #

    def score(self, q: str, doc_id: int) -> float:
        """Raw BM25 score of ``doc_id`` under query ``q``.

        Returns 0.0 for empty queries, OOV queries, or documents with
        no query-term overlap. Useful for tests and for reranking
        pipelines that need to rescore a candidate set.
        """
        self._require_indexed()
        if not (0 <= doc_id < self._doc_count):
            raise IndexError(f"doc_id {doc_id} out of range [0, {self._doc_count})")
        if not q:
            return 0.0
        q_tokens = self._tokenizer(q)
        if not q_tokens:
            return 0.0
        return self._score_tokens(q_tokens).get(doc_id, 0.0)

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return self._doc_count

    def idf(self, token: str) -> float:
        """IDF of ``token`` under the precomputed smoothed form.

        Returns 0.0 for tokens not present in the corpus; this matches
        the ``+1`` additive guard in the smoothed IDF formula, which
        would yield ``log((N+0.5)/0.5 + 1)`` for df=0 -- but since such
        a token contributes nothing at query time (no postings to walk)
        we return 0.0 here as the honest "no information" answer.
        """
        self._require_indexed()
        return self._idf.get(token, 0.0)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"BM25Retriever(k1={self.k1}, b={self.b}, "
            f"n_docs={self._doc_count}, vocab={len(self._idf)})"
        )
