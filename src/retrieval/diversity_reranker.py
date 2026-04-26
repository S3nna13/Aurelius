"""Maximal Marginal Relevance (MMR) diversity reranker.

Reference
---------
Carbonell, J. & Goldstein, J. (1998). "The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries." SIGIR'98.

Given a candidate set ``R`` with relevance scores and a set of already
selected documents ``S``, MMR selects the next document ``d*`` as::

    d* = argmax_{d in R \\ S}  lambda * rel(d) - (1 - lambda) * max_{d' in S} sim(d, d')

When ``S`` is empty, the penalty term is zero and the highest-relevance
candidate is chosen. ``lambda_ = 1.0`` reduces to pure relevance ranking;
``lambda_ = 0.0`` is pure inter-document diversity (ignoring relevance
after the first pick).

Two similarity modes are supported:

* ``"cosine"`` -- cosine similarity on dense embeddings (torch.Tensor).
* ``"jaccard"`` -- Jaccard overlap on token sets (``JaccardDiversityReranker``).

The module is pure PyTorch for the cosine kernel; no transformers, no
sklearn, no numpy dependency. It is intentionally small and side-effect
free so it can be composed with BM25 / hybrid / cross-encoder front-ends.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "MMRReranker",
    "JaccardDiversityReranker",
    "cosine_similarity",
    "jaccard_similarity",
]


# --------------------------------------------------------------------------- #
# Similarity primitives                                                       #
# --------------------------------------------------------------------------- #


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two 1-D torch tensors.

    Returns a Python ``float`` in ``[-1, 1]``. Zero-norm inputs yield
    ``0.0`` (rather than ``nan``) so downstream MMR arithmetic stays
    finite.
    """
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("cosine_similarity expects torch.Tensor inputs")
    if a.shape != b.shape:
        raise ValueError(f"cosine_similarity: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    a_f = a.to(dtype=torch.float32).flatten()
    b_f = b.to(dtype=torch.float32).flatten()
    na = torch.linalg.vector_norm(a_f)
    nb = torch.linalg.vector_norm(b_f)
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float(torch.dot(a_f, b_f) / (na * nb))


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard index ``|A ∩ B| / |A ∪ B|`` on arbitrary hashable sets.

    By convention ``J(∅, ∅) = 1.0`` (two empty token bags are identical
    under the set model).
    """
    if not isinstance(a, (set, frozenset)) or not isinstance(b, (set, frozenset)):
        raise TypeError("jaccard_similarity expects set/frozenset inputs")
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _validate_lambda(lambda_: float) -> float:
    if not isinstance(lambda_, (int, float)) or math.isnan(float(lambda_)):
        raise ValueError(f"lambda_ must be a real number in [0, 1], got {lambda_!r}")
    lam = float(lambda_)
    if lam < 0.0 or lam > 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lam}")
    return lam


# --------------------------------------------------------------------------- #
# MMR (dense similarity)                                                      #
# --------------------------------------------------------------------------- #


class MMRReranker:
    """Maximal Marginal Relevance reranker over dense embeddings.

    Parameters
    ----------
    lambda_:
        Trade-off weight. ``1.0`` is pure relevance (original order
        preserved); ``0.0`` is pure diversity.
    similarity:
        Currently supports ``"cosine"``. Any other value raises at
        construction time -- we do not silently fall back.
    """

    __slots__ = ("lambda_", "similarity", "_sim_fn")

    def __init__(self, lambda_: float = 0.5, similarity: str = "cosine") -> None:
        self.lambda_: float = _validate_lambda(lambda_)
        if similarity != "cosine":
            raise ValueError(f"MMRReranker: unknown similarity {similarity!r} (expected 'cosine')")
        self.similarity: str = similarity
        self._sim_fn = cosine_similarity

    def rerank(
        self,
        ranked: list[tuple[str, float]],
        embeddings: dict[str, Tensor],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Greedy MMR selection over ``ranked`` candidates.

        Parameters
        ----------
        ranked:
            Input list of ``(doc_id, relevance_score)`` pairs. Order is
            assumed to reflect the upstream retriever's ranking; MMR
            uses ``relevance_score`` numerically and does not rely on
            position.
        embeddings:
            Map from ``doc_id`` to a 1-D ``torch.Tensor`` embedding.
            Missing keys raise ``KeyError``.
        k:
            Maximum number of documents to return. If ``k`` exceeds
            ``len(ranked)``, all candidates are returned.

        Returns
        -------
        list of ``(doc_id, mmr_score)`` in selection order. The score
        is the MMR objective evaluated at pick time (which equals the
        raw relevance for the first pick, when no penalty applies).
        """
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"k must be a non-negative int, got {k!r}")
        if not ranked:
            return []
        # Defensive: detect missing embeddings early so we fail fast
        # rather than after a partial selection.
        for doc_id, _ in ranked:
            if doc_id not in embeddings:
                raise KeyError(f"MMRReranker: no embedding for doc_id {doc_id!r}")

        lam = self.lambda_
        remaining: list[tuple[str, float]] = list(ranked)
        selected: list[tuple[str, float]] = []
        # Cache of pairwise sims between a selected doc and every candidate.
        # Keyed by candidate doc_id -> running max similarity to selected.
        max_sim_to_selected: dict[str, float] = {doc_id: 0.0 for doc_id, _ in ranked}

        target = min(k, len(remaining))
        while len(selected) < target and remaining:
            best_idx = -1
            best_score = -math.inf
            for idx, (doc_id, rel) in enumerate(remaining):
                if selected:
                    penalty = max_sim_to_selected[doc_id]
                    mmr = lam * rel - (1.0 - lam) * penalty
                else:
                    mmr = lam * rel
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            chosen_id, _ = remaining.pop(best_idx)
            selected.append((chosen_id, best_score))
            # Update running max-sim for all still-remaining candidates
            chosen_emb = embeddings[chosen_id]
            for doc_id, _ in remaining:
                s = self._sim_fn(embeddings[doc_id], chosen_emb)
                if s > max_sim_to_selected[doc_id]:
                    max_sim_to_selected[doc_id] = s

        return selected


# --------------------------------------------------------------------------- #
# Jaccard-based lexical diversity reranker                                    #
# --------------------------------------------------------------------------- #


def _default_jaccard_tokenizer(text: str) -> set:
    # Cheap whitespace + lowercase split. Production callers are
    # expected to inject a real tokenizer (e.g. CodeAwareTokenizer).
    return set(text.lower().split())


class JaccardDiversityReranker:
    """MMR reranker using Jaccard similarity on token sets.

    This variant is useful when dense embeddings are unavailable and a
    purely lexical diversity signal is acceptable -- e.g. surfacing
    alternative phrasings of the same answer from a BM25 candidate set.
    """

    __slots__ = ("lambda_", "_tokenizer")

    def __init__(
        self,
        lambda_: float = 0.5,
        tokenizer: Callable[[str], set] | None = None,
    ) -> None:
        self.lambda_: float = _validate_lambda(lambda_)
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable or None")
        self._tokenizer: Callable[[str], set] = (
            tokenizer if tokenizer is not None else _default_jaccard_tokenizer
        )

    def rerank(
        self,
        ranked: list[tuple[str, float]],
        docs: dict[str, str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Greedy MMR selection using Jaccard overlap as penalty."""
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"k must be a non-negative int, got {k!r}")
        if not ranked:
            return []
        for doc_id, _ in ranked:
            if doc_id not in docs:
                raise KeyError(f"JaccardDiversityReranker: no document text for {doc_id!r}")

        lam = self.lambda_
        token_sets: dict[str, set] = {
            doc_id: set(self._tokenizer(docs[doc_id])) for doc_id, _ in ranked
        }

        remaining = list(ranked)
        selected: list[tuple[str, float]] = []
        max_sim: dict[str, float] = {doc_id: 0.0 for doc_id, _ in ranked}

        target = min(k, len(remaining))
        while len(selected) < target and remaining:
            best_idx = -1
            best_score = -math.inf
            for idx, (doc_id, rel) in enumerate(remaining):
                if selected:
                    mmr = lam * rel - (1.0 - lam) * max_sim[doc_id]
                else:
                    mmr = lam * rel
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            chosen_id, _ = remaining.pop(best_idx)
            selected.append((chosen_id, best_score))
            chosen_tokens = token_sets[chosen_id]
            for doc_id, _ in remaining:
                s = jaccard_similarity(token_sets[doc_id], chosen_tokens)
                if s > max_sim[doc_id]:
                    max_sim[doc_id] = s

        return selected
