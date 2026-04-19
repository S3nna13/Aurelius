"""Pure rank-list fusion utilities.

This module exposes standalone fusion algorithms that combine ``N`` already-
ranked outputs from independent retrievers into a single fused ranked list.
Unlike :mod:`src.retrieval.hybrid_retriever`, which is a concrete retriever
that internally fuses a sparse and a dense channel, the functions here are
pure: a caller passes ``list[list[tuple[doc_id, score]]]`` and receives
``list[tuple[doc_id, float]]`` back.

Algorithms
----------
* :func:`reciprocal_rank_fusion` (Cormack, Clarke & Buettcher, SIGIR 2009):

      score(d) = sum_i  w_i / (k + rank_i(d))

  where ``rank_i(d)`` is the 1-based rank of ``d`` in the ``i``-th ranked
  list, ``k`` defaults to 60 per the reference paper, and ``w_i`` defaults
  to 1 for each list. Documents absent from a list contribute 0 from that
  list. Weights, when supplied, must be non-negative and must have the same
  length as ``rankings``; they are not re-normalized, matching the
  "weighted RRF" variant used by e.g. Chen et al. (2022) "Out-of-Domain
  Semantic Parsing via Ranking".

* :func:`borda_count` -- Classical Borda (de Borda, 1781): a document at
  rank ``r`` (1-based) in a list of length ``L`` contributes ``L - r``
  points. Documents absent from a list contribute 0. This is the same
  rank-based positional rule used in information-retrieval fusion surveys
  (Aslam & Montague, SIGIR 2001, "Models for Metasearch").

* :func:`comb_sum` / :func:`comb_mnz` (Fox & Shaw, TREC-2, 1994,
  "Combination of Multiple Searches"): each input list's scores are first
  normalized (``"minmax"`` maps each list to ``[0, 1]`` independently;
  ``"none"`` uses raw scores), then CombSUM sums the normalized scores and
  CombMNZ multiplies that sum by the number of input lists in which the
  document appears.

Determinism
-----------
All functions are deterministic: ties in the fused score are broken by
ascending ``doc_id`` (lexicographic string order), matching the tie-break
policy of :mod:`src.retrieval.hybrid_retriever`.

Error policy
------------
No silent fallbacks. A ranked list containing the same ``doc_id`` twice
raises :class:`ValueError`. Mismatched ``weights`` length raises. Unknown
normalization modes raise. An unknown fusion method passed to :func:`fuse`
raises.

Stdlib only. No numpy, no torch.
"""

from __future__ import annotations

from typing import Callable, Iterable

__all__ = [
    "reciprocal_rank_fusion",
    "borda_count",
    "comb_sum",
    "comb_mnz",
    "fuse",
    "FUSION_REGISTRY",
]

Ranking = list[tuple[str, float]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_rankings(rankings: list[Ranking]) -> None:
    """Validate overall shape and per-list uniqueness of doc ids.

    Raises :class:`ValueError` if the argument is not a list of lists of
    ``(str, float)`` pairs, or if a single ranking contains duplicate
    ``doc_id`` values. Duplicates within a list are ambiguous (which rank
    wins?) so we refuse to guess.
    """
    if not isinstance(rankings, list):
        raise TypeError(
            f"rankings must be a list of ranked lists, got {type(rankings).__name__}"
        )
    for i, ranking in enumerate(rankings):
        if not isinstance(ranking, list):
            raise TypeError(
                f"rankings[{i}] must be a list of (doc_id, score) tuples, "
                f"got {type(ranking).__name__}"
            )
        seen: set[str] = set()
        for j, entry in enumerate(ranking):
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
            ):
                raise TypeError(
                    f"rankings[{i}][{j}] must be a (str, float) tuple, got {entry!r}"
                )
            doc_id = entry[0]
            if doc_id in seen:
                raise ValueError(
                    f"duplicate doc_id {doc_id!r} in rankings[{i}]; "
                    "fusion inputs must be de-duplicated"
                )
            seen.add(doc_id)


def _sort_fused(scores: dict[str, float], top_n: int | None) -> Ranking:
    """Sort ``scores`` by descending fused score, tie-break by ``doc_id`` asc.

    Returns the full sorted list, truncated to ``top_n`` if given. ``top_n``
    must be ``None`` or a positive int.
    """
    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be a positive int or None, got {top_n!r}")
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        items = items[:top_n]
    return [(doc_id, float(score)) for doc_id, score in items]


def _minmax_normalize(ranking: Ranking) -> dict[str, float]:
    """Min-max normalize scores of a single ranking into ``[0, 1]``.

    If all scores are equal the normalized value is 1.0 for every doc
    (they all tie at the maximum). An empty ranking yields an empty dict.
    """
    if not ranking:
        return {}
    scores = [s for _, s in ranking]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        return {doc_id: 1.0 for doc_id, _ in ranking}
    return {doc_id: (s - lo) / span for doc_id, s in ranking}


def _raw_scores(ranking: Ranking) -> dict[str, float]:
    """Return the ranking as a ``{doc_id: score}`` dict."""
    return {doc_id: float(s) for doc_id, s in ranking}


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[Ranking],
    k: int = 60,
    top_n: int | None = None,
    weights: list[float] | None = None,
) -> Ranking:
    """Fuse ``N`` ranked lists with Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings:
        A list of ranked lists. Each inner list is an ordered sequence of
        ``(doc_id, score)`` tuples; the score is not used by RRF but is
        accepted to match the common "ranker output" contract. Position in
        the inner list determines the 1-based rank.
    k:
        Smoothing constant. Must be a positive number. Defaults to 60 per
        Cormack et al. (2009).
    top_n:
        If provided, truncate the returned fused list to the top ``top_n``.
    weights:
        Optional per-list non-negative weights. If given, must have the
        same length as ``rankings``. The fused score becomes
        ``sum_i w_i / (k + rank_i)``.

    Returns
    -------
    A fused ranked list sorted by descending fused score, ties broken by
    ascending ``doc_id``.
    """
    if not isinstance(k, (int, float)) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive number, got {k!r}")
    _validate_rankings(rankings)

    if weights is not None:
        if len(weights) != len(rankings):
            raise ValueError(
                f"weights length {len(weights)} does not match rankings length "
                f"{len(rankings)}"
            )
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or isinstance(w, bool) or w < 0:
                raise ValueError(
                    f"weights[{i}] must be a non-negative number, got {w!r}"
                )

    fused: dict[str, float] = {}
    for list_idx, ranking in enumerate(rankings):
        w = 1.0 if weights is None else float(weights[list_idx])
        if w == 0.0:
            continue
        for rank0, (doc_id, _score) in enumerate(ranking):
            rank = rank0 + 1
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)

    return _sort_fused(fused, top_n)


# ---------------------------------------------------------------------------
# Borda count
# ---------------------------------------------------------------------------


def borda_count(
    rankings: list[Ranking],
    top_n: int | None = None,
) -> Ranking:
    """Fuse ``N`` ranked lists with Borda count.

    For each ranked list of length ``L``, the document at 1-based rank
    ``r`` contributes ``L - r`` points. Documents absent from a list
    contribute 0. Lengths may differ across lists; each list's points
    are taken relative to its own length, matching the standard IR
    Borda formulation (Aslam & Montague, SIGIR 2001).
    """
    _validate_rankings(rankings)

    fused: dict[str, float] = {}
    for ranking in rankings:
        L = len(ranking)
        for rank0, (doc_id, _score) in enumerate(ranking):
            points = L - (rank0 + 1)
            fused[doc_id] = fused.get(doc_id, 0.0) + float(points)

    return _sort_fused(fused, top_n)


# ---------------------------------------------------------------------------
# CombSUM / CombMNZ
# ---------------------------------------------------------------------------


def _normalize_per_list(
    rankings: list[Ranking], normalize: str
) -> list[dict[str, float]]:
    """Apply per-list normalization and return list of ``{doc_id: score}``."""
    if normalize == "minmax":
        return [_minmax_normalize(r) for r in rankings]
    if normalize == "none":
        return [_raw_scores(r) for r in rankings]
    raise ValueError(
        f"unknown normalize={normalize!r}; expected 'minmax' or 'none'"
    )


def comb_sum(
    rankings: list[Ranking],
    normalize: str = "minmax",
    top_n: int | None = None,
) -> Ranking:
    """Fuse ``N`` ranked lists with CombSUM (Fox & Shaw, 1994).

    Scores within each list are normalized (``"minmax"`` or ``"none"``)
    and then summed across lists. Missing entries contribute 0.
    """
    _validate_rankings(rankings)
    per_list = _normalize_per_list(rankings, normalize)

    fused: dict[str, float] = {}
    for scores in per_list:
        for doc_id, s in scores.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + s

    return _sort_fused(fused, top_n)


def comb_mnz(
    rankings: list[Ranking],
    normalize: str = "minmax",
    top_n: int | None = None,
) -> Ranking:
    """Fuse ``N`` ranked lists with CombMNZ (Fox & Shaw, 1994).

    CombMNZ is CombSUM multiplied by the number of input lists in which
    the document appears. Heuristically, this rewards documents that are
    retrieved independently by multiple retrievers.
    """
    _validate_rankings(rankings)
    per_list = _normalize_per_list(rankings, normalize)

    sums: dict[str, float] = {}
    hits: dict[str, int] = {}
    for scores in per_list:
        for doc_id, s in scores.items():
            sums[doc_id] = sums.get(doc_id, 0.0) + s
            hits[doc_id] = hits.get(doc_id, 0) + 1

    fused = {doc_id: sums[doc_id] * hits[doc_id] for doc_id in sums}
    return _sort_fused(fused, top_n)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


FUSION_REGISTRY: dict[str, Callable[..., Ranking]] = {
    "rrf": reciprocal_rank_fusion,
    "borda": borda_count,
    "combsum": comb_sum,
    "combmnz": comb_mnz,
}


def fuse(method: str, rankings: list[Ranking], **kwargs) -> Ranking:
    """Dispatch to one of the registered fusion methods by name.

    Valid ``method`` values are the keys of :data:`FUSION_REGISTRY`:
    ``"rrf"``, ``"borda"``, ``"combsum"``, ``"combmnz"``. Any other value
    raises :class:`ValueError`. ``**kwargs`` are forwarded verbatim to
    the chosen implementation, which means e.g. ``top_n`` / ``k`` /
    ``weights`` / ``normalize`` work the same as when called directly.
    """
    if not isinstance(method, str):
        raise TypeError(f"method must be a str, got {type(method).__name__}")
    try:
        impl = FUSION_REGISTRY[method]
    except KeyError:
        valid = ", ".join(sorted(FUSION_REGISTRY))
        raise ValueError(
            f"unknown fusion method {method!r}; expected one of: {valid}"
        ) from None
    return impl(rankings, **kwargs)
