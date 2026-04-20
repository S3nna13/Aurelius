"""Character n-gram overlap for near-duplicate document clustering.

Implements a lightweight **Jaccard similarity** on character n-gram
multisets (Shingle-based dedup, Broder 1997) so data pipelines can flag
redundant shards before expensive tokenisation.  Pure Python / stdlib only.
"""

from __future__ import annotations

import re
from collections import Counter


def _ngrams(text: str, n: int) -> Counter[str]:
    s = re.sub(r"\s+", " ", (text or "").strip().lower())
    if len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def ngram_jaccard(a: str, b: str, *, n: int = 5) -> float:
    """Jaccard similarity of character n-gram multisets in ``[0, 1]``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("a and b must be str")
    ca, cb = _ngrams(a, n), _ngrams(b, n)
    if not ca and not cb:
        return 1.0 if a == b else 0.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    if union == 0:
        return 0.0
    return inter / union


def cluster_duplicates(
    documents: list[str],
    *,
    n: int = 5,
    threshold: float = 0.85,
) -> list[list[int]]:
    """Greedy clustering: indices with pairwise Jaccard >= threshold merge.

    Returns a partition of ``range(len(documents))`` as disjoint lists.
    """
    if not isinstance(documents, list):
        raise TypeError("documents must be a list of str")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    for i, d in enumerate(documents):
        if not isinstance(d, str):
            raise TypeError(f"documents[{i}] must be str")

    m = len(documents)
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(m):
        for j in range(i + 1, m):
            if ngram_jaccard(documents[i], documents[j], n=n) >= threshold:
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(m):
        r = find(i)
        buckets.setdefault(r, []).append(i)
    return sorted(buckets.values(), key=lambda g: (min(g), len(g)))


class NgramDeduplicationToolkit:
    """Registry-friendly façade over :func:`ngram_jaccard` / :func:`cluster_duplicates`."""

    ngram_jaccard = staticmethod(ngram_jaccard)

    @staticmethod
    def cluster_duplicates(
        documents: list[str],
        *,
        n: int = 5,
        threshold: float = 0.85,
    ) -> list[list[int]]:
        return cluster_duplicates(documents, n=n, threshold=threshold)


__all__ = ["NgramDeduplicationToolkit", "cluster_duplicates", "ngram_jaccard"]
