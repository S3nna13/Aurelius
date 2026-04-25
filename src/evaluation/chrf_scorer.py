from __future__ import annotations

from collections import Counter
from typing import Sequence


def _sanitize(text: str) -> str:
    return "".join(c for c in str(text) if c.isprintable() or c in "\n\r\t")


def _character_ngrams(text: str, n: int) -> Counter[str]:
    text = text.strip()
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf_score(references: Sequence[str], hypothesis: str | None, n: int = 6) -> float:
    if not references or not hypothesis:
        return 0.0
    hyp = _sanitize(str(hypothesis))
    if not hyp.strip():
        return 0.0

    best = 0.0
    for ref in references:
        ref_str = _sanitize(str(ref))
        if not ref_str.strip():
            continue

        total_precision = 0.0
        total_recall = 0.0
        for order in range(1, n + 1):
            ref_ngrams = _character_ngrams(ref_str, order)
            hyp_ngrams = _character_ngrams(hyp, order)
            if not ref_ngrams or not hyp_ngrams:
                continue
            matches = sum(min(hyp_ngrams[g], ref_ngrams.get(g, 0)) for g in hyp_ngrams)
            total_precision += matches / max(sum(hyp_ngrams.values()), 1)
            total_recall += matches / max(sum(ref_ngrams.values()), 1)

        avg_precision = total_precision / n
        avg_recall = total_recall / n
        if avg_precision + avg_recall == 0:
            continue
        beta = 1.0
        f_score = (1 + beta**2) * (avg_precision * avg_recall) / (beta**2 * avg_precision + avg_recall)
        best = max(best, f_score * 100)
    return best
