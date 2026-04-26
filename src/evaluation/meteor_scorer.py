from __future__ import annotations

from collections import Counter


def _sanitize(text: str) -> str:
    return "".join(c for c in str(text) if c.isprintable() or c in "\n\r\t")


def _unigram_precision(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    ref_count = Counter(reference)
    hyp_count = Counter(hypothesis)
    matches = sum(min(hyp_count[w], ref_count.get(w, 0)) for w in hyp_count)
    return matches, len(hypothesis)


def _alignment(matches: list[int], ref_len: int, hyp_len: int) -> tuple[int, int]:
    mapped = min(len(matches), ref_len, hyp_len)
    if mapped == 0:
        return 0, 0
    chunks = 1
    for i in range(1, len(matches)):
        if matches[i] != matches[i - 1] + 1:
            chunks += 1
    return mapped, chunks


def meteor_score(references: list[str], hypothesis: str | None) -> float:
    if not references or not hypothesis:
        return 0.0
    if not all(isinstance(r, str) for r in references):
        return 0.0
    hyp = _sanitize(str(hypothesis))
    if not hyp.strip():
        return 0.0
    hyp_tokens = hyp.split()
    if not hyp_tokens:
        return 0.0

    best = 0.0
    for ref in references:
        ref_str = _sanitize(str(ref))
        if not ref_str.strip():
            continue
        ref_tokens = ref_str.split()
        if not ref_tokens:
            continue
        matches, hyp_len = _unigram_precision(ref_tokens, hyp_tokens)
        mapped, chunks = _alignment(
            [i for i, w in enumerate(ref_tokens) if w in hyp_tokens],
            len(ref_tokens),
            hyp_len,
        )
        precision = matches / hyp_len if hyp_len > 0 else 0.0
        recall = matches / len(ref_tokens) if ref_tokens else 0.0
        if precision + recall == 0:
            continue
        frag = chunks / mapped if mapped > 0 else 1.0
        penalty = 0.5 * (frag**3)
        fmean = (precision * recall) / (0.5 * recall + 0.5 * precision)
        score = fmean * (1 - penalty)
        best = max(best, score)
    return best
