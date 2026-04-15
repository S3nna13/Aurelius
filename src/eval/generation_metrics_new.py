"""Generation evaluation metrics: BLEU, METEOR, CIDEr, and aggregation helpers.

Implements:
- tokenize_simple       : lowercase + split on whitespace/punctuation
- compute_ngrams        : Counter of n-gram tuples
- meteor_score          : METEOR with unigram F-mean and chunk penalty
- compute_idf           : IDF weights over a corpus
- cider_score           : TF-IDF weighted cosine similarity (CIDEr)
- sentence_bleu         : sentence-level BLEU-1..4 with brevity penalty
- aggregate_scores      : corpus-level mean/min/max for a given metric
- GenerationEvaluator   : convenience class wrapping all metrics
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_simple(text: str) -> List[str]:
    """Lowercase and split on whitespace/punctuation; filter empty strings."""
    tokens = re.split(r"[\s\W]+", text.lower())
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# N-gram counter
# ---------------------------------------------------------------------------

def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Return a Counter of n-gram tuples from *tokens*."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------------

def meteor_score(
    hypothesis: str,
    reference: str,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """Compute METEOR between *hypothesis* and *reference*.

    Steps
    -----
    1. Unigram precision P and recall R (with clipping by reference counts).
    2. F-mean: F = P*R / (alpha*P + (1-alpha)*R).
    3. Chunk penalty: p = gamma * (chunks / matches)^beta, where *chunks* is
       the number of contiguous matched-word runs in hypothesis order.
    4. Final score: F * (1 - p), clipped to [0, 1].

    Returns 0.0 whenever there are no matches.
    """
    hyp_tokens = tokenize_simple(hypothesis)
    ref_tokens = tokenize_simple(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # --- unigram matching (clipped by reference counts) ---
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    matches = sum(min(c, ref_counts[w]) for w, c in hyp_counts.items())

    if matches == 0:
        return 0.0

    precision = matches / len(hyp_tokens)
    recall = matches / len(ref_tokens)

    denom = alpha * precision + (1.0 - alpha) * recall
    if denom == 0.0:
        return 0.0
    fmean = precision * recall / denom

    # --- chunk penalty ---
    # Build a set of (word, ref_position) pairs that are matched, then scan
    # hypothesis left-to-right and count contiguous runs in ref order.
    ref_word_positions: dict[str, List[int]] = {}
    for pos, word in enumerate(ref_tokens):
        ref_word_positions.setdefault(word, []).append(pos)

    # Greedy alignment: for each hypothesis token (in order) find the
    # earliest available reference position.
    available: dict[str, List[int]] = {
        w: sorted(ps) for w, ps in ref_word_positions.items()
    }
    used: dict[str, int] = {w: 0 for w in available}  # pointer into available list

    aligned_ref_positions: List[Optional[int]] = []
    for word in hyp_tokens:
        bucket = available.get(word)
        idx = used.get(word, 0)
        if bucket and idx < len(bucket) and used.get(word, 0) < ref_counts.get(word, 0):
            aligned_ref_positions.append(bucket[idx])
            used[word] = idx + 1
        else:
            aligned_ref_positions.append(None)

    # Count chunks: consecutive matched positions that are also consecutive
    # in the reference.
    matched_positions = [p for p in aligned_ref_positions if p is not None]
    if not matched_positions:
        return 0.0

    chunks = 1
    for i in range(1, len(matched_positions)):
        if matched_positions[i] != matched_positions[i - 1] + 1:
            chunks += 1

    actual_matches = len(matched_positions)
    penalty = gamma * (chunks / actual_matches) ** beta
    score = fmean * (1.0 - penalty)
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------

def compute_idf(corpus: List[str]) -> Dict[str, float]:
    """Compute IDF for each word: log((N+1) / (df+1)).

    *corpus* is a list of strings (documents).  Each document is tokenized
    with :func:`tokenize_simple`.
    """
    N = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        words = set(tokenize_simple(doc))
        for w in words:
            df[w] += 1

    return {w: math.log((N + 1) / (count + 1)) for w, count in df.items()}


# ---------------------------------------------------------------------------
# CIDEr
# ---------------------------------------------------------------------------

def cider_score(
    hypothesis: str,
    references: List[str],
    n_max: int = 4,
) -> float:
    """Compute CIDEr score between *hypothesis* and *references*.

    For each n in 1..n_max:
      - Compute IDF on the *references* corpus (tokenized n-grams treated as
        joined strings so we can reuse compute_idf).
      - Build TF-IDF vectors for hypothesis and each reference.
      - Cosine similarity between hypothesis vector and mean reference vector.
    Average the similarities over all n, then clip to [0, 10].
    """
    if not references:
        return 0.0

    hyp_tokens = tokenize_simple(hypothesis)
    ref_token_lists = [tokenize_simple(r) for r in references]

    total = 0.0
    for n in range(1, n_max + 1):
        # Build n-gram strings for IDF computation
        def ngram_str(tokens: List[str]) -> str:
            return " ".join(
                "_".join(tokens[i: i + n])
                for i in range(max(0, len(tokens) - n + 1))
            )

        ref_ngram_docs = [ngram_str(tl) for tl in ref_token_lists]
        idf = compute_idf(ref_ngram_docs)

        # If all IDF values are 0 (e.g. single-reference corpus), fall back to
        # uniform weight of 1.0 so that TF alone determines the vector.
        all_idf_zero = all(v == 0.0 for v in idf.values()) if idf else True
        default_idf = 1.0 if all_idf_zero else math.log((len(references) + 1) / 1)

        def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
            ngrams = compute_ngrams(tokens, n)
            total_ng = sum(ngrams.values()) or 1
            vec: Dict[str, float] = {}
            for gram, cnt in ngrams.items():
                key = "_".join(gram)
                tf = cnt / total_ng
                weight = 1.0 if all_idf_zero else idf.get(key, default_idf)
                vec[key] = tf * weight
            return vec

        hyp_vec = tfidf_vec(hyp_tokens)

        # Mean reference vector
        ref_vecs = [tfidf_vec(tl) for tl in ref_token_lists]
        all_keys = set(hyp_vec) | {k for rv in ref_vecs for k in rv}

        if not all_keys:
            continue

        mean_ref: Dict[str, float] = {}
        for k in all_keys:
            mean_ref[k] = sum(rv.get(k, 0.0) for rv in ref_vecs) / len(ref_vecs)

        # Cosine similarity
        dot = sum(hyp_vec.get(k, 0.0) * mean_ref.get(k, 0.0) for k in all_keys)
        hyp_norm = math.sqrt(sum(v * v for v in hyp_vec.values()))
        ref_norm = math.sqrt(sum(v * v for v in mean_ref.values()))

        if hyp_norm == 0.0 or ref_norm == 0.0:
            sim = 0.0
        else:
            sim = dot / (hyp_norm * ref_norm)

        total += sim

    raw = total / n_max
    # Scale to CIDEr range [0, 10]
    scaled = raw * 10.0
    return float(max(0.0, min(10.0, scaled)))


# ---------------------------------------------------------------------------
# Sentence BLEU
# ---------------------------------------------------------------------------

def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Sentence-level BLEU-1..max_n with brevity penalty.

    Uses clipped n-gram precision and geometric mean over orders 1..max_n.
    Returns 0.0 if hypothesis is empty.
    """
    hyp_tokens = tokenize_simple(hypothesis)
    ref_tokens = tokenize_simple(reference)

    if not hyp_tokens:
        return 0.0

    log_avg = 0.0
    active = 0
    for n in range(1, max_n + 1):
        hyp_ng = compute_ngrams(hyp_tokens, n)
        ref_ng = compute_ngrams(ref_tokens, n)
        if not hyp_ng:
            continue
        clipped = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
        total = sum(hyp_ng.values())
        precision = clipped / total if total > 0 else 0.0
        if precision == 0.0:
            return 0.0  # Any zero n-gram precision → BLEU = 0
        log_avg += math.log(precision)
        active += 1

    if active == 0:
        return 0.0

    geo_mean = math.exp(log_avg / active)

    # Brevity penalty
    r = len(ref_tokens)
    c = len(hyp_tokens)
    bp = 1.0 if c >= r else math.exp(1.0 - r / c)

    return float(min(1.0, bp * geo_mean))


# ---------------------------------------------------------------------------
# Aggregate scores
# ---------------------------------------------------------------------------

_METRIC_FN = {
    "bleu": lambda h, r: sentence_bleu(h, r),
    "meteor": lambda h, r: meteor_score(h, r),
    "cider": lambda h, r: cider_score(h, [r]),
}


def aggregate_scores(
    hypotheses: List[str],
    references: List[str],
    metric: str,
) -> Dict[str, float]:
    """Corpus-level mean/min/max for *metric* over aligned hyp/ref pairs.

    *metric* must be one of "bleu", "meteor", "cider".
    Returns {"mean": ..., "min": ..., "max": ...}.
    """
    if metric not in _METRIC_FN:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(_METRIC_FN)}")
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")

    fn = _METRIC_FN[metric]
    scores = [fn(h, r) for h, r in zip(hypotheses, references)]

    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------

class GenerationEvaluator:
    """Convenience evaluator wrapping BLEU, METEOR, and CIDEr."""

    DEFAULT_METRICS = ["bleu", "meteor", "cider"]

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        self.metrics = metrics if metrics is not None else list(self.DEFAULT_METRICS)
        for m in self.metrics:
            if m not in _METRIC_FN:
                raise ValueError(f"Unknown metric '{m}'")

    def score_pair(self, hyp: str, ref: str) -> Dict[str, float]:
        """Score a single (hypothesis, reference) pair with all configured metrics."""
        return {m: _METRIC_FN[m](hyp, ref) for m in self.metrics}

    def score_corpus(
        self,
        hyps: List[str],
        refs: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Return per-metric aggregate stats over aligned corpus pairs."""
        return {m: aggregate_scores(hyps, refs, m) for m in self.metrics}
