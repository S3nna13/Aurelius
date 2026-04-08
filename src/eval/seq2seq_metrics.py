"""Sequence-to-sequence evaluation metrics: ROUGE variants, METEOR approximation, chrF."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _char_ngrams(text: str, n: int) -> Counter:
    if n <= 0 or len(text) < n:
        return Counter()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _lcs_length_chars(a: str, b: str) -> int:
    """Character-level LCS length via two-row DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Token-level edit distance (insert / delete / substitute, cost 1 each)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n]


# ---------------------------------------------------------------------------
# 1. ROUGE-N
# ---------------------------------------------------------------------------

def rouge_n(hypothesis: str, reference: str, n: int = 2) -> dict[str, float]:
    """N-gram ROUGE: precision, recall, F1.

    Returns {"rouge_n_precision": float, "rouge_n_recall": float, "rouge_n_f1": float}.
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)

    if not hyp_ngrams and not ref_ngrams:
        return {"rouge_n_precision": 1.0, "rouge_n_recall": 1.0, "rouge_n_f1": 1.0}

    if not hyp_ngrams or not ref_ngrams:
        return {"rouge_n_precision": 0.0, "rouge_n_recall": 0.0, "rouge_n_f1": 0.0}

    overlap = hyp_ngrams & ref_ngrams
    overlap_count = sum(overlap.values())

    precision = overlap_count / sum(hyp_ngrams.values())
    recall = overlap_count / sum(ref_ngrams.values())

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "rouge_n_precision": precision,
        "rouge_n_recall": recall,
        "rouge_n_f1": f1,
    }


# ---------------------------------------------------------------------------
# 2. ROUGE-L (character-level LCS)
# ---------------------------------------------------------------------------

def rouge_l(hypothesis: str, reference: str) -> dict[str, float]:
    """LCS-based ROUGE-L using character-level LCS.

    Returns {"rouge_l_precision": float, "rouge_l_recall": float, "rouge_l_f1": float}.
    """
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)

    if not hyp and not ref:
        return {"rouge_l_precision": 1.0, "rouge_l_recall": 1.0, "rouge_l_f1": 1.0}
    if not hyp or not ref:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    lcs = _lcs_length_chars(hyp, ref)

    if lcs == 0:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": f1,
    }


# ---------------------------------------------------------------------------
# 3. ROUGE-W (Weighted LCS)
# ---------------------------------------------------------------------------

def rouge_w(hypothesis: str, reference: str, weight: float = 1.2) -> float:
    """Weighted LCS ROUGE — consecutive matches weighted more heavily.

    Uses the WLCS scoring function: consecutive matches accumulate a score of
    k^weight instead of k, so runs of length k score k^weight vs k*1^weight.

    Returns F1 score as a float.
    """
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)

    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0

    m, n = len(hyp), len(ref)

    # WLCS DP: c[i][j] = wlcs score ending at hyp[i-1], ref[j-1]
    # f[i][j] = cumulative WLCS score
    c = [[0.0] * (n + 1) for _ in range(m + 1)]
    w = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                k = c[i - 1][j - 1]  # consecutive run length before this match
                c[i][j] = k + 1
                w[i][j] = w[i - 1][j - 1] + (k + 1) ** weight - k ** weight
            else:
                c[i][j] = 0.0
                w[i][j] = max(w[i - 1][j], w[i][j - 1])

    wlcs = w[m][n]

    # Normalisation factors: f(|hyp|) and f(|ref|) using same power function
    # f(x) = x^weight as the "perfect" score for a sequence of length x
    norm_hyp = m ** weight
    norm_ref = n ** weight

    if norm_hyp == 0 or norm_ref == 0:
        return 0.0

    precision = wlcs / norm_hyp
    recall = wlcs / norm_ref

    if precision + recall == 0:
        return 0.0

    f1 = (1 + weight ** 2) * precision * recall / (weight ** 2 * precision + recall)
    return min(f1, 1.0)


# ---------------------------------------------------------------------------
# 4. METEOR (approximation, exact match only)
# ---------------------------------------------------------------------------

def meteor_score(
    hypothesis: str,
    reference: str,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """METEOR approximation (exact match, no stemming / WordNet).

    Score = (1 - penalty) * P*R / (alpha*P + (1-alpha)*R)
    penalty = gamma * (chunks / matched_unigrams)^beta
    fragmentation: number of contiguous matched blocks in hypothesis order.
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)

    # Greedy exact-match: mark each matched reference token as used
    matched_hyp_indices: list[int] = []
    ref_available = dict(ref_counter)

    for i, token in enumerate(hyp_tokens):
        if ref_available.get(token, 0) > 0:
            matched_hyp_indices.append(i)
            ref_available[token] -= 1

    matched = len(matched_hyp_indices)

    if matched == 0:
        return 0.0

    precision = matched / len(hyp_tokens)
    recall = matched / len(ref_tokens)

    if alpha * precision + (1 - alpha) * recall == 0:
        return 0.0

    # Count contiguous chunks in hypothesis (by index gaps)
    chunks = 1
    for k in range(1, len(matched_hyp_indices)):
        if matched_hyp_indices[k] != matched_hyp_indices[k - 1] + 1:
            chunks += 1

    fragmentation = chunks / matched
    penalty = gamma * (fragmentation ** beta)

    score = (1 - penalty) * (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    return max(0.0, score)


# ---------------------------------------------------------------------------
# 5. chrF
# ---------------------------------------------------------------------------

def chrf_score(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF).

    Averages precision and recall over n-gram orders 1..n, then computes F_beta.
    Returns float in [0, 1].
    """
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)

    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    active = 0

    for order in range(1, n + 1):
        hyp_ngrams = _char_ngrams(hyp, order)
        ref_ngrams = _char_ngrams(ref, order)

        if not hyp_ngrams or not ref_ngrams:
            continue

        overlap = hyp_ngrams & ref_ngrams
        overlap_count = sum(overlap.values())

        total_precision += overlap_count / sum(hyp_ngrams.values())
        total_recall += overlap_count / sum(ref_ngrams.values())
        active += 1

    if active == 0:
        return 0.0

    avg_precision = total_precision / active
    avg_recall = total_recall / active

    denom = beta ** 2 * avg_precision + avg_recall
    if denom == 0:
        return 0.0

    return (1 + beta ** 2) * avg_precision * avg_recall / denom


# ---------------------------------------------------------------------------
# 6. TER
# ---------------------------------------------------------------------------

def ter_score(hypothesis: str, reference: str) -> float:
    """Translation Edit Rate.

    edit_distance(hyp_tokens, ref_tokens) / len(ref_tokens).
    Returns float; 0.0 = perfect match, >1.0 = worse than reference length.
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not ref_tokens:
        return 0.0 if not hyp_tokens else float(len(hyp_tokens))

    ed = _edit_distance(hyp_tokens, ref_tokens)
    return ed / len(ref_tokens)


# ---------------------------------------------------------------------------
# 7. Seq2SeqMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class Seq2SeqMetrics:
    """Aggregated seq2seq evaluation scores."""

    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    meteor: float = 0.0
    chrf: float = 0.0
    ter: float = 0.0

    @property
    def overall(self) -> float:
        """Macro-average of the five primary metrics (excludes TER)."""
        return (self.rouge_1 + self.rouge_2 + self.rouge_l + self.meteor + self.chrf) / 5


# ---------------------------------------------------------------------------
# 8. evaluate_seq2seq
# ---------------------------------------------------------------------------

def evaluate_seq2seq(hypothesis: str, reference: str) -> Seq2SeqMetrics:
    """Compute all seq2seq metrics for a single hypothesis/reference pair."""
    r1 = rouge_n(hypothesis, reference, n=1)["rouge_n_f1"]
    r2 = rouge_n(hypothesis, reference, n=2)["rouge_n_f1"]
    rl = rouge_l(hypothesis, reference)["rouge_l_f1"]
    m = meteor_score(hypothesis, reference)
    c = chrf_score(hypothesis, reference)
    t = ter_score(hypothesis, reference)

    return Seq2SeqMetrics(
        rouge_1=r1,
        rouge_2=r2,
        rouge_l=rl,
        meteor=m,
        chrf=c,
        ter=t,
    )


# ---------------------------------------------------------------------------
# 9. corpus_seq2seq_metrics
# ---------------------------------------------------------------------------

def corpus_seq2seq_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Average each metric over all hypothesis/reference pairs.

    Returns {"rouge_1", "rouge_2", "rouge_l", "meteor", "chrf", "ter"}.
    """
    n = len(hypotheses)
    if n == 0:
        return {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "meteor": 0.0,
            "chrf": 0.0,
            "ter": 0.0,
        }

    totals: dict[str, float] = {
        "rouge_1": 0.0,
        "rouge_2": 0.0,
        "rouge_l": 0.0,
        "meteor": 0.0,
        "chrf": 0.0,
        "ter": 0.0,
    }

    for hyp, ref in zip(hypotheses, references):
        m = evaluate_seq2seq(hyp, ref)
        totals["rouge_1"] += m.rouge_1
        totals["rouge_2"] += m.rouge_2
        totals["rouge_l"] += m.rouge_l
        totals["meteor"] += m.meteor
        totals["chrf"] += m.chrf
        totals["ter"] += m.ter

    return {k: v / n for k, v in totals.items()}
