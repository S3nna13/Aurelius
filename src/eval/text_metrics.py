"""Text generation evaluation metrics.

Implements exact match, token F1, ROUGE-L, and BLEU from scratch.
All string normalization: lowercase + strip punctuation + collapse whitespace.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalized prediction == normalized reference, else 0.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1: harmonic mean of precision and recall over word bags.

    Precision = |pred_tokens ∩ ref_tokens| / |pred_tokens|
    Recall    = |pred_tokens ∩ ref_tokens| / |ref_tokens|
    F1        = 2 * P * R / (P + R)

    Token overlap uses Counter intersection (handles repeated tokens).
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Counter intersection: min of counts for each token
    common = pred_counter & ref_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length. O(|a| * |b|)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0

    # Use two-row DP to save memory
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


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence.

    Precision = LCS / |pred_tokens|
    Recall    = LCS / |ref_tokens|
    F1        = 2 * P * R / (P + R)
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)

    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def ngrams(tokens: list[str], n: int) -> Counter:
    """Count n-grams in token list."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(grams)


def bleu(
    prediction: str,
    references: list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Corpus-level BLEU score (single sentence, multiple references).

    Modified n-gram precision clipped to max reference count.
    Brevity penalty: exp(1 - ref_len/pred_len) if pred_len < ref_len, else 1.
    Geometric mean of n-gram precisions with uniform weights.
    If smooth=True, add +1 smoothing for n-gram orders with 0 matches (Lin et al.).
    """
    pred_tokens = normalize_text(prediction).split()
    ref_token_lists = [normalize_text(r).split() for r in references]

    pred_len = len(pred_tokens)

    if pred_len == 0:
        return 0.0

    # Best reference length: closest to prediction length
    ref_len = min((abs(len(r) - pred_len), len(r)) for r in ref_token_lists)[1]

    # Brevity penalty
    if pred_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len)

    # Compute modified n-gram precision for each order.
    # Only include orders where prediction is long enough to form ngrams.
    log_avg = 0.0
    active_orders = 0

    for n in range(1, max_n + 1):
        pred_ngrams = ngrams(pred_tokens, n)
        if not pred_ngrams:
            # Prediction is too short to form ngrams of this order — skip.
            # (Not counted, not penalized.)
            continue

        active_orders += 1

        # Clip counts to max across references
        clipped_count = 0
        for gram, count in pred_ngrams.items():
            max_ref_count = max(
                ngrams(ref_tokens, n).get(gram, 0) for ref_tokens in ref_token_lists
            )
            clipped_count += min(count, max_ref_count)

        total_pred_ngrams = sum(pred_ngrams.values())

        if clipped_count == 0:
            if smooth:
                # Epsilon smoothing: very small positive value to avoid log(0)
                # while keeping score near zero for truly non-overlapping text.
                precision = 1e-7 / total_pred_ngrams
            else:
                return 0.0
        else:
            precision = clipped_count / total_pred_ngrams

        log_avg += math.log(precision)

    if active_orders == 0:
        return 0.0

    return bp * math.exp(log_avg / active_orders)


def corpus_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute all metrics over lists of predictions and references.

    Returns dict with keys: exact_match, token_f1, rouge_l, bleu.
    Values are corpus averages (mean over examples).
    """
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "token_f1": 0.0, "rouge_l": 0.0, "bleu": 0.0}

    em_total = 0.0
    f1_total = 0.0
    rl_total = 0.0
    bleu_total = 0.0

    for pred, ref in zip(predictions, references):
        em_total += exact_match(pred, ref)
        f1_total += token_f1(pred, ref)
        rl_total += rouge_l(pred, ref)
        bleu_total += bleu(pred, [ref])

    return {
        "exact_match": em_total / n,
        "token_f1": f1_total / n,
        "rouge_l": rl_total / n,
        "bleu": bleu_total / n,
    }
