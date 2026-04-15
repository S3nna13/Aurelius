"""
metrics.py — Core evaluation metrics for the Aurelius LLM project.

Pure PyTorch for tensor ops; pure Python stdlib for text processing.
"""

from __future__ import annotations

import math
import string
from collections import Counter
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer.

    Split on whitespace, strip punctuation from each token, lowercase, and
    filter empty tokens.  No external libraries are used.
    """
    tokens = []
    for raw in text.split():
        token = raw.lower().strip(string.punctuation)
        if token:
            tokens.append(token)
    return tokens


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Return a Counter of n-gram tuples for the given token list."""
    if n <= 0 or n > len(tokens):
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Corpus BLEU-1 through *max_n* with geometric mean.

    Brevity penalty: ``min(1, exp(1 - ref_len / hyp_len))``.
    With *smooth=True* add 1 to numerator and denominator for every n-gram
    order where the raw clipped count is zero (Add-1 / epsilon smoothing).
    Returns a float in [0, 1].
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if hyp_len == 0:
        return 0.0

    # Brevity penalty
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / hyp_len)

    log_avg = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)

        # Clipped counts
        clipped = sum(
            min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())

        if smooth:
            clipped += 1
            total += 1

        if total == 0:
            return 0.0

        precision = clipped / total
        if precision == 0.0:
            return 0.0
        log_avg += math.log(precision)

    return bp * math.exp(log_avg / max_n)


# ---------------------------------------------------------------------------
# ROUGE-N
# ---------------------------------------------------------------------------

def rouge_n(
    reference: str,
    hypothesis: str,
    n: int = 2,
) -> Dict[str, float]:
    """ROUGE-N precision, recall, and F1.

    Returns a dict with keys ``"precision"``, ``"recall"``, ``"f1"``.
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    ref_ngrams = compute_ngrams(ref_tokens, n)
    hyp_ngrams = compute_ngrams(hyp_tokens, n)

    ref_total = sum(ref_ngrams.values())
    hyp_total = sum(hyp_ngrams.values())

    if ref_total == 0 and hyp_total == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if ref_total == 0 or hyp_total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum(
        min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items()
    )

    precision = overlap / hyp_total
    recall = overlap / ref_total
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute the length of the Longest Common Subsequence of *a* and *b*."""
    m, n = len(a), len(b)
    # Use two-row DP to save memory
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(
    reference: str,
    hypothesis: str,
) -> Dict[str, float]:
    """ROUGE-L precision, recall, and F1 based on LCS.

    Returns a dict with keys ``"precision"``, ``"recall"``, ``"f1"``.
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if ref_len == 0 and hyp_len == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if ref_len == 0 or hyp_len == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs / hyp_len
    recall = lcs / ref_len
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Distinct-N
# ---------------------------------------------------------------------------

def distinct_n(texts: List[str], n: int = 2) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams across all texts."""
    all_ngrams: List[tuple] = []
    for text in texts:
        tokens = tokenize(text)
        ngrams = compute_ngrams(tokens, n)
        all_ngrams.extend(ngrams.elements())

    total = len(all_ngrams)
    if total == 0:
        return 0.0
    unique = len(set(all_ngrams))
    return unique / total


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    log_probs: Tensor,
    lengths: Tensor,
) -> float:
    """Compute perplexity: ``exp(-mean_per_token_log_prob)``.

    Args:
        log_probs: ``(B,)`` tensor of summed log-probabilities per sequence.
        lengths:   ``(B,)`` tensor of token counts per sequence.

    Returns:
        Scalar float perplexity.
    """
    log_probs = log_probs.float()
    lengths = lengths.float()
    per_token_log_prob = log_probs.sum() / lengths.sum()
    return float(torch.exp(-per_token_log_prob).item())


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

class EmbeddingSimilarity:
    """Cosine-similarity metric backed by an arbitrary embedding function."""

    def __init__(self, embed_fn: Callable[[str], Tensor]) -> None:
        self._embed = embed_fn

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between the embeddings of *text_a* and *text_b*."""
        ea = self._embed(text_a).float()
        eb = self._embed(text_b).float()
        cos = torch.nn.functional.cosine_similarity(
            ea.unsqueeze(0), eb.unsqueeze(0)
        )
        return float(cos.item())

    def batch_similarity(
        self, refs: List[str], hyps: List[str]
    ) -> List[float]:
        """Return pairwise cosine similarities for corresponding (ref, hyp) pairs."""
        results: List[float] = []
        for r, h in zip(refs, hyps):
            results.append(self.similarity(r, h))
        return results


# ---------------------------------------------------------------------------
# MetricsEvaluator
# ---------------------------------------------------------------------------

class MetricsEvaluator:
    """High-level evaluator that aggregates multiple metrics over a corpus."""

    def __init__(self, embed_fn: Optional[Callable[[str], Tensor]] = None) -> None:
        self._embed_sim = EmbeddingSimilarity(embed_fn) if embed_fn is not None else None

    def evaluate(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """Compute mean BLEU-4, ROUGE-2 F1, ROUGE-L F1, distinct-2, and (optionally)
        mean embedding similarity across all reference/hypothesis pairs.

        Returns a dict with keys:
            ``"bleu"``, ``"rouge_2_f1"``, ``"rouge_l_f1"``, ``"distinct_2"``,
            and (if embed_fn was provided) ``"embedding_similarity"``.
        """
        bleu_scores: List[float] = []
        rouge2_scores: List[float] = []
        rougel_scores: List[float] = []
        emb_scores: List[float] = []

        for ref, hyp in zip(references, hypotheses):
            bleu_scores.append(bleu_score(ref, hyp, max_n=4, smooth=True))
            rouge2_scores.append(rouge_n(ref, hyp, n=2)["f1"])
            rougel_scores.append(rouge_l(ref, hyp)["f1"])
            if self._embed_sim is not None:
                emb_scores.append(self._embed_sim.similarity(ref, hyp))

        def _mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        metrics: Dict[str, float] = {
            "bleu": _mean(bleu_scores),
            "rouge_2_f1": _mean(rouge2_scores),
            "rouge_l_f1": _mean(rougel_scores),
            "distinct_2": distinct_n(hypotheses, n=2),
        }
        if self._embed_sim is not None:
            metrics["embedding_similarity"] = _mean(emb_scores)

        return metrics

    def get_summary(self, metrics: Dict[str, float]) -> str:
        """Format a metrics dict as a human-readable string."""
        lines = ["Evaluation Metrics:"]
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)
