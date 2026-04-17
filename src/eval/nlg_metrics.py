"""
Natural Language Generation evaluation metrics.
Pure stdlib + torch only — no external NLP libraries.

Implements: BLEU, ROUGE (N, L, W), METEOR-lite, SemanticSimilarity (BERTScore-lite).
"""

import math
from collections import Counter
from typing import List, Tuple, Dict

import torch


# ---------------------------------------------------------------------------
# NGramCounter
# ---------------------------------------------------------------------------

class NGramCounter:
    """Count n-grams in a token sequence."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    def count(self, tokens: list) -> Counter:
        """Return Counter of n-gram tuples for *tokens*."""
        ngrams: Counter = Counter()
        for i in range(len(tokens) - self.n + 1):
            gram = tuple(tokens[i : i + self.n])
            ngrams[gram] += 1
        return ngrams

    def clip_count(self, hyp_counts: Counter, ref_counts: Counter) -> int:
        """Clipped count: sum of min(hyp_count, ref_count) over shared n-grams."""
        total = 0
        for gram, hyp_c in hyp_counts.items():
            ref_c = ref_counts.get(gram, 0)
            total += min(hyp_c, ref_c)
        return total


# ---------------------------------------------------------------------------
# BLEUScore
# ---------------------------------------------------------------------------

class BLEUScore:
    """BLEU metric (Papineni et al. 2002)."""

    def __init__(self, max_n: int = 4, smooth: bool = True) -> None:
        self.max_n = max_n
        self.smooth = smooth
        self._counters = [NGramCounter(n) for n in range(1, max_n + 1)]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _brevity_penalty(self, hyp_len: int, ref_len: int) -> float:
        """BP = min(1, exp(1 - ref_len / hyp_len)).  Returns 0 if hyp_len==0."""
        if hyp_len == 0:
            return 0.0
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1.0 - ref_len / hyp_len)

    def bleu_n(self, hypothesis: List[str], reference: List[str], n: int) -> float:
        """n-gram precision for a single sentence pair."""
        if not hypothesis:
            return 0.0
        counter = NGramCounter(n)
        hyp_counts = counter.count(hypothesis)
        ref_counts = counter.count(reference)
        clipped = counter.clip_count(hyp_counts, ref_counts)
        total = sum(hyp_counts.values())
        if total == 0:
            return 0.0
        return clipped / total

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def sentence_bleu(self, hypothesis: List[str], reference: List[str]) -> float:
        """BLEU-N (up to max_n) for a single sentence pair."""
        if not hypothesis:
            return 0.0

        bp = self._brevity_penalty(len(hypothesis), len(reference))
        if bp == 0.0:
            return 0.0

        log_sum = 0.0
        for n in range(1, self.max_n + 1):
            counter = self._counters[n - 1]
            hyp_counts = counter.count(hypothesis)
            ref_counts = counter.count(reference)
            numerator = counter.clip_count(hyp_counts, ref_counts)
            denominator = max(len(hypothesis) - n + 1, 0)

            if self.smooth:
                numerator += 1
                denominator += 1

            if denominator == 0:
                # hypothesis shorter than n-gram order; add smoothed log(1)=0
                log_sum += 0.0
                continue
            if numerator == 0:
                return 0.0

            log_sum += math.log(numerator / denominator)

        return bp * math.exp(log_sum / self.max_n)

    def corpus_bleu(
        self,
        hypotheses: List[List[str]],
        references: List[List[str]],
    ) -> float:
        """Corpus-level BLEU.

        Aggregates clipped counts and total counts across all sentences
        before computing precision (not the mean of sentence BLEUs).
        """
        if not hypotheses:
            return 0.0

        # per-order accumulators
        clipped_totals = [0] * self.max_n
        hyp_totals = [0] * self.max_n
        total_hyp_len = 0
        total_ref_len = 0

        for hyp, ref in zip(hypotheses, references):
            total_hyp_len += len(hyp)
            total_ref_len += len(ref)
            for n in range(1, self.max_n + 1):
                counter = self._counters[n - 1]
                hyp_counts = counter.count(hyp)
                ref_counts = counter.count(ref)
                clipped_totals[n - 1] += counter.clip_count(hyp_counts, ref_counts)
                hyp_totals[n - 1] += max(len(hyp) - n + 1, 0)

        bp = self._brevity_penalty(total_hyp_len, total_ref_len)
        if bp == 0.0:
            return 0.0

        log_sum = 0.0
        for n in range(1, self.max_n + 1):
            numerator = clipped_totals[n - 1]
            denominator = hyp_totals[n - 1]
            if self.smooth:
                numerator += 1
                denominator += 1
            if denominator == 0:
                continue
            if numerator == 0:
                return 0.0
            log_sum += math.log(numerator / denominator)

        return bp * math.exp(log_sum / self.max_n)


# ---------------------------------------------------------------------------
# ROUGEScore
# ---------------------------------------------------------------------------

class ROUGEScore:
    """ROUGE metrics (Lin 2004)."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # ROUGE-N
    # ------------------------------------------------------------------

    def rouge_n(
        self, hypothesis: List[str], reference: List[str], n: int = 2
    ) -> Dict[str, float]:
        """ROUGE-N: precision, recall, f1."""
        if not hypothesis or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        counter = NGramCounter(n)
        hyp_counts = counter.count(hypothesis)
        ref_counts = counter.count(reference)
        overlap = counter.clip_count(hyp_counts, ref_counts)

        hyp_total = sum(hyp_counts.values())
        ref_total = sum(ref_counts.values())

        precision = overlap / hyp_total if hyp_total > 0 else 0.0
        recall = overlap / ref_total if ref_total > 0 else 0.0
        f1 = _f1(precision, recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # ROUGE-L  (LCS-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _lcs_length(x: list, y: list) -> int:
        """Dynamic-programming LCS length."""
        m, n = len(x), len(y)
        # Use two-row DP to save memory
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def rouge_l(
        self, hypothesis: List[str], reference: List[str]
    ) -> Dict[str, float]:
        """ROUGE-L based on LCS."""
        if not hypothesis or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        lcs = self._lcs_length(hypothesis, reference)
        precision = lcs / len(hypothesis) if hypothesis else 0.0
        recall = lcs / len(reference) if reference else 0.0
        f1 = _f1(precision, recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    # ------------------------------------------------------------------
    # ROUGE-W  (weighted LCS)
    # ------------------------------------------------------------------

    @staticmethod
    def _wlcs(x: list, y: list, weight: float) -> float:
        """Weighted LCS score — rewards consecutive matches."""
        m, n = len(x), len(y)
        # c[i][j] = weighted LCS score; w[i][j] = consecutive run length
        c = [[0.0] * (n + 1) for _ in range(m + 1)]
        w = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    k = w[i - 1][j - 1]
                    c[i][j] = c[i - 1][j - 1] + (k + 1) ** weight - k ** weight
                    w[i][j] = k + 1
                else:
                    if c[i - 1][j] >= c[i][j - 1]:
                        c[i][j] = c[i - 1][j]
                        w[i][j] = 0
                    else:
                        c[i][j] = c[i][j - 1]
                        w[i][j] = 0
        return c[m][n]

    def rouge_w(
        self,
        hypothesis: List[str],
        reference: List[str],
        weight: float = 1.2,
    ) -> Dict[str, float]:
        """ROUGE-W: weighted LCS."""
        if not hypothesis or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        wlcs_score = self._wlcs(hypothesis, reference, weight)
        # normalise against self-WLCS (maximum achievable)
        hyp_self = self._wlcs(hypothesis, hypothesis, weight)
        ref_self = self._wlcs(reference, reference, weight)

        precision = wlcs_score / hyp_self if hyp_self > 0 else 0.0
        recall = wlcs_score / ref_self if ref_self > 0 else 0.0
        f1 = _f1(precision, recall)
        return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# METEORScore
# ---------------------------------------------------------------------------

class METEORScore:
    """Simplified METEOR (Banerjee & Lavie 2005) — exact match only."""

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _match_tokens(
        self, hyp: List[str], ref: List[str]
    ) -> List[Tuple[int, int]]:
        """Greedy exact-match alignment, each ref token used at most once."""
        ref_available = list(range(len(ref)))
        matches: List[Tuple[int, int]] = []
        for h_idx, h_tok in enumerate(hyp):
            for r_idx in ref_available:
                if ref[r_idx] == h_tok:
                    matches.append((h_idx, r_idx))
                    ref_available.remove(r_idx)
                    break
        return matches

    @staticmethod
    def _count_chunks(matches: List[Tuple[int, int]]) -> int:
        """Count contiguous chunks in matched pairs (both sides consecutive)."""
        if not matches:
            return 0
        chunks = 1
        for k in range(1, len(matches)):
            h_prev, r_prev = matches[k - 1]
            h_curr, r_curr = matches[k]
            if h_curr != h_prev + 1 or r_curr != r_prev + 1:
                chunks += 1
        return chunks

    def score(self, hypothesis: List[str], reference: List[str]) -> float:
        """METEOR score in [0, 1]."""
        if not hypothesis or not reference:
            return 0.0

        matches = self._match_tokens(hypothesis, reference)
        matched = len(matches)
        if matched == 0:
            return 0.0

        precision = matched / len(hypothesis)
        recall = matched / len(reference)

        denom = self.alpha * precision + (1.0 - self.alpha) * recall
        f_mean = (precision * recall / denom) if denom > 0 else 0.0

        chunks = self._count_chunks(matches)
        # Fragmentation penalty: 0 when all matches are in one contiguous chunk.
        # Use (chunks - 1) so a perfectly-contiguous match incurs no penalty.
        frag = (chunks - 1) / matched if matched > 0 else 0.0
        penalty = self.gamma * frag ** self.beta

        meteor = f_mean * (1.0 - penalty)
        # Clamp to [0, 1] for numerical safety
        return max(0.0, min(1.0, meteor))


# ---------------------------------------------------------------------------
# SemanticSimilarityScore  (BERTScore-lite, no pretrained model)
# ---------------------------------------------------------------------------

class SemanticSimilarityScore:
    """Embedding-based similarity via char-hash embeddings."""

    def __init__(self, embed_dim: int = 16) -> None:
        self.embed_dim = embed_dim

    def embed(self, tokens: List[str]) -> torch.Tensor:
        """Return (len, embed_dim) unit-normed tensor.

        For each token: embedding[d] = sum over chars of sin(ord(c) * 0.01 * (i+1))
        where i is the char index within the token (0-based).
        The dimension index d is used to seed per-dimension offsets via sin/cos mix.
        """
        if not tokens:
            return torch.zeros(0, self.embed_dim)

        vecs = []
        for token in tokens:
            vec = torch.zeros(self.embed_dim)
            for i, c in enumerate(token):
                val = math.sin(ord(c) * 0.01 * (i + 1))
                # Distribute across all dimensions with dimension-varying phase
                for d in range(self.embed_dim):
                    vec[d] += math.sin(ord(c) * 0.01 * (i + 1) + d * 0.5)
            # Unit-norm
            norm = vec.norm()
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)

        return torch.stack(vecs)  # (len, embed_dim)

    def greedy_match(
        self, hyp_embeddings: torch.Tensor, ref_embeddings: torch.Tensor
    ) -> float:
        """Mean of max cosine similarities (hyp token → best ref token)."""
        if hyp_embeddings.shape[0] == 0 or ref_embeddings.shape[0] == 0:
            return 0.0
        # cosine similarity matrix: (|hyp|, |ref|)
        sim = torch.mm(hyp_embeddings, ref_embeddings.t())
        max_sims = sim.max(dim=1).values
        return max_sims.mean().item()

    def score(
        self, hypothesis: List[str], reference: List[str]
    ) -> Dict[str, float]:
        """BERTScore-lite: precision, recall, f1."""
        if not hypothesis or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        hyp_emb = self.embed(hypothesis)
        ref_emb = self.embed(reference)

        precision = self.greedy_match(hyp_emb, ref_emb)
        recall = self.greedy_match(ref_emb, hyp_emb)
        f1 = _f1(precision, recall)
        return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _f1(precision: float, recall: float, beta: float = 1.0) -> float:
    denom = beta**2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta**2) * precision * recall / denom
