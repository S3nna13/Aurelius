"""Aurelius ROUGE scorer: ROUGE-N and ROUGE-L scoring (Lin 2004)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RougeScores:
    rouge1: float
    rouge2: float
    rougeL: float


class RougeScorer:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L scores."""

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase and split on whitespace/punctuation boundaries."""
        return re.findall(r'\w+', text.lower())

    def _ngrams(self, tokens: list[str], n: int) -> list[tuple]:
        """Return all n-grams as a list of tuples via a sliding window."""
        if n <= 0 or len(tokens) < n:
            return []
        return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]

    def _lcs_length(self, a: list[str], b: list[str]) -> int:
        """Compute LCS length between two token lists using dynamic programming."""
        len_a = len(a)
        len_b = len(b)
        # Use two-row rolling array to save memory
        prev = [0] * (len_b + 1)
        curr = [0] * (len_b + 1)
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (len_b + 1)
        return prev[len_b]

    def rouge_n(self, hypothesis: str, reference: str, n: int) -> float:
        """Recall = |matching n-grams| / |reference n-grams|.

        Returns 0.0 if reference has no n-grams.
        """
        ref_tokens = self._tokenize(reference)
        hyp_tokens = self._tokenize(hypothesis)

        ref_ngrams = self._ngrams(ref_tokens, n)
        if not ref_ngrams:
            return 0.0

        hyp_ngrams = self._ngrams(hyp_tokens, n)

        # Count matches using a frequency map
        ref_counts: dict[tuple, int] = {}
        for gram in ref_ngrams:
            ref_counts[gram] = ref_counts.get(gram, 0) + 1

        hyp_counts: dict[tuple, int] = {}
        for gram in hyp_ngrams:
            hyp_counts[gram] = hyp_counts.get(gram, 0) + 1

        matches = 0
        for gram, count in hyp_counts.items():
            matches += min(count, ref_counts.get(gram, 0))

        return matches / len(ref_ngrams)

    def rouge_l(self, hypothesis: str, reference: str) -> float:
        """LCS-based recall = lcs_len / len(ref_tokens).

        Returns 0.0 if reference is empty.
        """
        ref_tokens = self._tokenize(reference)
        if not ref_tokens:
            return 0.0
        hyp_tokens = self._tokenize(hypothesis)
        lcs_len = self._lcs_length(hyp_tokens, ref_tokens)
        return lcs_len / len(ref_tokens)

    def score(self, hypothesis: str, reference: str) -> RougeScores:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L for a single hypothesis/reference pair."""
        return RougeScores(
            rouge1=self.rouge_n(hypothesis, reference, 1),
            rouge2=self.rouge_n(hypothesis, reference, 2),
            rougeL=self.rouge_l(hypothesis, reference),
        )

    def corpus_score(
        self, hypotheses: list[str], references: list[str]
    ) -> RougeScores:
        """Compute mean ROUGE scores across a corpus of hypothesis/reference pairs.

        Raises ValueError if lengths differ.
        """
        if len(hypotheses) != len(references):
            raise ValueError(
                f"hypotheses and references must have the same length, "
                f"got {len(hypotheses)} and {len(references)}"
            )
        if not hypotheses:
            return RougeScores(rouge1=0.0, rouge2=0.0, rougeL=0.0)

        scores = [self.score(h, r) for h, r in zip(hypotheses, references)]
        n = len(scores)
        return RougeScores(
            rouge1=sum(s.rouge1 for s in scores) / n,
            rouge2=sum(s.rouge2 for s in scores) / n,
            rougeL=sum(s.rougeL for s in scores) / n,
        )


ROUGE_SCORER_REGISTRY = {"default": RougeScorer}
