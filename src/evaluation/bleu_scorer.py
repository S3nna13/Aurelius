"""Aurelius BLEU scorer: BLEU score (Papineni et al. 2002)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BLEUResult:
    score: float
    precisions: list[float]
    brevity_penalty: float
    hypothesis_len: int
    reference_len: int


class BLEUScorer:
    """Computes BLEU score for hypothesis/reference pairs."""

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase and extract word tokens."""
        return re.findall(r"\w+", text.lower())

    def _ngrams(self, tokens: list[str], n: int) -> dict[tuple, int]:
        """Return a Counter-style dict of n-gram counts."""
        counts: dict[tuple, int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    def _clipped_precision(self, hyp_tokens: list[str], ref_tokens: list[str], n: int) -> float:
        """Clipped precision for order n.

        For each hypothesis n-gram, clip count to the max count in the
        reference. Returns 0.0 if hypothesis has no n-grams.
        """
        hyp_ngrams = self._ngrams(hyp_tokens, n)
        ref_ngrams = self._ngrams(ref_tokens, n)

        total_hyp = sum(hyp_ngrams.values())
        if total_hyp == 0:
            return 0.0

        clipped_sum = 0
        for gram, hyp_count in hyp_ngrams.items():
            ref_count = ref_ngrams.get(gram, 0)
            clipped_sum += min(hyp_count, ref_count)

        return clipped_sum / max(total_hyp, 1)

    def score(self, hypothesis: str, reference: str, max_n: int = 4) -> BLEUResult:
        """Compute BLEU for a single hypothesis/reference pair."""
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)

        hyp_len = len(hyp_tokens)
        ref_len = len(ref_tokens)

        precisions: list[float] = []
        for n in range(1, max_n + 1):
            p = self._clipped_precision(hyp_tokens, ref_tokens, n)
            precisions.append(p)

        # Geometric mean via log sum
        log_sum = sum(math.log(p + 1e-10) for p in precisions)
        geo_mean = math.exp(log_sum / max_n)

        # Brevity penalty
        effective_hyp_len = max(hyp_len, 1)
        if hyp_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / effective_hyp_len)

        bleu = bp * geo_mean
        return BLEUResult(
            score=bleu,
            precisions=precisions,
            brevity_penalty=bp,
            hypothesis_len=hyp_len,
            reference_len=ref_len,
        )

    def corpus_score(
        self,
        hypotheses: list[str],
        references: list[str],
        max_n: int = 4,
    ) -> BLEUResult:
        """Compute mean BLEU score across a corpus of hypothesis/reference pairs.

        Raises ValueError if lengths differ.
        """
        if len(hypotheses) != len(references):
            raise ValueError(
                f"hypotheses and references must have the same length, "
                f"got {len(hypotheses)} and {len(references)}"
            )
        if not hypotheses:
            return BLEUResult(
                score=0.0,
                precisions=[0.0] * max_n,
                brevity_penalty=1.0,
                hypothesis_len=0,
                reference_len=0,
            )

        results = [self.score(h, r, max_n) for h, r in zip(hypotheses, references)]
        n = len(results)
        mean_score = sum(r.score for r in results) / n
        mean_bp = sum(r.brevity_penalty for r in results) / n
        mean_hyp_len = sum(r.hypothesis_len for r in results) // n
        mean_ref_len = sum(r.reference_len for r in results) // n
        mean_precisions = [sum(r.precisions[i] for r in results) / n for i in range(max_n)]
        return BLEUResult(
            score=mean_score,
            precisions=mean_precisions,
            brevity_penalty=mean_bp,
            hypothesis_len=mean_hyp_len,
            reference_len=mean_ref_len,
        )


BLEU_SCORER_REGISTRY = {"default": BLEUScorer}
