from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PairwiseResult:
    winner: str = "tie"
    score_a: float = 0.0
    score_b: float = 0.0


class PairwiseEvaluator:
    def compare(self, response_a: str, response_b: str) -> PairwiseResult:
        if not response_a and not response_b:
            return PairwiseResult()
        len_a, len_b = len(response_a.split()), len(response_b.split())
        if len_a > len_b:
            return PairwiseResult(winner="A", score_a=1.0, score_b=0.5)
        elif len_b > len_a:
            return PairwiseResult(winner="B", score_a=0.5, score_b=1.0)
        return PairwiseResult(
            winner="A" if response_a > response_b else "B",
            score_a=1.0 if response_a > response_b else 0.5,
            score_b=0.5 if response_a > response_b else 1.0,
        )


PAIRWISE_EVALUATOR = PairwiseEvaluator()
