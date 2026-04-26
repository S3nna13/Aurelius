from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BertScoreResult:
    precision: float
    recall: float
    f1: float


class BertScoreScorer:
    def score(self, reference: str, hypothesis: str) -> BertScoreResult:
        if not reference or not hypothesis:
            return BertScoreResult(0.0, 0.0, 0.0)
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not ref_tokens or not hyp_tokens:
            return BertScoreResult(0.0, 0.0, 0.0)
        overlap = ref_tokens & hyp_tokens
        precision = len(overlap) / len(hyp_tokens)
        recall = len(overlap) / len(ref_tokens)
        if precision + recall == 0:
            return BertScoreResult(0.0, 0.0, 0.0)
        f1 = 2 * precision * recall / (precision + recall)
        return BertScoreResult(precision=precision, recall=recall, f1=f1)


BERT_SCORE = BertScoreScorer()
