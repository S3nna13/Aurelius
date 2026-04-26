"""Aurelius F1 scorer: token-level F1 for QA evaluation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class F1Result:
    f1: float
    precision: float
    recall: float
    common_tokens: int
    predicted_tokens: int
    gold_tokens: int


class F1Scorer:
    """Computes token-level F1 score between predicted and gold answers."""

    def _normalize(self, text: str) -> list[str]:
        """Lowercase, remove punctuation, and split into tokens."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def score(self, prediction: str, gold: str) -> F1Result:
        """Compute token-level F1 between prediction and gold.

        Both empty → f1=1.0. One empty and other non-empty → f1=0.0.
        """
        pred_tokens = self._normalize(prediction)
        gold_tokens = self._normalize(gold)

        if not pred_tokens and not gold_tokens:
            return F1Result(
                f1=1.0,
                precision=1.0,
                recall=1.0,
                common_tokens=0,
                predicted_tokens=0,
                gold_tokens=0,
            )

        pred_counter = Counter(pred_tokens)
        gold_counter = Counter(gold_tokens)

        common: Counter = pred_counter & gold_counter
        common_count = sum(common.values())

        precision = common_count / len(pred_tokens) if pred_tokens else 0.0
        recall = common_count / len(gold_tokens) if gold_tokens else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return F1Result(
            f1=f1,
            precision=precision,
            recall=recall,
            common_tokens=common_count,
            predicted_tokens=len(pred_tokens),
            gold_tokens=len(gold_tokens),
        )

    def batch_score(self, predictions: list[str], golds: list[str]) -> list[F1Result]:
        """Compute F1 for each prediction/gold pair.

        Raises ValueError if lengths differ.
        """
        if len(predictions) != len(golds):
            raise ValueError(
                f"predictions and golds must have the same length, "
                f"got {len(predictions)} and {len(golds)}"
            )
        return [self.score(p, g) for p, g in zip(predictions, golds)]

    def mean_f1(self, results: list[F1Result]) -> float:
        """Compute mean F1 over a list of F1Result objects. Returns 0.0 if empty."""
        if not results:
            return 0.0
        return sum(r.f1 for r in results) / len(results)


F1_SCORER_REGISTRY = {"default": F1Scorer}
