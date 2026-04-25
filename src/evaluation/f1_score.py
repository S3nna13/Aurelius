"""F1 score evaluator for classification tasks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class F1Score:
    """Compute F1, precision, recall for binary classification."""

    def compute(self, true_pos: int, false_pos: int, false_neg: int) -> dict[str, float]:
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    def compute_batch(self, predictions: list[int], references: list[int]) -> dict[str, float]:
        tp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 1)
        fp = sum(1 for p, r in zip(predictions, references) if p == 1 and r == 0)
        fn = sum(1 for p, r in zip(predictions, references) if p == 0 and r == 1)
        return self.compute(tp, fp, fn)


F1_SCORE = F1Score()