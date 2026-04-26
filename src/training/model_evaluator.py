"""Model evaluation and deployment gate (Aurelius).

Pure-Python port of Heavens_Gate TRAINING_PIPELINE/model_evaluator.py --
no torch/transformers dependency. Evaluates a set of predictions vs
references and compares two eval results to detect regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalResult:
    """Metrics from a single model evaluation run."""

    adapter_path: str
    perplexity: float = 0.0
    accuracy: float = 0.0
    n_samples: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison of old vs new adapter."""

    old: EvalResult
    new: EvalResult
    regressed: bool = False
    perplexity_delta_pct: float = 0.0  # positive = new is worse
    accuracy_delta_pct: float = 0.0  # positive = new is better
    rejection_reason: str = ""


class ModelEvaluator:
    """Evaluate and compare model adapters.

    Args:
        max_regression_pct: Max allowed perplexity regression (%) before
            flagging the new adapter as regressed.
        min_accuracy: Optional floor; a new adapter below this accuracy
            is regressed regardless of delta (0.0 disables the check).
    """

    def __init__(self, max_regression_pct: float = 5.0, min_accuracy: float = 0.0) -> None:
        self.max_regression_pct = max_regression_pct
        self.min_accuracy = min_accuracy

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compare(self, old: EvalResult, new: EvalResult) -> ComparisonResult:
        """Compare two eval results, flagging regressions."""
        perplexity_delta_pct = 0.0
        accuracy_delta_pct = 0.0

        if old.perplexity > 0:
            perplexity_delta_pct = (new.perplexity - old.perplexity) / old.perplexity * 100.0
        if old.accuracy > 0:
            accuracy_delta_pct = (new.accuracy - old.accuracy) / old.accuracy * 100.0

        regressed = False
        rejection_reason = ""

        if perplexity_delta_pct > self.max_regression_pct:
            regressed = True
            rejection_reason = (
                f"Perplexity increased by {perplexity_delta_pct:.1f}% "
                f"(threshold: {self.max_regression_pct}%)"
            )
        elif self.min_accuracy > 0 and new.accuracy < self.min_accuracy:
            regressed = True
            rejection_reason = f"Accuracy {new.accuracy:.3f} below floor {self.min_accuracy:.3f}"

        return ComparisonResult(
            old=old,
            new=new,
            regressed=regressed,
            perplexity_delta_pct=perplexity_delta_pct,
            accuracy_delta_pct=accuracy_delta_pct,
            rejection_reason=rejection_reason,
        )

    def evaluate_samples(
        self,
        predictions: list[str],
        references: list[str],
        adapter_path: str = "",
    ) -> EvalResult:
        """Evaluate a list of predictions against references.

        Uses exact (case-insensitive, stripped) match for accuracy, and a
        simulated "perplexity" proxy equal to the mean token count per
        prediction (a cheap length-based surrogate -- 1.0 if empty).
        """
        errors: list[str] = []
        if len(predictions) != len(references):
            errors.append(
                f"length mismatch: predictions={len(predictions)} references={len(references)}"
            )
            return EvalResult(adapter_path=adapter_path, errors=errors)

        n = len(predictions)
        if n == 0:
            return EvalResult(adapter_path=adapter_path, perplexity=1.0, accuracy=0.0, n_samples=0)

        correct = 0
        total_tokens = 0
        for pred, ref in zip(predictions, references):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
            total_tokens += len(pred.split())

        accuracy = correct / n
        perplexity = total_tokens / n if total_tokens > 0 else 1.0
        return EvalResult(
            adapter_path=adapter_path,
            perplexity=perplexity,
            accuracy=accuracy,
            n_samples=n,
        )

    def is_deployment_safe(self, result: EvalResult, baseline: EvalResult | None = None) -> bool:
        """Return True if *result* has no errors and doesn't regress vs baseline."""
        if result.errors:
            return False
        if baseline is None:
            return True
        return not self.compare(baseline, result).regressed


MODEL_EVALUATOR_REGISTRY = {"default": ModelEvaluator}
