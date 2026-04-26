"""GSM8K Scorer: grade model answers on the GSM8K math benchmark.

GSM8K (Cobbe et al., 2021) uses a "####" separator to mark the final numeric
answer in a chain-of-thought solution.  This module provides:

  * ``GSM8KAnswer``  — parsed answer dataclass
  * ``GSM8KScorer`` — extract, compare, and batch-score predictions
  * ``BENCHMARK_REGISTRY["gsm8k"]`` — singleton scorer instance

Only stdlib + torch (for dtype compatibility if needed) are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GSM8KAnswer:
    """Parsed representation of a model or ground-truth answer.

    Attributes:
        raw_text:         The original response string.
        extracted_number: The numeric answer that was extracted, or ``None``
                          if extraction failed.
        steps:            List of reasoning lines (sentences/steps) preceding
                          the final answer.
    """

    raw_text: str
    extracted_number: float | None
    steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class GSM8KScorer:
    """Score model predictions on the GSM8K benchmark.

    The scorer follows the official GSM8K evaluation convention:

    1. Split on ``####`` to locate the final answer block.
    2. Extract the last float-parseable number from that block.
    3. Fallback: scan the whole text for the last float-parseable token.
    """

    # Regex: optional leading $, digits with optional commas, optional decimal
    _NUMBER_RE = re.compile(r"-?[\$]?[\d,]+(?:\.\d+)?")

    # ---------------------------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------------------------

    def _parse_number(self, text: str) -> float | None:
        """Return the *last* parseable number in *text*, or ``None``."""
        # Remove commas (e.g. "1,234" → "1234") and $ signs before parsing
        candidates = self._NUMBER_RE.findall(text)
        for raw in reversed(candidates):
            cleaned = raw.replace(",", "").lstrip("$")
            try:
                return float(cleaned)
            except ValueError:
                continue
        return None

    def extract_answer(self, text: str) -> GSM8KAnswer:
        """Parse a response string into a ``GSM8KAnswer``.

        Strategy:
        1. Split on ``####``; if found, use everything after it as the answer
           block.
        2. Extract reasoning steps as non-empty lines *before* the separator.
        3. Fallback if ``####`` is absent: treat the full text as the answer
           block and derive steps from sentences.

        Args:
            text: Raw model output string.

        Returns:
            ``GSM8KAnswer`` with extracted number and steps.
        """
        if "####" in text:
            parts = text.split("####", 1)
            step_block, answer_block = parts[0], parts[1]
            steps = [ln.strip() for ln in step_block.splitlines() if ln.strip()]
            number = self._parse_number(answer_block)
        else:
            # Fallback: scan whole text for last number
            steps = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
            number = self._parse_number(text)

        return GSM8KAnswer(
            raw_text=text,
            extracted_number=number,
            steps=steps,
        )

    # ---------------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------------

    def score(self, predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
        """Return ``True`` iff the predicted answer matches the ground truth.

        Numeric comparison with an absolute tolerance of *tolerance* (default
        0.01, matching the official GSM8K evaluator).

        Args:
            predicted:    Raw model prediction string.
            ground_truth: Reference answer string.
            tolerance:    Absolute tolerance for floating-point comparison.

        Returns:
            ``True`` if both answers parse to numbers within *tolerance*,
            ``False`` otherwise.
        """
        pred_ans = self.extract_answer(predicted)
        gt_ans = self.extract_answer(ground_truth)

        if pred_ans.extracted_number is None or gt_ans.extracted_number is None:
            return False

        return abs(pred_ans.extracted_number - gt_ans.extracted_number) <= tolerance

    def batch_score(
        self,
        predictions: list[str],
        ground_truths: list[str],
        tolerance: float = 0.01,
    ) -> dict:
        """Score a batch of predictions.

        Args:
            predictions:   List of raw model prediction strings.
            ground_truths: List of reference answer strings (same length).
            tolerance:     Numeric match tolerance.

        Returns:
            Dictionary with keys:
            - ``accuracy``  (float): Fraction of correct predictions.
            - ``n_correct`` (int):   Number of correct predictions.
            - ``n_total``   (int):   Total number of examples.
            - ``avg_steps`` (float): Average reasoning steps per prediction.

        Raises:
            ValueError: If *predictions* and *ground_truths* have different lengths.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"predictions and ground_truths must have the same length, "
                f"got {len(predictions)} vs {len(ground_truths)}"
            )

        n_total = len(predictions)
        if n_total == 0:
            return {
                "accuracy": 0.0,
                "n_correct": 0,
                "n_total": 0,
                "avg_steps": 0.0,
            }

        n_correct = 0
        total_steps = 0
        for pred, gt in zip(predictions, ground_truths):
            if self.score(pred, gt, tolerance=tolerance):
                n_correct += 1
            parsed = self.extract_answer(pred)
            total_steps += len(parsed.steps)

        return {
            "accuracy": n_correct / n_total,
            "n_correct": n_correct,
            "n_total": n_total,
            "avg_steps": total_steps / n_total,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

try:
    from src.eval import BENCHMARK_REGISTRY  # type: ignore[attr-defined]
except Exception:
    BENCHMARK_REGISTRY: dict = {}  # type: ignore[assignment]

BENCHMARK_REGISTRY["gsm8k"] = GSM8KScorer()
