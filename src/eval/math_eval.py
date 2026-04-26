"""
math_eval.py — MATH/AIME answer extraction and accuracy evaluation.

Supports:
- \\boxed{} extraction (LaTeX math benchmark convention)
- Last-number fallback extraction
- Fraction normalization (e.g. "3/7" → 0.4286)
- Tolerance-based numeric comparison
- Per-category accuracy breakdown
- AIME-specific integer scoring (answers 0–999)

No dependencies beyond Python stdlib + dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MathEvalConfig:
    numeric_tolerance: float = 1e-6  # absolute tolerance for float comparison
    extract_boxed: bool = True  # try \\boxed{} first
    extract_last_number: bool = True  # fallback: last number in response
    normalize_fractions: bool = True  # convert "p/q" → float


# ---------------------------------------------------------------------------
# Answer container
# ---------------------------------------------------------------------------


@dataclass
class MathAnswer:
    raw_text: str  # original model response
    extracted: str | None  # extracted answer string (stripped)
    numeric: float | None  # parsed float, if applicable
    is_fraction: bool = False  # True when extracted answer was a fraction


# ---------------------------------------------------------------------------
# Regex helpers (compiled once at module load)
# ---------------------------------------------------------------------------

# Matches \\boxed{...} with balanced-ish single-level braces.
# Handles nested braces up to 2 levels deep to cover common cases.
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

# Matches integers and decimals (positive and negative).
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Matches a fraction like "3/7" or "-1/2" (no spaces around slash).
_FRACTION_RE = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class MathEval:
    """Evaluator for MATH/AIME-style benchmarks."""

    def __init__(self, config: MathEvalConfig | None = None) -> None:
        self.config = config if config is not None else MathEvalConfig()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_answer(self, response: str) -> MathAnswer:
        """Extract the answer from a model response string.

        Priority:
        1. \\boxed{} if config.extract_boxed is True.
        2. Last number in text if config.extract_last_number is True.
        3. Otherwise return raw response stripped.

        Attempts to parse the extracted string as a float (handles fractions).
        """
        extracted: str | None = None
        numeric: float | None = None
        is_fraction: bool = False

        if self.config.extract_boxed:
            match = _BOXED_RE.search(response)
            if match:
                extracted = match.group(1).strip()

        if extracted is None and self.config.extract_last_number:
            numbers = _NUMBER_RE.findall(response)
            if numbers:
                extracted = numbers[-1]

        if extracted is None:
            # Neither boxed nor number found — return raw stripped text
            raw_stripped = response.strip() or None
            return MathAnswer(
                raw_text=response,
                extracted=raw_stripped,
                numeric=None,
                is_fraction=False,
            )

        # Parse numeric value from extracted string
        norm_str, numeric = self.normalize_answer(extracted)
        frac_match = _FRACTION_RE.match(extracted.strip())
        is_fraction = frac_match is not None and self.config.normalize_fractions

        return MathAnswer(
            raw_text=response,
            extracted=extracted,
            numeric=numeric,
            is_fraction=is_fraction,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_answer(self, answer: str) -> tuple[str, float | None]:
        """Normalize an answer string.

        Returns:
            (normalized_str, numeric_value)

        normalized_str is lowercased and whitespace-stripped.
        numeric_value is a float if the string parses as a number or fraction,
        else None.
        """
        normalized = answer.strip().lower()

        # Try plain float / int parse first.
        try:
            return normalized, float(normalized)
        except ValueError:
            pass

        # Try fraction p/q.
        if self.config.normalize_fractions:
            frac_match = _FRACTION_RE.match(normalized)
            if frac_match:
                num = int(frac_match.group(1))
                den = int(frac_match.group(2))
                if den != 0:
                    return normalized, num / den

        return normalized, None

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------

    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Return True if predicted matches ground_truth.

        Checks:
        1. Exact normalized string match.
        2. Numeric match within config.numeric_tolerance.
        """
        pred_norm, pred_num = self.normalize_answer(predicted)
        gt_norm, gt_num = self.normalize_answer(ground_truth)

        # Exact string match.
        if pred_norm == gt_norm:
            return True

        # Numeric match.
        if pred_num is not None and gt_num is not None:
            if abs(pred_num - gt_num) <= self.config.numeric_tolerance:
                return True

        return False

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: list[str],
        ground_truths: list[str],
        categories: list[str] | None = None,
    ) -> dict:
        """Evaluate a list of model responses against ground-truth answers.

        Args:
            predictions: Raw model response strings.
            ground_truths: Correct answer strings.
            categories: Optional per-problem category labels.

        Returns:
            {
                "accuracy": float,
                "n_correct": int,
                "n_total": int,
                "n_extraction_failed": int,
                "by_category": {cat: {"accuracy": float, "n": int}}
                               (only when categories is not None)
            }
        """
        n_total = len(predictions)
        n_correct = 0
        n_extraction_failed = 0

        # Per-category accumulators: cat → [correct_count, total_count]
        cat_acc: dict[str, list[int]] = {}

        for i, (pred_raw, gt) in enumerate(zip(predictions, ground_truths)):
            answer = self.extract_answer(pred_raw)

            # Extraction failed if we couldn't pull any text or the extracted
            # string does not parse numerically and has no boxed content.
            if answer.extracted is None:
                n_extraction_failed += 1
                correct = False
            else:
                correct = self.is_correct(answer.extracted, gt)

            if correct:
                n_correct += 1

            if categories is not None:
                cat = categories[i]
                if cat not in cat_acc:
                    cat_acc[cat] = [0, 0]
                cat_acc[cat][1] += 1
                if correct:
                    cat_acc[cat][0] += 1

        accuracy = n_correct / n_total if n_total > 0 else 0.0

        result: dict = {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "n_extraction_failed": n_extraction_failed,
        }

        if categories is not None:
            result["by_category"] = {
                cat: {
                    "accuracy": counts[0] / counts[1] if counts[1] > 0 else 0.0,
                    "n": counts[1],
                }
                for cat, counts in cat_acc.items()
            }

        return result

    # ------------------------------------------------------------------
    # AIME-specific scoring
    # ------------------------------------------------------------------

    def aime_score(
        self,
        predictions: list[str],
        ground_truths: list[str],
    ) -> dict:
        """AIME-specific evaluation.

        AIME answers are non-negative integers in [0, 999].
        Extracts integer from prediction and compares to ground-truth integer.

        Returns:
            {"score": int, "accuracy": float}
        """
        score = 0
        total = len(predictions)

        for pred_raw, gt in zip(predictions, ground_truths):
            answer = self.extract_answer(pred_raw)

            # Parse ground-truth as integer.
            gt_norm, gt_num = self.normalize_answer(gt)
            if gt_num is None:
                continue  # malformed ground truth; skip

            gt_int = round(gt_num)

            # Parse prediction as integer.
            if answer.extracted is not None:
                _, pred_num = self.normalize_answer(answer.extracted)
                if pred_num is not None:
                    pred_int = round(pred_num)
                    if pred_int == gt_int:
                        score += 1

        accuracy = score / total if total > 0 else 0.0
        return {"score": score, "accuracy": accuracy}


# Registry wiring is performed in src/eval/__init__.py via the additive
# registration pattern (BENCHMARK_REGISTRY.setdefault("math_eval", MathEval)).
# No import is needed here to avoid circular imports.
