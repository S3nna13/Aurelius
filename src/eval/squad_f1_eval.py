"""SQuAD F1/EM Evaluator — Rajpurkar et al. 2016.

Implements token-level F1 and Exact Match (EM) for extractive QA, following
the official SQuAD evaluation script normalization pipeline.

Registered in BENCHMARK_REGISTRY under the key "squad_f1".
Cycle 138-D.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class SQuADConfig:
    """Configuration for SQuADEval normalization and scoring."""

    strip_articles: bool = True
    strip_punctuation: bool = True
    lowercase: bool = True
    no_answer_token: str = ""  # for SQuAD 2.0 no-answer questions


@dataclass
class SQuADExample:
    """A single QA example with prediction and one or more gold answers."""

    question: str
    prediction: str
    gold_answers: List[str]


@dataclass
class SQuADResult:
    """Scored result for a single SQuAD example."""

    exact_match: float          # 0.0 or 1.0
    f1: float                   # token-level F1 in [0, 1]
    prediction_normalized: str
    best_gold_normalized: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


class SQuADEval:
    """SQuAD Exact Match / F1 evaluator (Rajpurkar et al. 2016).

    Parameters
    ----------
    config:
        Normalization flags.  Defaults are equivalent to the official script.
    """

    def __init__(self, config: SQuADConfig | None = None) -> None:
        self.config = config or SQuADConfig()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """Apply the standard SQuAD normalization pipeline.

        Steps (in order):
        1. Lowercase
        2. Remove articles (a / an / the)
        3. Remove punctuation
        4. Collapse whitespace
        """
        if self.config.lowercase:
            text = text.lower()
        if self.config.strip_articles:
            text = _ARTICLES_RE.sub(" ", text)
        if self.config.strip_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        # Collapse any run of whitespace to a single space and strip edges.
        text = " ".join(text.split())
        return text

    # ------------------------------------------------------------------
    # Scoring primitives
    # ------------------------------------------------------------------

    def token_f1(self, prediction: str, gold: str) -> float:
        """Token-level F1 between two *already-normalized* strings.

        Returns 0.0 if either string is empty (avoiding zero-division).
        """
        pred_tokens = prediction.split()
        gold_tokens = gold.split()

        if not pred_tokens or not gold_tokens:
            return 0.0

        pred_counts = Counter(pred_tokens)
        gold_counts = Counter(gold_tokens)

        # Element-wise min gives the common-token counts.
        common: Counter = pred_counts & gold_counts
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2.0 * precision * recall / (precision + recall)
        return f1

    def exact_match(self, prediction: str, gold: str) -> bool:
        """Exact string equality on two *already-normalized* strings."""
        return prediction == gold

    # ------------------------------------------------------------------
    # Per-example and batch scoring
    # ------------------------------------------------------------------

    def score_single(self, example: SQuADExample) -> SQuADResult:
        """Score one example against all gold answers, keep the best."""
        pred_norm = self.normalize(example.prediction)

        best_em: float = 0.0
        best_f1: float = 0.0
        best_gold_norm: str = ""

        for gold in example.gold_answers:
            gold_norm = self.normalize(gold)
            em = 1.0 if self.exact_match(pred_norm, gold_norm) else 0.0
            f1 = self.token_f1(pred_norm, gold_norm)
            if f1 > best_f1 or (f1 == best_f1 and em > best_em):
                best_f1 = f1
                best_em = em
                best_gold_norm = gold_norm
            # EM must be the max too (EM can be 1 while F1 is already 1.0)
            if em > best_em:
                best_em = em

        return SQuADResult(
            exact_match=best_em,
            f1=best_f1,
            prediction_normalized=pred_norm,
            best_gold_normalized=best_gold_norm,
        )

    def evaluate(self, examples: List[SQuADExample]) -> dict:
        """Aggregate EM and F1 over a list of examples.

        Returns
        -------
        dict with keys "exact_match", "f1", "n_examples".
        """
        if not examples:
            return {"exact_match": 0.0, "f1": 0.0, "n_examples": 0}

        results = [self.score_single(ex) for ex in examples]
        n = len(results)
        mean_em = sum(r.exact_match for r in results) / n
        mean_f1 = sum(r.f1 for r in results) / n
        return {"exact_match": mean_em, "f1": mean_f1, "n_examples": n}

    # ------------------------------------------------------------------
    # SQuAD 2.0 helpers
    # ------------------------------------------------------------------

    def no_answer_accuracy(self, examples: List[SQuADExample]) -> float:
        """Fraction of examples where prediction and all golds are empty.

        Designed for SQuAD 2.0 no-answer questions.  An example counts as a
        correct no-answer prediction when:
          - prediction normalizes to config.no_answer_token (default "")
          - every gold answer also normalizes to config.no_answer_token
        """
        if not examples:
            return 0.0

        no_ans = self.config.no_answer_token
        correct = 0
        for ex in examples:
            pred_is_empty = self.normalize(ex.prediction) == no_ans
            golds_all_empty = all(
                self.normalize(g) == no_ans for g in ex.gold_answers
            )
            if pred_is_empty and golds_all_empty:
                correct += 1

        return correct / len(examples)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def per_length_analysis(
        self, examples: List[SQuADExample]
    ) -> dict:
        """Bin examples by best gold answer word count and report EM/F1.

        Bins:
          "short"  — gold < 3 words
          "medium" — gold in [3, 7] words
          "long"   — gold > 7 words

        Returns
        -------
        dict: {"short": {"em": float, "f1": float}, "medium": ..., "long": ...}
        """
        bins: dict[str, list[SQuADResult]] = {
            "short": [],
            "medium": [],
            "long": [],
        }

        for ex in examples:
            result = self.score_single(ex)
            # Use the best gold answer length as the binning criterion.
            gold_len = len(result.best_gold_normalized.split())
            if gold_len < 3:
                bins["short"].append(result)
            elif gold_len <= 7:
                bins["medium"].append(result)
            else:
                bins["long"].append(result)

        analysis: dict[str, dict[str, float]] = {}
        for bin_name, bin_results in bins.items():
            if bin_results:
                n = len(bin_results)
                analysis[bin_name] = {
                    "em": sum(r.exact_match for r in bin_results) / n,
                    "f1": sum(r.f1 for r in bin_results) / n,
                }
            else:
                analysis[bin_name] = {"em": 0.0, "f1": 0.0}

        return analysis


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.eval import BENCHMARK_REGISTRY  # noqa: E402

BENCHMARK_REGISTRY["squad_f1"] = SQuADEval

__all__ = [
    "SQuADConfig",
    "SQuADExample",
    "SQuADResult",
    "SQuADEval",
]
