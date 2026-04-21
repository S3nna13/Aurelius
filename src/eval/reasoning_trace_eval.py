"""
reasoning_trace_eval.py — Reasoning Trace Evaluator (Cycle 135-F)

Evaluates chain-of-thought traces on multiple dimensions without needing
a ground-truth verifier. Pure Python + PyTorch only (no external ML libs).

Metrics
-------
- faithfulness_score   : fraction of non-answer steps that contain explicit
                         reasoning signal words ("because", "therefore", …)
- length_efficiency    : penalises traces that are far longer than the answer
- redundancy_score     : 1 – fraction of repeated word n-grams
- step_consistency     : mean pairwise Jaccard word-overlap between consecutive
                         steps (measures coherent flow)
- overall              : weighted average of the four metrics above

Registry
--------
  BENCHMARK_REGISTRY["reasoning_trace_eval"] = ReasoningTraceEval
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict

import torch  # imported for project consistency; tensor ops available if needed

from src.eval import BENCHMARK_REGISTRY

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTraceConfig:
    """Hyper-parameters for the reasoning-trace evaluator."""
    step_delimiter: str = "\n"
    answer_prefix: str = "Therefore"
    min_steps: int = 2
    max_length_ratio: float = 5.0
    redundancy_ngram: int = 4


# ---------------------------------------------------------------------------
# Per-step analysis dataclass
# ---------------------------------------------------------------------------

_REASONING_WORDS = frozenset(
    {"because", "therefore", "since", "thus", "so"}
)

_MATH_PATTERN = re.compile(r"[\d=+\-*/]")


@dataclass
class StepAnalysis:
    """Structured analysis of a single reasoning step."""
    text: str
    word_count: int
    is_answer_step: bool
    has_reasoning_words: bool
    has_math_notation: bool


# ---------------------------------------------------------------------------
# Evaluation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TraceEvalResult:
    """Evaluation result for a single (trace, answer) pair."""
    n_steps: int
    faithfulness_score: float
    length_efficiency: float
    redundancy_score: float
    has_final_answer: bool
    step_consistency: float
    overall: float


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class ReasoningTraceEval:
    """
    Evaluates chain-of-thought reasoning traces across multiple quality
    dimensions.

    Parameters
    ----------
    config : ReasoningTraceConfig
        Configuration controlling delimiters, prefixes, and thresholds.
    """

    # Weights for overall score (must sum to 1.0)
    _WEIGHTS: Dict[str, float] = field(default_factory=dict)

    def __init__(self, config: ReasoningTraceConfig | None = None) -> None:
        self.config = config or ReasoningTraceConfig()
        self._weights = {
            "faithfulness": 0.3,
            "efficiency":   0.2,
            "redundancy":   0.2,
            "consistency":  0.3,
        }

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_steps(self, trace: str) -> List[StepAnalysis]:
        """
        Split *trace* by ``config.step_delimiter`` and return a
        :class:`StepAnalysis` for every non-empty step.
        """
        raw_steps = trace.split(self.config.step_delimiter)
        analyses: List[StepAnalysis] = []
        for raw in raw_steps:
            text = raw.strip()
            if not text:
                continue
            words = text.lower().split()
            word_count = len(words)
            is_answer = text.startswith(self.config.answer_prefix)
            has_rw = bool(_REASONING_WORDS.intersection(words))
            has_math = bool(_MATH_PATTERN.search(text))
            analyses.append(
                StepAnalysis(
                    text=text,
                    word_count=word_count,
                    is_answer_step=is_answer,
                    has_reasoning_words=has_rw,
                    has_math_notation=has_math,
                )
            )
        return analyses

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def faithfulness_score(self, steps: List[StepAnalysis]) -> float:
        """
        Fraction of *non-answer* steps that contain at least one explicit
        reasoning signal word.

        Returns 0.0 when there are no non-answer steps.
        """
        non_answer = [s for s in steps if not s.is_answer_step]
        if not non_answer:
            return 0.0
        n_faithful = sum(1 for s in non_answer if s.has_reasoning_words)
        return n_faithful / len(non_answer)

    def length_efficiency(self, trace: str, answer: str) -> float:
        """
        Penalises traces that are much longer than the final answer.

        efficiency = min(1, max_length_ratio / actual_ratio)

        where actual_ratio = trace_len / max(answer_len, 1).
        """
        trace_len = len(trace)
        answer_len = max(len(answer), 1)
        actual_ratio = trace_len / answer_len
        if actual_ratio <= 0:
            return 1.0
        efficiency = self.config.max_length_ratio / actual_ratio
        return min(1.0, efficiency)

    def redundancy_score(self, trace: str) -> float:
        """
        Measures lexical diversity via word n-gram uniqueness.

        score = unique_ngrams / total_ngrams

        A score of 1.0 means every n-gram is distinct; lower scores
        indicate more repetition.  Returns 1.0 for traces too short to
        form any n-gram.
        """
        n = self.config.redundancy_ngram
        words = trace.lower().split()
        if len(words) < n:
            return 1.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        return unique / total

    def step_consistency(self, steps: List[StepAnalysis]) -> float:
        """
        Mean Jaccard word-overlap between consecutive step pairs.

        Higher scores indicate that each step builds on vocabulary
        established in the previous step, suggesting coherent flow.
        Returns 0.0 when there are fewer than 2 steps.
        """
        if len(steps) < 2:
            return 0.0
        scores: List[float] = []
        for a, b in zip(steps[:-1], steps[1:]):
            set_a = set(a.text.lower().split())
            set_b = set(b.text.lower().split())
            union = set_a | set_b
            if not union:
                scores.append(0.0)
            else:
                intersection = set_a & set_b
                scores.append(len(intersection) / len(union))
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Top-level evaluation
    # ------------------------------------------------------------------

    def evaluate(self, trace: str, answer: str) -> TraceEvalResult:
        """
        Evaluate a single (trace, answer) pair and return a
        :class:`TraceEvalResult`.
        """
        steps = self.parse_steps(trace)
        n_steps = len(steps)

        faith = self.faithfulness_score(steps)
        efficiency = self.length_efficiency(trace, answer)
        redundancy = self.redundancy_score(trace)
        consistency = self.step_consistency(steps)
        has_final = any(s.is_answer_step for s in steps)

        overall = (
            self._weights["faithfulness"] * faith
            + self._weights["efficiency"] * efficiency
            + self._weights["redundancy"] * redundancy
            + self._weights["consistency"] * consistency
        )
        # Clamp to [0, 1] for safety
        overall = max(0.0, min(1.0, overall))

        return TraceEvalResult(
            n_steps=n_steps,
            faithfulness_score=faith,
            length_efficiency=efficiency,
            redundancy_score=redundancy,
            has_final_answer=has_final,
            step_consistency=consistency,
            overall=overall,
        )

    def evaluate_batch(
        self, traces: List[str], answers: List[str]
    ) -> List[TraceEvalResult]:
        """
        Evaluate a batch of (trace, answer) pairs.

        Parameters
        ----------
        traces  : parallel list of reasoning trace strings
        answers : parallel list of final-answer strings

        Returns
        -------
        list of :class:`TraceEvalResult` in the same order.
        """
        if len(traces) != len(answers):
            raise ValueError(
                f"traces and answers must have equal length, "
                f"got {len(traces)} vs {len(answers)}"
            )
        return [self.evaluate(t, a) for t, a in zip(traces, answers)]

    def aggregate(self, results: List[TraceEvalResult]) -> Dict[str, float]:
        """
        Compute the mean of each metric across *results*.

        Returns
        -------
        dict with keys: n_steps_mean, faithfulness_mean, efficiency_mean,
                        redundancy_mean, consistency_mean, overall_mean
        """
        if not results:
            return {
                "n_steps_mean": 0.0,
                "faithfulness_mean": 0.0,
                "efficiency_mean": 0.0,
                "redundancy_mean": 0.0,
                "consistency_mean": 0.0,
                "overall_mean": 0.0,
            }
        n = len(results)
        return {
            "n_steps_mean": sum(r.n_steps for r in results) / n,
            "faithfulness_mean": sum(r.faithfulness_score for r in results) / n,
            "efficiency_mean": sum(r.length_efficiency for r in results) / n,
            "redundancy_mean": sum(r.redundancy_score for r in results) / n,
            "consistency_mean": sum(r.step_consistency for r in results) / n,
            "overall_mean": sum(r.overall for r in results) / n,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY["reasoning_trace_eval"] = ReasoningTraceEval
