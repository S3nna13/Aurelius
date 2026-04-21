"""Tool Bench Evaluator — agent tool-use quality evaluation.

Measures:
  - tool_selection_accuracy  : correct tool selected (ignoring params)
  - parameter_accuracy       : parameters correctly extracted per matched call
  - sequence_exact_match     : full sequence (name + params) exact match
  - format_compliance        : raw response contains valid JSON tool-call syntax

Cycle 132-D.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str
    parameters: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolBenchSample:
    sample_id: str
    task_description: str
    available_tools: list[str]
    expected_tool_calls: list[ToolCall]
    predicted_tool_calls: list[ToolCall]


@dataclass
class ToolBenchConfig:
    param_match_threshold: float = 0.8   # fraction of params that must match
    order_sensitive: bool = True          # does tool call order matter?
    partial_credit: bool = True           # partial credit for partial param match


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class ToolBench:
    """Evaluate agent tool-use quality across a list of ToolBenchSamples."""

    def __init__(self, config: Optional[ToolBenchConfig] = None) -> None:
        self.config = config or ToolBenchConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _params_match_fraction(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
        """Return fraction of expected parameters correctly predicted.

        Comparison is done after str() normalisation so numeric types round-trip
        cleanly against string representations.
        """
        if not expected:
            # No expected params — trivially a perfect match if predicted also
            # has no extra params (or we simply credit it fully).
            return 1.0
        matches = sum(
            1
            for k, v in expected.items()
            if k in predicted and str(predicted[k]) == str(v)
        )
        return matches / len(expected)

    @staticmethod
    def _calls_match_exactly(a: ToolCall, b: ToolCall) -> bool:
        """True iff both tool names and all parameters match exactly."""
        if a.tool_name != b.tool_name:
            return False
        if set(a.parameters.keys()) != set(b.parameters.keys()):
            return False
        for k, v in a.parameters.items():
            if k not in b.parameters or str(b.parameters[k]) != str(v):
                return False
        return True

    # ------------------------------------------------------------------
    # Public metrics
    # ------------------------------------------------------------------

    def tool_selection_accuracy(self, samples: list[ToolBenchSample]) -> float:
        """Fraction of expected tools correctly selected (ignoring params).

        For each sample we compute |intersection of names| / |expected names|,
        respecting order when config.order_sensitive=True.
        """
        if not samples:
            return 0.0

        total_score = 0.0
        for sample in samples:
            expected_names = [tc.tool_name for tc in sample.expected_tool_calls]
            predicted_names = [tc.tool_name for tc in sample.predicted_tool_calls]

            if not expected_names:
                total_score += 1.0
                continue

            if self.config.order_sensitive:
                # Position-by-position comparison up to len(expected)
                n = len(expected_names)
                pred_slice = predicted_names[:n]
                matches = sum(
                    1 for e, p in zip(expected_names, pred_slice) if e == p
                )
                total_score += matches / n
            else:
                # Set-based: how many expected names appear in predicted
                pred_set = set(predicted_names)
                matches = sum(1 for name in expected_names if name in pred_set)
                total_score += matches / len(expected_names)

        return total_score / len(samples)

    def parameter_accuracy(self, samples: list[ToolBenchSample]) -> float:
        """Average parameter-match fraction across all matched tool calls.

        A "matched" call is one where the tool name aligns (position-by-position
        when order_sensitive, or by first-name match otherwise).
        """
        if not samples:
            return 0.0

        total_param_score = 0.0
        total_matched_calls = 0

        for sample in samples:
            expected = sample.expected_tool_calls
            predicted = sample.predicted_tool_calls

            if self.config.order_sensitive:
                pairs = list(zip(expected, predicted))
                pairs = [(e, p) for e, p in pairs if e.tool_name == p.tool_name]
            else:
                pred_by_name: dict[str, ToolCall] = {}
                for pc in predicted:
                    pred_by_name.setdefault(pc.tool_name, pc)
                pairs = [
                    (ec, pred_by_name[ec.tool_name])
                    for ec in expected
                    if ec.tool_name in pred_by_name
                ]

            for ec, pc in pairs:
                frac = self._params_match_fraction(ec.parameters, pc.parameters)
                if self.config.partial_credit:
                    total_param_score += frac
                else:
                    total_param_score += 1.0 if frac >= self.config.param_match_threshold else 0.0
                total_matched_calls += 1

        if total_matched_calls == 0:
            return 0.0
        return total_param_score / total_matched_calls

    def sequence_exact_match(self, samples: list[ToolBenchSample]) -> float:
        """Fraction of samples whose entire tool call sequence matches exactly.

        Both tool names and all parameters must match for every call in order.
        """
        if not samples:
            return 0.0

        n_exact = 0
        for sample in samples:
            expected = sample.expected_tool_calls
            predicted = sample.predicted_tool_calls

            if len(expected) != len(predicted):
                continue

            if all(self._calls_match_exactly(e, p) for e, p in zip(expected, predicted)):
                n_exact += 1

        return n_exact / len(samples)

    # ------------------------------------------------------------------
    # Format compliance
    # ------------------------------------------------------------------

    # Minimal pattern: {"tool": "...", "parameters": {...}}
    _FORMAT_RE = re.compile(
        r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{',
        re.DOTALL,
    )

    def format_compliance(self, raw_responses: list[str]) -> float:
        """Fraction of raw responses containing a valid tool-call JSON fragment.

        A valid fragment must match: {"tool": "...", "parameters": {...}}
        """
        if not raw_responses:
            return 0.0

        n_valid = sum(
            1 for resp in raw_responses if self._FORMAT_RE.search(resp) is not None
        )
        return n_valid / len(raw_responses)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        samples: list[ToolBenchSample],
        raw_responses: Optional[list[str]] = None,
    ) -> dict:
        """Run all metrics and return a results dict.

        Keys: tool_selection, parameter_accuracy, sequence_exact_match,
              n_samples, and optionally format_compliance.
        """
        result: dict = {
            "tool_selection": self.tool_selection_accuracy(samples),
            "parameter_accuracy": self.parameter_accuracy(samples),
            "sequence_exact_match": self.sequence_exact_match(samples),
            "n_samples": len(samples),
        }
        if raw_responses is not None:
            result["format_compliance"] = self.format_compliance(raw_responses)
        return result
