"""Chain-of-thought reasoning quality scorer."""

from __future__ import annotations

import re
from dataclasses import dataclass


_STEP_SPLIT_RE = re.compile(
    r"(?:\n\s*|^\s*)(?:Step\s+\d+[:.)]|\d+[:.)])\s*",
    re.IGNORECASE,
)


@dataclass
class ChainOfThoughtScorer:
    """Scores chain-of-thought reasoning quality.

    Metrics:
    - step_count: number of reasoning steps
    - step_clarity: average clarity score per step (0-1)
    - logical_flow: whether steps follow logically (0-1)
    - conclusion_match: whether conclusion matches expected answer (0 or 1)
    """

    _max_length: int = 50_000

    def score(self, reasoning: str, expected_answer: str | None = None) -> dict[str, float]:
        """Score chain-of-thought reasoning.

        Returns:
            Dict with keys: step_count, step_clarity, logical_flow,
            conclusion_match, overall.

        Raises:
            ValueError: If reasoning is empty, exceeds max length, or wrong type.
        """
        if not isinstance(reasoning, str):
            raise ValueError(f"reasoning must be a string, got {type(reasoning).__name__}")
        if expected_answer is not None and not isinstance(expected_answer, str):
            raise ValueError(
                f"expected_answer must be a string or None, got {type(expected_answer).__name__}"
            )
        if len(reasoning) == 0:
            raise ValueError("reasoning must not be empty")
        if len(reasoning) > self._max_length:
            raise ValueError(
                f"reasoning length ({len(reasoning)}) exceeds maximum ({self._max_length})"
            )

        steps = self._extract_steps(reasoning)
        step_count = len(steps)

        if step_count == 0:
            step_clarity = 0.0
            logical_flow = 0.0
            conclusion_match = 0.0
        else:
            clarity_scores = [self._clarity(step) for step in steps]
            step_clarity = sum(clarity_scores) / step_count
            logical_flow = self._logical_flow(steps)

            if expected_answer is not None:
                conclusion = steps[-1].strip().lower()
                expected = expected_answer.strip().lower()
                conclusion_match = 1.0 if conclusion == expected else 0.0
            else:
                conclusion_match = 0.0

        if expected_answer is not None:
            overall = 0.2 * step_clarity + 0.3 * logical_flow + 0.5 * conclusion_match
        else:
            overall = 0.4 * step_clarity + 0.6 * logical_flow

        return {
            "step_count": step_count,
            "step_clarity": step_clarity,
            "logical_flow": logical_flow,
            "conclusion_match": conclusion_match,
            "overall": overall,
        }

    def _extract_steps(self, reasoning: str) -> list[str]:
        """Split reasoning by numbered markers like "Step 1:", "1.", etc."""
        parts = _STEP_SPLIT_RE.split(reasoning)
        # Filter out empty strings from leading/trailing splits
        steps = [part.strip() for part in parts if part.strip()]
        return steps

    def _clarity(self, step: str) -> float:
        """Simple heuristic: longer than 5 words = 1.0, else 0.5."""
        words = step.split()
        return 1.0 if len(words) > 5 else 0.5

    def _logical_flow(self, steps: list[str]) -> float:
        """Simple heuristic: if steps mention "therefore" / "because" / "so" = 1.0, else 0.7."""
        keywords = ("therefore", "because", "so")
        for step in steps:
            lower_step = step.lower()
            if any(kw in lower_step for kw in keywords):
                return 1.0
        return 0.7


DEFAULT_CHAIN_OF_THOUGHT_SCORER = ChainOfThoughtScorer()
CHAIN_OF_THOUGHT_SCORER_REGISTRY = {"default": DEFAULT_CHAIN_OF_THOUGHT_SCORER}
