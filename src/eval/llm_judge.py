"""LLM-as-judge evaluation framework for AureliusTransformer.

Uses a language model to score or compare responses on specified criteria.
Parses numeric scores from generated text — no external dependencies.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class JudgingCriteria:
    """A single evaluation criterion for LLM-as-judge."""

    name: str
    description: str
    min_score: int = field(init=False)
    max_score: int = field(init=False)
    _scale: tuple[int, int] = field(default=(1, 10), repr=False)

    def __init__(self, name: str, description: str, scale: tuple[int, int] = (1, 10)):
        self.name = name
        self.description = description
        self.min_score, self.max_score = scale


STANDARD_CRITERIA = [
    JudgingCriteria("helpfulness", "How helpful is the response?"),
    JudgingCriteria("accuracy", "How factually accurate is the response?"),
    JudgingCriteria("coherence", "How coherent and well-structured is the response?"),
    JudgingCriteria("harmlessness", "Is the response free from harmful content?"),
]


class LLMJudge:
    """
    Use a language model to judge response quality.

    Builds structured prompts asking the model to score responses
    on specified criteria, then parses the scores from generated text.

    Args:
        model: the judge model (must have a callable generate or forward interface)
        tokenizer_encode: callable(text: str) -> list[int]
        tokenizer_decode: callable(ids: list[int]) -> str
        criteria: list of JudgingCriteria
        max_score_tokens: max tokens to generate for scoring
    """

    def __init__(
        self,
        model,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        criteria: list[JudgingCriteria] | None = None,
        max_score_tokens: int = 32,
    ):
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.criteria = criteria if criteria is not None else STANDARD_CRITERIA[:2]
        self.max_score_tokens = max_score_tokens

    def build_prompt(self, question: str, response: str, criterion: JudgingCriteria) -> str:
        """Build judging prompt for a single criterion."""
        return (
            f"Question: {question}\n"
            f"Response: {response}\n\n"
            f"Rate the response on {criterion.name} ({criterion.description})\n"
            f"on a scale of {criterion.min_score}-{criterion.max_score}. "
            f"Reply with just the number."
        )

    def _generate_text(self, prompt: str) -> str:
        """Encode prompt, run model, decode output tokens."""
        input_ids = self.tokenizer_encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            # Try generate() first, fall back to a single forward pass
            if hasattr(self.model, "generate"):
                output_ids = self.model.generate(input_tensor, max_new_tokens=self.max_score_tokens)
                # output_ids may include the prompt; take only new tokens
                if output_ids.shape[-1] > len(input_ids):
                    new_ids = output_ids[0, len(input_ids) :].tolist()
                else:
                    new_ids = output_ids[0].tolist()
            else:
                logits = self.model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # Greedy decode max_score_tokens steps
                new_ids = []
                current = input_tensor
                for _ in range(self.max_score_tokens):
                    out = self.model(current)
                    if isinstance(out, tuple):
                        out = out[0]
                    next_id = out[0, -1].argmax(-1).unsqueeze(0).unsqueeze(0)
                    new_ids.append(next_id.item())
                    current = torch.cat([current, next_id], dim=-1)

        return self.tokenizer_decode(new_ids)

    def parse_score(self, generated_text: str, criterion: JudgingCriteria) -> float | None:
        """
        Extract numeric score from generated text.

        Looks for the first integer or float, clamps to [min_score, max_score].
        Returns None if no valid number is found.
        """
        matches = re.findall(r"\b\d+(?:\.\d+)?\b", generated_text)
        if not matches:
            return None
        value = float(matches[0])
        return float(max(criterion.min_score, min(criterion.max_score, value)))

    def judge_single(self, question: str, response: str) -> dict:
        """
        Score a single response on all criteria.

        Returns: {criterion_name: score, ..., 'mean_score': float}
        """
        scores: dict[str, float] = {}
        for criterion in self.criteria:
            prompt = self.build_prompt(question, response, criterion)
            generated = self._generate_text(prompt)
            score = self.parse_score(generated, criterion)
            if score is None:
                score = float(criterion.min_score)
            scores[criterion.name] = score

        if scores:
            scores["mean_score"] = sum(v for k, v in scores.items()) / len(scores)
        else:
            scores["mean_score"] = 0.0
        return scores

    def compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str,
    ) -> dict:
        """
        Compare two responses. Score each, return:
        {
            'scores_a': {crit: score},
            'scores_b': {crit: score},
            'winner': 'A' | 'B' | 'tie',
            'margin': float,
        }
        """
        scores_a = self.judge_single(question, response_a)
        scores_b = self.judge_single(question, response_b)

        mean_a = scores_a["mean_score"]
        mean_b = scores_b["mean_score"]
        margin = abs(mean_a - mean_b)

        if mean_a > mean_b:
            winner = "A"
        elif mean_b > mean_a:
            winner = "B"
        else:
            winner = "tie"

        return {
            "scores_a": scores_a,
            "scores_b": scores_b,
            "winner": winner,
            "margin": margin,
        }

    def batch_judge(self, examples: list[dict]) -> list[dict]:
        """
        Judge multiple (question, response) pairs.

        examples: list of {'question': str, 'response': str}
        Returns: list of score dicts
        """
        return [self.judge_single(ex["question"], ex["response"]) for ex in examples]


class PointwiseJudge(LLMJudge):
    """Judge that scores responses independently (not pairwise)."""

    def calibrate_scale(self, sample_responses: list[str], question: str) -> float:
        """
        Run judge on sample responses to estimate scale calibration.

        Returns mean score across samples as a reference point.
        """
        if not sample_responses:
            return 0.0
        total = 0.0
        for response in sample_responses:
            result = self.judge_single(question, response)
            total += result["mean_score"]
        return total / len(sample_responses)


class PairwiseJudge(LLMJudge):
    """Judge that compares response pairs directly."""

    def tournament(self, question: str, responses: list[str]) -> list[tuple[str, float]]:
        """
        Round-robin tournament: compare all pairs.

        Returns: list of (response, win_rate) sorted by win_rate descending.
        """
        n = len(responses)
        if n == 0:
            return []
        if n == 1:
            return [(responses[0], 1.0)]

        wins = [0] * n
        total_matches = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                result = self.compare_responses(question, responses[i], responses[j])
                total_matches[i] += 1
                total_matches[j] += 1
                if result["winner"] == "A":
                    wins[i] += 1
                elif result["winner"] == "B":
                    wins[j] += 1
                else:
                    # Tie: award 0.5 to each
                    wins[i] += 0  # no credit for a tie in win_rate
                    wins[j] += 0

        ranked = []
        for i, response in enumerate(responses):
            matches = total_matches[i]
            win_rate = wins[i] / matches if matches > 0 else 0.0
            ranked.append((response, win_rate))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
