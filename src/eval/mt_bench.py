"""MT-Bench style evaluation harness for Aurelius LLM project.

Pure PyTorch only — no HuggingFace, scipy, or sklearn dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

class MTBenchCategory:
    WRITING = "writing"
    ROLEPLAY = "roleplay"
    REASONING = "reasoning"
    MATH = "math"
    CODING = "coding"
    EXTRACTION = "extraction"
    STEM = "stem"
    HUMANITIES = "humanities"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MTBenchQuestion:
    question_id: int
    category: str
    turns: List[str]
    reference_answer: Optional[str] = None


@dataclass
class MTBenchResult:
    question_id: int
    category: str
    scores: List[float]
    judge_outputs: List[str]
    mean_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_score = sum(self.scores) / len(self.scores) if self.scores else 0.0


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

_SCORE_PATTERNS = [
    re.compile(r"\bscore\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\brating\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", re.IGNORECASE),
    re.compile(r"(?:^|[\s.,;:!?])\b([0-9]+(?:\.[0-9]+)?)\b\s*$"),
]


def extract_score_from_text(text: str) -> Optional[float]:
    """Parse a numeric score (1-10) from judge output text.

    Tries the following patterns in order:
      - "Score: X" / "Score = X"
      - "Rating: X" / "Rating = X"
      - "X/10"
      - standalone digit at end of text

    Returns a float in [1, 10] if found, else None.
    """
    for pattern in _SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                value = float(match.group(1))
                if 1.0 <= value <= 10.0:
                    return value
            except (ValueError, IndexError):
                continue
    return None


# ---------------------------------------------------------------------------
# Judge prompt builder
# ---------------------------------------------------------------------------

def build_judge_prompt(question: str, response: str, category: str = "general") -> str:
    """Build a structured prompt asking the judge to rate the response 1-10.

    The prompt instructs the judge to output "Score: X" at the end.
    """
    prompt = (
        f"You are an expert evaluator for {category} tasks.\n\n"
        f"Question:\n{question}\n\n"
        f"Response:\n{response}\n\n"
        "Please evaluate the response above on a scale from 1 to 10, where:\n"
        "  1 = very poor / completely wrong\n"
        " 10 = excellent / perfectly correct\n\n"
        "Consider accuracy, clarity, completeness, and appropriateness for the category.\n"
        "At the end of your evaluation, output exactly: Score: X\n"
        "(where X is an integer from 1 to 10)"
    )
    return prompt


# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------

class JudgeModel:
    """Wraps a generate function and provides scoring utilities."""

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        self._generate = generate_fn

    def score_response(
        self,
        question: str,
        response: str,
        category: str = "general",
    ) -> float:
        """Score a single (question, response) pair.

        Builds a judge prompt, calls generate_fn, parses the score.
        Returns 5.0 as a default if no score is parseable.
        """
        prompt = build_judge_prompt(question, response, category)
        judge_text = self._generate(prompt)
        score = extract_score_from_text(judge_text)
        return score if score is not None else 5.0

    def score_multi_turn(
        self,
        questions: List[str],
        responses: List[str],
        category: str,
    ) -> List[float]:
        """Score each (question, response) pair independently."""
        return [
            self.score_response(q, r, category)
            for q, r in zip(questions, responses)
        ]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MTBenchEvaluator:
    """End-to-end MT-Bench evaluator."""

    def __init__(
        self,
        judge: JudgeModel,
        generate_fn: Callable[[str], str],
    ) -> None:
        self.judge = judge
        self._generate = generate_fn

    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate a model response for the given prompt."""
        return self._generate(prompt)

    def evaluate_question(self, question: MTBenchQuestion) -> MTBenchResult:
        """For each turn, generate a response then score it."""
        scores: List[float] = []
        judge_outputs: List[str] = []

        for turn_prompt in question.turns:
            response = self.generate_response(turn_prompt)
            judge_prompt = build_judge_prompt(turn_prompt, response, question.category)
            judge_text = self._generate(judge_prompt)
            score = extract_score_from_text(judge_text)
            scores.append(score if score is not None else 5.0)
            judge_outputs.append(judge_text)

        return MTBenchResult(
            question_id=question.question_id,
            category=question.category,
            scores=scores,
            judge_outputs=judge_outputs,
        )

    def evaluate_all(self, questions: List[MTBenchQuestion]) -> List[MTBenchResult]:
        """Evaluate every question and return a list of results."""
        return [self.evaluate_question(q) for q in questions]

    def compute_summary(self, results: List[MTBenchResult]) -> Dict:
        """Compute overall and per-category mean scores.

        Returns:
            {
                'overall_score': float,
                'per_category': Dict[str, float],
                'n_questions': int,
            }
        """
        if not results:
            return {"overall_score": 0.0, "per_category": {}, "n_questions": 0}

        per_category: Dict[str, List[float]] = {}
        for result in results:
            per_category.setdefault(result.category, []).append(result.mean_score)

        per_category_means: Dict[str, float] = {
            cat: sum(scores) / len(scores)
            for cat, scores in per_category.items()
        }

        overall = sum(r.mean_score for r in results) / len(results)

        return {
            "overall_score": overall,
            "per_category": per_category_means,
            "n_questions": len(results),
        }


# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------

def get_sample_questions() -> List[MTBenchQuestion]:
    """Return 5 sample MT-Bench style questions covering different categories."""
    return [
        MTBenchQuestion(
            question_id=1,
            category=MTBenchCategory.WRITING,
            turns=[
                "Write a short poem about the nature of artificial intelligence.",
                "Now rewrite it in the style of Shakespeare.",
            ],
        ),
        MTBenchQuestion(
            question_id=2,
            category=MTBenchCategory.MATH,
            turns=[
                "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?",
            ],
            reference_answer="f'(x) = 3x^2 + 4x - 5",
        ),
        MTBenchQuestion(
            question_id=3,
            category=MTBenchCategory.CODING,
            turns=[
                "Write a Python function that checks if a string is a palindrome.",
            ],
        ),
        MTBenchQuestion(
            question_id=4,
            category=MTBenchCategory.REASONING,
            turns=[
                "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            ],
        ),
        MTBenchQuestion(
            question_id=5,
            category=MTBenchCategory.STEM,
            turns=[
                "Explain how a transformer neural network processes a sequence of tokens.",
                "How does attention differ from convolution in this context?",
            ],
        ),
    ]
