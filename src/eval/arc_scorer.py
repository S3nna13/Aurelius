"""ARC scoring harness (Clark et al., 2018; arXiv:1803.05457).

AI2 Reasoning Challenge: 4-way multiple-choice science questions at Easy and
Challenge difficulty levels. Supports letter-extraction and text-matching
answer parsing. Pure stdlib -- re + dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ARCQuestion:
    question_id: str
    question: str
    choices: dict[str, str]
    correct_key: str
    difficulty: str = "Easy"


@dataclass(frozen=True)
class ARCResult:
    question_id: str
    predicted_key: str
    is_correct: bool


_LETTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(\s*([A-Da-d])\s*\)"),
    re.compile(r"\b([A-Da-d])\s*\)"),
    re.compile(r"^\s*([A-Da-d])\b", re.MULTILINE),
]


class ARCScorer:
    def __init__(self) -> None:
        pass

    def parse_answer(self, model_output: str, choices: dict[str, str]) -> str:
        for pat in _LETTER_PATTERNS:
            m = pat.search(model_output)
            if m:
                return m.group(1).upper()

        output_lower = model_output.lower()
        for key, text in choices.items():
            if text.lower() in output_lower:
                return key

        return "?"

    def score_answer(self, question: ARCQuestion, answer: str) -> ARCResult:
        predicted = self.parse_answer(answer, question.choices)
        return ARCResult(
            question_id=question.question_id,
            predicted_key=predicted,
            is_correct=(predicted == question.correct_key),
        )

    def score_batch(
        self,
        questions: list[ARCQuestion],
        answers: list[str],
    ) -> dict[str, object]:
        results = [self.score_answer(q, a) for q, a in zip(questions, answers)]
        n_total = len(results)
        n_correct = sum(1 for r in results if r.is_correct)
        accuracy = n_correct / n_total if n_total else 0.0

        easy_results = [r for q, r in zip(questions, results) if q.difficulty == "Easy"]
        challenge_results = [r for q, r in zip(questions, results) if q.difficulty == "Challenge"]

        easy_accuracy = (
            sum(1 for r in easy_results if r.is_correct) / len(easy_results)
            if easy_results
            else 0.0
        )
        challenge_accuracy = (
            sum(1 for r in challenge_results if r.is_correct) / len(challenge_results)
            if challenge_results
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "easy_accuracy": easy_accuracy,
            "challenge_accuracy": challenge_accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
        }

    def load_sample_questions(self) -> list[ARCQuestion]:
        return [
            ARCQuestion(
                question_id="arc_0",
                question="Which of the following is a source of energy for living things?",
                choices={"A": "Sunlight", "B": "Gravity", "C": "Magnetism", "D": "Sound"},
                correct_key="A",
                difficulty="Easy",
            ),
            ARCQuestion(
                question_id="arc_1",
                question="What is the primary function of the human heart?",
                choices={
                    "A": "Filter blood",
                    "B": "Pump blood",
                    "C": "Produce blood cells",
                    "D": "Digest nutrients",
                },
                correct_key="B",
                difficulty="Easy",
            ),
            ARCQuestion(
                question_id="arc_2",
                question="Which gas do plants absorb from the atmosphere during photosynthesis?",
                choices={"A": "Oxygen", "B": "Nitrogen", "C": "Carbon dioxide", "D": "Hydrogen"},
                correct_key="C",
                difficulty="Easy",
            ),
            ARCQuestion(
                question_id="arc_3",
                question="A student notices that a metal rod expands when heated. This best demonstrates which property of matter?",  # noqa: E501
                choices={
                    "A": "Thermal conductivity",
                    "B": "Thermal expansion",
                    "C": "Heat capacity",
                    "D": "Latent heat",
                },
                correct_key="B",
                difficulty="Challenge",
            ),
            ARCQuestion(
                question_id="arc_4",
                question="Which factor most directly affects the rate of a chemical reaction?",
                choices={
                    "A": "Color of the reactants",
                    "B": "Mass of the container",
                    "C": "Temperature of the reactants",
                    "D": "Shape of the reaction vessel",
                },
                correct_key="C",
                difficulty="Challenge",
            ),
        ]


ARC_REGISTRY: dict[str, type] = {"default": ARCScorer}

__all__ = [
    "ARCQuestion",
    "ARCResult",
    "ARCScorer",
    "ARC_REGISTRY",
]
