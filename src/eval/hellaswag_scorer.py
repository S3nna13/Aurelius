"""HellaSwag scoring harness (Zellers et al., 2019; arXiv:1905.07830).

Commonsense NLI: pick the most plausible continuation from 4 candidates.
Supports string answer parsing (digit, letter, or text) and log-probability
scoring (argmax). Pure stdlib -- re + dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HellaSwagExample:
    example_id: str
    context: str
    activity_label: str
    endings: List[str]
    correct_idx: int


@dataclass(frozen=True)
class HellaSwagResult:
    example_id: str
    predicted_idx: int
    is_correct: bool


_DIGIT_RE = re.compile(r"\b([0-3])\b")
_LETTER_RE = re.compile(r"\(\s*([A-Da-d])\s*\)")
_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


class HellaSwagScorer:
    def __init__(self) -> None:
        pass

    def score_answer(self, example: HellaSwagExample, answer: str) -> HellaSwagResult:
        m = _DIGIT_RE.search(answer)
        if m:
            idx = int(m.group(1))
            return HellaSwagResult(
                example_id=example.example_id,
                predicted_idx=idx,
                is_correct=(idx == example.correct_idx),
            )

        m = _LETTER_RE.search(answer)
        if m:
            idx = _LETTER_TO_IDX[m.group(1).upper()]
            return HellaSwagResult(
                example_id=example.example_id,
                predicted_idx=idx,
                is_correct=(idx == example.correct_idx),
            )

        answer_lower = answer.lower()
        best_idx = -1
        best_len = -1
        for i, ending in enumerate(example.endings):
            ending_lower = ending.lower()
            if ending_lower in answer_lower and len(ending_lower) > best_len:
                best_len = len(ending_lower)
                best_idx = i

        if best_idx >= 0:
            return HellaSwagResult(
                example_id=example.example_id,
                predicted_idx=best_idx,
                is_correct=(best_idx == example.correct_idx),
            )

        return HellaSwagResult(
            example_id=example.example_id,
            predicted_idx=0,
            is_correct=(example.correct_idx == 0),
        )

    def score_logprobs(
        self, example: HellaSwagExample, logprobs: List[float]
    ) -> HellaSwagResult:
        predicted_idx = max(range(len(logprobs)), key=lambda i: logprobs[i])
        return HellaSwagResult(
            example_id=example.example_id,
            predicted_idx=predicted_idx,
            is_correct=(predicted_idx == example.correct_idx),
        )

    def score_batch(
        self,
        examples: List[HellaSwagExample],
        answers: List[str],
    ) -> Dict[str, object]:
        results = [self.score_answer(ex, a) for ex, a in zip(examples, answers)]
        n_total = len(results)
        n_correct = sum(1 for r in results if r.is_correct)
        accuracy = n_correct / n_total if n_total else 0.0
        return {"accuracy": accuracy, "n_correct": n_correct, "n_total": n_total}

    def load_sample_examples(self) -> List[HellaSwagExample]:
        return [
            HellaSwagExample(
                example_id="hswag_0",
                context="A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath.",
                activity_label="Bathing dog",
                endings=[
                    "She rinses the bucket out and then the dog runs away.",
                    "She uses a hose to rinse the dog clean.",
                    "She chases the dog with the bucket to give it a bath.",
                    "She throws the bucket at the dog.",
                ],
                correct_idx=2,
            ),
            HellaSwagExample(
                example_id="hswag_1",
                context="A man is slicing a loaf of bread on a cutting board.",
                activity_label="Cooking",
                endings=[
                    "He puts the slices in the freezer.",
                    "He arranges the slices on a plate.",
                    "He throws the bread in the trash.",
                    "He eats the entire loaf whole.",
                ],
                correct_idx=1,
            ),
            HellaSwagExample(
                example_id="hswag_2",
                context="Two people are playing chess. One player moves a knight.",
                activity_label="Playing chess",
                endings=[
                    "The other player flips the board in frustration.",
                    "The game immediately ends in a draw.",
                    "The other player considers their response carefully.",
                    "Both players leave the room.",
                ],
                correct_idx=2,
            ),
            HellaSwagExample(
                example_id="hswag_3",
                context="A child is learning to ride a bicycle with training wheels.",
                activity_label="Learning to ride a bike",
                endings=[
                    "The child falls off and gives up forever.",
                    "The child pedals slowly down the sidewalk.",
                    "The child immediately wins a race.",
                    "The child removes the training wheels immediately.",
                ],
                correct_idx=1,
            ),
            HellaSwagExample(
                example_id="hswag_4",
                context="A person is watering plants in a garden.",
                activity_label="Gardening",
                endings=[
                    "They accidentally flood the entire yard.",
                    "They move to each plant and give it water.",
                    "They stop and watch television instead.",
                    "They pour all the water on one plant.",
                ],
                correct_idx=1,
            ),
        ]


HELLASWAG_REGISTRY: Dict[str, type] = {"default": HellaSwagScorer}

__all__ = [
    "HellaSwagExample",
    "HellaSwagResult",
    "HellaSwagScorer",
    "HELLASWAG_REGISTRY",
]
