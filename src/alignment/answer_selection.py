"""Answer selection helpers for reranking multiple candidate responses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidateAnswer:
    text: str
    reward: float
    length_penalty: float = 0.0

    @property
    def adjusted_score(self) -> float:
        return self.reward - self.length_penalty


def select_best_answer(candidates: list[CandidateAnswer]) -> CandidateAnswer:
    """Pick the candidate with the highest adjusted score."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    return max(candidates, key=lambda candidate: candidate.adjusted_score)


def top_k_answers(candidates: list[CandidateAnswer], k: int) -> list[CandidateAnswer]:
    """Return the top-k candidates by adjusted score."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return sorted(candidates, key=lambda candidate: candidate.adjusted_score, reverse=True)[:k]


def answer_selection_probs(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax probabilities over candidate scores."""
    if scores.dim() != 1:
        raise ValueError("scores must be 1D")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return torch.softmax(scores / temperature, dim=0)


def reward_margin(best: CandidateAnswer, runner_up: CandidateAnswer) -> float:
    """Score margin between the best and second-best answers."""
    return best.adjusted_score - runner_up.adjusted_score
