"""Inference-time compute scaling: best-of-N, majority vote, and self-refinement.

Scales inference compute by running multiple attempts and selecting or
refining the best output using configurable strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class InferenceScalingConfig:
    """Configuration for inference-time compute scaling."""

    n_samples: int = 8
    strategy: str = "best_of_n"  # "best_of_n" | "majority_vote" | "self_refine"
    n_refinement_steps: int = 3
    temperature: float = 1.0


class BestOfN:
    """Select the best candidate from N generations using a scorer function."""

    def __init__(self, scorer_fn: Callable[[list[str]], Tensor]) -> None:
        """
        Args:
            scorer_fn: Callable that accepts a list of N candidate strings and
                       returns a 1-D float tensor of shape (N,) with a score
                       for each candidate.
        """
        self.scorer_fn = scorer_fn

    def select(self, candidates: list[str]) -> tuple[str, int]:
        """Return the best candidate and its index (argmax of scores).

        Args:
            candidates: List of candidate strings.

        Returns:
            Tuple of (best_candidate_string, best_index).
        """
        scores: Tensor = self.scorer_fn(candidates)
        best_idx: int = int(torch.argmax(scores).item())
        return candidates[best_idx], best_idx

    def select_batch(self, candidates_per_prompt: list[list[str]]) -> list[str]:
        """Return the best candidate for each prompt in a batch.

        Args:
            candidates_per_prompt: List of candidate lists, one per prompt.

        Returns:
            List of best candidate strings, one per prompt.
        """
        return [self.select(candidates)[0] for candidates in candidates_per_prompt]


class MajorityVoter:
    """Select the most common answer across N completions."""

    def __init__(self, extractor_fn: Callable[[str], str]) -> None:
        """
        Args:
            extractor_fn: Callable that extracts a normalised answer string
                          from a raw completion string.
        """
        self.extractor_fn = extractor_fn

    def vote(self, candidates: list[str]) -> tuple[str, dict[str, int]]:
        """Return the majority answer and a dict of answer counts.

        Args:
            candidates: List of raw completion strings.

        Returns:
            Tuple of (majority_answer, {answer: count, ...}).
        """
        answers = [self.extractor_fn(c) for c in candidates]
        counts: dict[str, int] = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1
        majority_answer = max(counts, key=lambda a: counts[a])
        return majority_answer, counts

    def confidence(self, answer_counts: dict[str, int]) -> float:
        """Return max_count / total as a confidence in [0, 1].

        Args:
            answer_counts: Dict mapping answer strings to occurrence counts.

        Returns:
            Float confidence value.
        """
        total = sum(answer_counts.values())
        if total == 0:
            return 0.0
        max_count = max(answer_counts.values())
        return max_count / total


class SelfRefiner:
    """Iteratively refine a response by critiquing then regenerating."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        critique_fn: Callable[[str, str], str],
        n_steps: int = 3,
    ) -> None:
        """
        Args:
            generate_fn: Callable that accepts a prompt string and returns a
                         completion string.
            critique_fn: Callable that accepts (prompt, response) and returns a
                         critique/feedback string used as the next prompt.
            n_steps: Number of refinement iterations.
        """
        self.generate_fn = generate_fn
        self.critique_fn = critique_fn
        self.n_steps = n_steps

    def refine(self, prompt: str, initial_response: str) -> tuple[str, list[str]]:
        """Refine initial_response for n_steps iterations.

        Each step: critique current response → generate improved response.

        Args:
            prompt: Original prompt string.
            initial_response: Initial (unrefined) model response.

        Returns:
            Tuple of (final_response, all_responses) where all_responses
            contains the initial response plus one entry per refinement step
            (length == n_steps + 1).
        """
        history: list[str] = [initial_response]
        current_response = initial_response
        for _ in range(self.n_steps):
            critique = self.critique_fn(prompt, current_response)
            current_response = self.generate_fn(critique)
            history.append(current_response)
        return current_response, history

    def refine_batch(self, prompts: list[str], initial_responses: list[str]) -> list[str]:
        """Apply refine to each prompt/response pair.

        Args:
            prompts: List of prompt strings.
            initial_responses: Corresponding list of initial response strings.

        Returns:
            List of final refined response strings.
        """
        return [self.refine(p, r)[0] for p, r in zip(prompts, initial_responses)]


class ComputeScalingOrchestrator:
    """Dispatch inference-time scaling to the configured strategy."""

    def __init__(
        self,
        config: InferenceScalingConfig,
        best_of_n: BestOfN | None = None,
        voter: MajorityVoter | None = None,
        refiner: SelfRefiner | None = None,
    ) -> None:
        """
        Args:
            config: InferenceScalingConfig controlling which strategy to use.
            best_of_n: BestOfN instance (required for "best_of_n" strategy).
            voter: MajorityVoter instance (required for "majority_vote" strategy).
            refiner: SelfRefiner instance (required for "self_refine" strategy).
        """
        self.config = config
        self.best_of_n = best_of_n
        self.voter = voter
        self.refiner = refiner

    def run(self, prompt: str, candidates: list[str]) -> str:
        """Run the configured scaling strategy and return the selected response.

        Args:
            prompt: Original prompt (used by self_refine).
            candidates: List of candidate response strings.

        Returns:
            Selected or refined response string.

        Raises:
            ValueError: If strategy is unknown or required component is missing.
        """
        strategy = self.config.strategy

        if strategy == "best_of_n":
            if self.best_of_n is None:
                raise ValueError("best_of_n instance required for 'best_of_n' strategy")
            best, _ = self.best_of_n.select(candidates)
            return best

        elif strategy == "majority_vote":
            if self.voter is None:
                raise ValueError("voter instance required for 'majority_vote' strategy")
            majority, _ = self.voter.vote(candidates)
            return majority

        elif strategy == "self_refine":
            if self.refiner is None:
                raise ValueError("refiner instance required for 'self_refine' strategy")
            final, _ = self.refiner.refine(prompt, candidates[0])
            return final

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from: 'best_of_n', 'majority_vote', 'self_refine'."
            )
