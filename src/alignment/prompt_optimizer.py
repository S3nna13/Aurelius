"""Black-box prompt optimization via evolutionary search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PromptOptimizerConfig:
    prompt_len: int = 8  # number of prefix tokens to optimize
    vocab_size: int = 256
    population_size: int = 10  # number of candidate prompts
    n_elite: int = 3  # top-k survivors per generation
    mutation_rate: float = 0.1  # fraction of tokens to randomly mutate
    n_generations: int = 5
    temperature: float = 1.0  # for scoring


def score_prompt(
    model: nn.Module,
    prompt_ids: Tensor,  # (prompt_len,) prefix tokens
    target_ids: Tensor,  # (target_len,) target sequence to elicit
) -> float:
    """Score a prompt by how well the model predicts target_ids after prompt_ids.

    Concretely: compute -cross_entropy(model([prompt_ids | target_ids[:-1]]), target_ids)
    Higher score = model assigns higher probability to the target.
    Returns a float.
    """
    with torch.no_grad():
        # Build input: [prompt_ids | target_ids[:-1]], shape (1, prompt_len + target_len - 1)
        # Labels: target_ids, shape (target_len,)
        # We predict each target token from the preceding context.
        input_seq = torch.cat([prompt_ids, target_ids[:-1]], dim=0).unsqueeze(0)  # (1, L)
        _, logits, _ = model(input_seq)
        # logits: (1, L, vocab_size)
        # We want to evaluate logits at the target_ids positions.
        # Positions prompt_len .. prompt_len + target_len - 1 predict target_ids[0..target_len-1]
        prompt_len = prompt_ids.shape[0]
        target_len = target_ids.shape[0]
        # Slice out the logits corresponding to target token predictions
        target_logits = logits[
            0, prompt_len - 1 : prompt_len - 1 + target_len, :
        ]  # (target_len, V)
        labels = target_ids  # (target_len,)
        loss = F.cross_entropy(target_logits, labels)
        return -loss.item()


def initialize_population(
    population_size: int,
    prompt_len: int,
    vocab_size: int,
) -> Tensor:
    """Randomly initialize population of prompt candidates.
    Returns (population_size, prompt_len) int64 tensor."""
    return torch.randint(0, vocab_size, (population_size, prompt_len), dtype=torch.int64)


def select_elite(
    population: Tensor,  # (pop_size, prompt_len)
    scores: Tensor,  # (pop_size,)
    n_elite: int,
) -> tuple[Tensor, Tensor]:
    """Select top-n_elite candidates by score.
    Returns (elite_population (n_elite, prompt_len), elite_scores (n_elite,))."""
    # Sort descending by score
    sorted_indices = torch.argsort(scores, descending=True)
    elite_indices = sorted_indices[:n_elite]
    elite_pop = population[elite_indices]
    elite_scores = scores[elite_indices]
    return elite_pop, elite_scores


def mutate_population(
    elite: Tensor,  # (n_elite, prompt_len)
    population_size: int,
    vocab_size: int,
    mutation_rate: float,
) -> Tensor:
    """Expand elite to full population by cloning + random token mutations.
    Returns (population_size, prompt_len) int64."""
    n_elite, prompt_len = elite.shape
    # Tile elite to fill population_size
    repeats = (population_size + n_elite - 1) // n_elite
    tiled = elite.repeat(repeats, 1)[:population_size]  # (population_size, prompt_len)

    # Generate mutation mask: True where we mutate
    mask = torch.rand(population_size, prompt_len) < mutation_rate  # (pop_size, prompt_len)
    # Random replacement tokens
    random_tokens = torch.randint(0, vocab_size, (population_size, prompt_len), dtype=torch.int64)
    # Apply mutation
    mutated = torch.where(mask, random_tokens, tiled)
    return mutated.to(torch.int64)


class BlackBoxPromptOptimizer:
    """Evolutionary black-box prompt optimizer."""

    def __init__(
        self,
        model: nn.Module,
        config: PromptOptimizerConfig,
        score_fn: Callable[[nn.Module, Tensor, Tensor], float] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.score_fn = score_fn if score_fn is not None else score_prompt
        self._history: list[dict] = []

    def optimize(self, target_ids: Tensor) -> tuple[Tensor, float]:
        """Run n_generations of evolution to find best prompt for target_ids.
        Returns (best_prompt_ids (prompt_len,), best_score float)."""
        cfg = self.config
        self._history = []

        # Initialize population
        population = initialize_population(cfg.population_size, cfg.prompt_len, cfg.vocab_size)

        best_prompt: Tensor = population[0]
        best_score: float = float("-inf")

        for gen in range(cfg.n_generations):
            # Score all candidates
            scores = torch.tensor(
                [
                    self.score_fn(self.model, population[i], target_ids)
                    for i in range(cfg.population_size)
                ],
                dtype=torch.float32,
            )

            # Track generation stats
            gen_best_score = scores.max().item()
            gen_mean_score = scores.mean().item()
            self._history.append(
                {
                    "generation": gen,
                    "best_score": gen_best_score,
                    "mean_score": gen_mean_score,
                }
            )

            # Update global best
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_prompt = population[scores.argmax()].clone()

            # Select elite
            elite, _ = select_elite(population, scores, cfg.n_elite)

            # Mutate to produce next generation (skip mutation on last gen to save compute)
            if gen < cfg.n_generations - 1:
                population = mutate_population(
                    elite, cfg.population_size, cfg.vocab_size, cfg.mutation_rate
                )
            else:
                # On the last generation we already scored, no next generation needed
                break

        return best_prompt, best_score

    def get_history(self) -> list[dict]:
        """Return per-generation history: [{"generation": int, "best_score": float, "mean_score": float}]."""  # noqa: E501
        return list(self._history)


class PromptOptimizerTrainer:
    """Wrapper for repeated prompt optimization across multiple targets."""

    def __init__(
        self,
        model: nn.Module,
        config: PromptOptimizerConfig,
    ) -> None:
        self.model = model
        self.config = config

    def optimize_batch(self, target_batch: list[Tensor]) -> list[tuple[Tensor, float]]:
        """Optimize prompt for each target in batch. Returns list of (prompt, score)."""
        results = []
        for target_ids in target_batch:
            optimizer = BlackBoxPromptOptimizer(self.model, self.config)
            prompt, score = optimizer.optimize(target_ids)
            results.append((prompt, score))
        return results

    def best_universal_prompt(
        self, target_batch: list[Tensor], n_trials: int = 3
    ) -> tuple[Tensor, float]:
        """Find one prompt that maximizes average score across all targets.
        Try n_trials random restarts, return best.
        Returns (prompt_ids (prompt_len,), mean_score float)."""
        cfg = self.config
        best_prompt: Tensor | None = None
        best_mean_score: float = float("-inf")

        for _ in range(n_trials):
            # Run one trial: optimize a population across all targets jointly
            population = initialize_population(cfg.population_size, cfg.prompt_len, cfg.vocab_size)
            trial_best_prompt: Tensor = population[0]
            trial_best_mean: float = float("-inf")

            for gen in range(cfg.n_generations):
                # Score each candidate as mean score across all targets
                scores_list = []
                for i in range(cfg.population_size):
                    candidate_scores = [
                        score_prompt(self.model, population[i], target_ids)
                        for target_ids in target_batch
                    ]
                    scores_list.append(sum(candidate_scores) / len(candidate_scores))

                scores = torch.tensor(scores_list, dtype=torch.float32)
                gen_best_idx = int(scores.argmax().item())
                gen_best_mean = scores[gen_best_idx].item()

                if gen_best_mean > trial_best_mean:
                    trial_best_mean = gen_best_mean
                    trial_best_prompt = population[gen_best_idx].clone()

                if gen < cfg.n_generations - 1:
                    elite, _ = select_elite(population, scores, cfg.n_elite)
                    population = mutate_population(
                        elite, cfg.population_size, cfg.vocab_size, cfg.mutation_rate
                    )
                else:
                    break

            if trial_best_mean > best_mean_score:
                best_mean_score = trial_best_mean
                best_prompt = trial_best_prompt

        assert best_prompt is not None  # noqa: S101
        return best_prompt, best_mean_score
