"""Evolutionary search over model merging configurations.

Evolves weight coefficients for linear merging of multiple model checkpoints
using a simple genetic algorithm: Dirichlet initialization, Gaussian mutation,
uniform crossover, and elite selection.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvoMergeConfig:
    """Hyperparameters for evolutionary merge search."""

    population_size: int = 10
    n_generations: int = 5
    mutation_std: float = 0.1
    elite_fraction: float = 0.2
    crossover_prob: float = 0.5


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------

def linear_merge(
    models: list[nn.Module],
    coefficients: list[float],
) -> dict[str, torch.Tensor]:
    """Linearly combine model state dicts using given coefficients.

    Args:
        models: List of nn.Module instances.
        coefficients: List of floats (one per model) that sum to 1.

    Returns:
        Merged state dict: for each key k, sum(c_i * sd_i[k]).
    """
    if len(models) != len(coefficients):
        raise ValueError("models and coefficients must have the same length")

    state_dicts = [m.state_dict() for m in models]
    merged: dict[str, torch.Tensor] = {}

    for key in state_dicts[0]:
        merged[key] = sum(
            coefficients[i] * state_dicts[i][key].float()
            for i in range(len(models))
        ).to(state_dicts[0][key].dtype)

    return merged


def evaluate_model(
    state_dict: dict[str, torch.Tensor],
    model_template: nn.Module,
    eval_fn,
) -> float:
    """Load a state dict into model_template, evaluate, and return the score.

    Args:
        state_dict: State dict to load.
        model_template: Model instance used as template (modified in-place temporarily).
        eval_fn: Callable(model) -> float, higher is better.

    Returns:
        Float score returned by eval_fn.
    """
    original_sd = copy.deepcopy(model_template.state_dict())
    model_template.load_state_dict(state_dict)
    try:
        score = float(eval_fn(model_template))
    finally:
        model_template.load_state_dict(original_sd)
    return score


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """A candidate merge configuration."""

    coefficients: list[float]
    fitness: float = 0.0


# ---------------------------------------------------------------------------
# Evolutionary merger
# ---------------------------------------------------------------------------

class EvolutionaryMerger:
    """Evolves merge coefficients over a population of Individuals."""

    def __init__(
        self,
        base_models: list[nn.Module],
        config: EvoMergeConfig | None = None,
    ) -> None:
        self.base_models = base_models
        self.config = config or EvoMergeConfig()
        self._n = len(base_models)

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def initialize_population(self) -> list[Individual]:
        """Return population_size Individuals with Dirichlet-sampled coefficients."""
        population = []
        for _ in range(self.config.population_size):
            coeffs = np.random.dirichlet(np.ones(self._n)).tolist()
            population.append(Individual(coefficients=coeffs))
        return population

    # ------------------------------------------------------------------
    # Variation operators
    # ------------------------------------------------------------------

    def mutate(self, individual: Individual) -> Individual:
        """Add Gaussian noise and project back onto the probability simplex."""
        coeffs = np.array(individual.coefficients, dtype=float)
        noise = np.random.normal(0.0, self.config.mutation_std, size=self._n)
        coeffs = coeffs + noise
        # Clip to non-negative and renormalize
        coeffs = np.clip(coeffs, 1e-9, None)
        coeffs = coeffs / coeffs.sum()
        return Individual(coefficients=coeffs.tolist())

    def crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        """Uniform crossover of coefficients then renormalize to simplex."""
        a = np.array(parent_a.coefficients, dtype=float)
        b = np.array(parent_b.coefficients, dtype=float)
        mask = np.random.rand(self._n) < self.config.crossover_prob
        child = np.where(mask, a, b)
        child = np.clip(child, 1e-9, None)
        child = child / child.sum()
        return Individual(coefficients=child.tolist())

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------

    def evolve(self, eval_fn, n_eval_samples: int = 10) -> Individual:
        """Run the full evolutionary loop and return the best Individual.

        Args:
            eval_fn: Callable(model) -> float, higher is better.
            n_eval_samples: Unused but kept for API compatibility.

        Returns:
            Best Individual found across all generations.
        """
        population = self.initialize_population()
        n_elites = max(1, int(self.config.population_size * self.config.elite_fraction))

        # Use the first model as evaluation template (state dict is restored after each call)
        template = self.base_models[0]

        best_individual: Individual | None = None

        for gen in range(self.config.n_generations):
            # Evaluate fitness for every individual
            for ind in population:
                merged_sd = linear_merge(self.base_models, ind.coefficients)
                ind.fitness = evaluate_model(merged_sd, template, eval_fn)

            # Sort: higher fitness is better
            population.sort(key=lambda x: x.fitness, reverse=True)

            if best_individual is None or population[0].fitness > best_individual.fitness:
                best_individual = copy.deepcopy(population[0])

            logger.debug(
                "Generation %d/%d — best fitness: %.4f",
                gen + 1,
                self.config.n_generations,
                best_individual.fitness,
            )

            # Select elites
            elites = population[:n_elites]

            # Build next generation
            next_gen: list[Individual] = list(copy.deepcopy(elites))
            while len(next_gen) < self.config.population_size:
                pa = elites[np.random.randint(len(elites))]
                pb = elites[np.random.randint(len(elites))]
                child = self.crossover(pa, pb)
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        # Final evaluation on last generation to capture any improvements
        for ind in population:
            merged_sd = linear_merge(self.base_models, ind.coefficients)
            ind.fitness = evaluate_model(merged_sd, template, eval_fn)
        population.sort(key=lambda x: x.fitness, reverse=True)
        if best_individual is None or population[0].fitness > best_individual.fitness:
            best_individual = copy.deepcopy(population[0])

        return best_individual  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_merged_state_dict(self, individual: Individual) -> dict[str, torch.Tensor]:
        """Return the merged state dict for the given individual."""
        return linear_merge(self.base_models, individual.coefficients)
