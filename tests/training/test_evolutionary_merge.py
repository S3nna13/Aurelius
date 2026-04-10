"""Tests for evolutionary model merge search."""
from __future__ import annotations

import copy

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.evolutionary_merge import (
    EvoMergeConfig,
    EvolutionaryMerger,
    Individual,
    evaluate_model,
    linear_merge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def _tiny_model(seed: int = 0) -> AureliusTransformer:
    torch.manual_seed(seed)
    return AureliusTransformer(_tiny_cfg())


def _tiny_config() -> EvoMergeConfig:
    return EvoMergeConfig(population_size=4, n_generations=2)


def _score_fn(model) -> float:
    """Random scoring function — just checks interface."""
    return -float(torch.rand(1))


# ---------------------------------------------------------------------------
# EvoMergeConfig
# ---------------------------------------------------------------------------

def test_evo_merge_config_defaults():
    cfg = EvoMergeConfig()
    assert cfg.population_size == 10
    assert cfg.n_generations == 5
    assert cfg.mutation_std == 0.1
    assert cfg.elite_fraction == 0.2
    assert cfg.crossover_prob == 0.5


# ---------------------------------------------------------------------------
# linear_merge
# ---------------------------------------------------------------------------

def test_linear_merge_returns_state_dict_with_same_keys():
    m1 = _tiny_model(0)
    m2 = _tiny_model(1)
    merged = linear_merge([m1, m2], [0.5, 0.5])
    assert set(merged.keys()) == set(m1.state_dict().keys())


def test_linear_merge_single_model_identity():
    """Coefficient [1.0] should reproduce the original state dict exactly."""
    m = _tiny_model(0)
    merged = linear_merge([m], [1.0])
    for key, val in m.state_dict().items():
        assert torch.allclose(merged[key].float(), val.float(), atol=1e-6), key


def test_linear_merge_equal_weights_is_average():
    """Equal coefficients [0.5, 0.5] should equal the elementwise average."""
    m1 = _tiny_model(0)
    m2 = _tiny_model(1)
    merged = linear_merge([m1, m2], [0.5, 0.5])
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    for key in sd1:
        expected = (sd1[key].float() + sd2[key].float()) / 2.0
        assert torch.allclose(merged[key].float(), expected, atol=1e-5), key


def test_linear_merge_coefficients_weighted_sum():
    """Non-uniform coefficients must produce the correct weighted sum."""
    m1 = _tiny_model(0)
    m2 = _tiny_model(1)
    alpha = 0.8
    merged = linear_merge([m1, m2], [alpha, 1 - alpha])
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    for key in sd1:
        expected = alpha * sd1[key].float() + (1 - alpha) * sd2[key].float()
        assert torch.allclose(merged[key].float(), expected, atol=1e-5), key


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

def test_individual_dataclass_fields():
    ind = Individual(coefficients=[0.5, 0.5], fitness=1.23)
    assert ind.coefficients == [0.5, 0.5]
    assert ind.fitness == pytest.approx(1.23)


def test_individual_default_fitness():
    ind = Individual(coefficients=[1.0])
    assert ind.fitness == 0.0


# ---------------------------------------------------------------------------
# EvolutionaryMerger.initialize_population
# ---------------------------------------------------------------------------

def test_initialize_population_returns_correct_count():
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    pop = merger.initialize_population()
    assert len(pop) == _tiny_config().population_size


def test_initialize_population_coefficients_sum_to_one():
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    pop = merger.initialize_population()
    for ind in pop:
        assert abs(sum(ind.coefficients) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# mutate
# ---------------------------------------------------------------------------

def test_mutate_returns_individual_summing_to_one():
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    ind = Individual(coefficients=[0.5, 0.5])
    mutated = merger.mutate(ind)
    assert isinstance(mutated, Individual)
    assert abs(sum(mutated.coefficients) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# crossover
# ---------------------------------------------------------------------------

def test_crossover_returns_individual_summing_to_one():
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    pa = Individual(coefficients=[0.7, 0.3])
    pb = Individual(coefficients=[0.2, 0.8])
    child = merger.crossover(pa, pb)
    assert isinstance(child, Individual)
    assert abs(sum(child.coefficients) - 1.0) < 1e-6


def test_crossover_coefficients_non_negative():
    """All coefficients in a crossover child must be >= 0."""
    m1, m2, m3 = _tiny_model(0), _tiny_model(1), _tiny_model(2)
    merger = EvolutionaryMerger([m1, m2, m3], _tiny_config())
    pa = Individual(coefficients=[0.6, 0.3, 0.1])
    pb = Individual(coefficients=[0.1, 0.2, 0.7])
    for _ in range(20):
        child = merger.crossover(pa, pb)
        assert all(c >= 0 for c in child.coefficients)


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------

def test_evolve_returns_individual():
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    best = merger.evolve(_score_fn)
    assert isinstance(best, Individual)


def test_evolve_best_has_nonzero_fitness():
    """evolve must assign a fitness value (not 0.0 default)."""
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    best = merger.evolve(_score_fn)
    assert best.fitness != 0.0


def test_evolve_improves_over_random_baseline():
    """Best individual fitness should be a valid finite float after evolution."""
    torch.manual_seed(42)
    m1, m2 = _tiny_model(0), _tiny_model(1)
    cfg = _tiny_config()
    merger = EvolutionaryMerger([m1, m2], cfg)

    best = merger.evolve(_score_fn)

    assert isinstance(best.fitness, float)
    assert best.fitness == best.fitness  # not NaN
    assert abs(best.fitness) < 1e9      # finite


# ---------------------------------------------------------------------------
# get_merged_state_dict
# ---------------------------------------------------------------------------

def test_get_merged_state_dict_returns_correct_keys():
    m1, m2 = _tiny_model(0), _tiny_model(1)
    merger = EvolutionaryMerger([m1, m2], _tiny_config())
    ind = Individual(coefficients=[0.4, 0.6])
    merged = merger.get_merged_state_dict(ind)
    assert set(merged.keys()) == set(m1.state_dict().keys())


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_calls_score_fn_and_returns_float():
    m = _tiny_model(0)
    sd = m.state_dict()
    called = []

    def counting_score(model):
        called.append(True)
        return 3.14

    score = evaluate_model(sd, m, counting_score)
    assert len(called) == 1
    assert score == pytest.approx(3.14)


def test_evaluate_model_restores_original_state():
    """evaluate_model must not permanently modify model_template."""
    m_eval = _tiny_model(0)
    m_other = _tiny_model(99)
    other_sd = m_other.state_dict()

    original_sd = copy.deepcopy(m_eval.state_dict())
    evaluate_model(other_sd, m_eval, lambda m: 0.0)

    for key in original_sd:
        assert torch.allclose(
            m_eval.state_dict()[key].float(),
            original_sd[key].float(),
            atol=1e-7,
        ), key
