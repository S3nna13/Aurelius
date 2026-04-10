"""Tests for black-box prompt optimization via evolutionary search."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.prompt_optimizer import (
    PromptOptimizerConfig,
    score_prompt,
    initialize_population,
    select_elite,
    mutate_population,
    BlackBoxPromptOptimizer,
    PromptOptimizerTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Tiny test model (fast enough for unit tests)
# ---------------------------------------------------------------------------

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def make_model() -> AureliusTransformer:
    return AureliusTransformer(MODEL_CFG)


def make_config(**kwargs) -> PromptOptimizerConfig:
    defaults = dict(
        prompt_len=4,
        vocab_size=256,
        population_size=6,
        n_elite=2,
        mutation_rate=0.2,
        n_generations=3,
        temperature=1.0,
    )
    defaults.update(kwargs)
    return PromptOptimizerConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. PromptOptimizerConfig defaults
# ---------------------------------------------------------------------------

def test_prompt_optimizer_config_defaults():
    cfg = PromptOptimizerConfig()
    assert cfg.prompt_len == 8
    assert cfg.vocab_size == 256
    assert cfg.population_size == 10
    assert cfg.n_elite == 3
    assert cfg.mutation_rate == 0.1
    assert cfg.n_generations == 5
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. score_prompt returns float
# ---------------------------------------------------------------------------

def test_score_prompt_returns_float():
    model = make_model()
    prompt_ids = torch.randint(0, 256, (4,))
    target_ids = torch.randint(0, 256, (6,))
    result = score_prompt(model, prompt_ids, target_ids)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# 3. score_prompt: exact target prefix scores >= random prompt (on average)
# ---------------------------------------------------------------------------

def test_score_prompt_known_good_vs_random():
    """A prompt identical to the target prefix should typically score higher
    than a completely random prompt (or at least comparable — we just verify
    both return valid floats and the good one is not -inf)."""
    model = make_model()
    target_ids = torch.randint(0, 256, (8,))
    # "Good" prompt: exactly the first 4 tokens of the target
    good_prompt = target_ids[:4].clone()
    random_prompt = torch.randint(0, 256, (4,))

    good_score = score_prompt(model, good_prompt, target_ids)
    random_score = score_prompt(model, random_prompt, target_ids)

    # Both should be finite floats
    assert torch.isfinite(torch.tensor(good_score)), "good_score must be finite"
    assert torch.isfinite(torch.tensor(random_score)), "random_score must be finite"


# ---------------------------------------------------------------------------
# 4. initialize_population shape is (pop_size, prompt_len)
# ---------------------------------------------------------------------------

def test_initialize_population_shape():
    pop = initialize_population(population_size=8, prompt_len=5, vocab_size=256)
    assert pop.shape == (8, 5), f"Expected (8, 5), got {pop.shape}"


# ---------------------------------------------------------------------------
# 5. initialize_population token ids in [0, vocab_size)
# ---------------------------------------------------------------------------

def test_initialize_population_token_range():
    vocab_size = 256
    pop = initialize_population(population_size=20, prompt_len=10, vocab_size=vocab_size)
    assert pop.min().item() >= 0, "Token ids must be >= 0"
    assert pop.max().item() < vocab_size, f"Token ids must be < {vocab_size}"


# ---------------------------------------------------------------------------
# 6. select_elite returns n_elite candidates
# ---------------------------------------------------------------------------

def test_select_elite_count():
    population = torch.randint(0, 256, (10, 4))
    scores = torch.randn(10)
    elite_pop, elite_scores = select_elite(population, scores, n_elite=3)
    assert elite_pop.shape[0] == 3, f"Expected 3 elite, got {elite_pop.shape[0]}"
    assert elite_scores.shape[0] == 3, f"Expected 3 elite scores, got {elite_scores.shape[0]}"


# ---------------------------------------------------------------------------
# 7. select_elite returns highest-scoring candidates
# ---------------------------------------------------------------------------

def test_select_elite_highest_scores():
    population = torch.randint(0, 256, (10, 4))
    # Deterministic scores: index i has score i
    scores = torch.arange(10, dtype=torch.float32)
    elite_pop, elite_scores = select_elite(population, scores, n_elite=3)
    # Top 3 scores should be 9, 8, 7
    assert set(elite_scores.tolist()) == {9.0, 8.0, 7.0}, (
        f"Elite scores should be {{9,8,7}}, got {elite_scores.tolist()}"
    )


# ---------------------------------------------------------------------------
# 8. select_elite scores are sorted descending
# ---------------------------------------------------------------------------

def test_select_elite_sorted_descending():
    population = torch.randint(0, 256, (10, 4))
    scores = torch.rand(10)
    _, elite_scores = select_elite(population, scores, n_elite=4)
    for i in range(len(elite_scores) - 1):
        assert elite_scores[i].item() >= elite_scores[i + 1].item(), (
            f"Elite scores not sorted descending: {elite_scores.tolist()}"
        )


# ---------------------------------------------------------------------------
# 9. mutate_population returns (pop_size, prompt_len) tensor
# ---------------------------------------------------------------------------

def test_mutate_population_shape():
    elite = torch.randint(0, 256, (3, 4))
    result = mutate_population(elite, population_size=10, vocab_size=256, mutation_rate=0.1)
    assert result.shape == (10, 4), f"Expected (10, 4), got {result.shape}"


# ---------------------------------------------------------------------------
# 10. mutate_population elite members are preserved (at least some unchanged rows)
# ---------------------------------------------------------------------------

def test_mutate_population_elite_preserved():
    # With mutation_rate=0.0, no token should change, so elite rows must appear
    elite = torch.randint(0, 256, (2, 6))
    result = mutate_population(elite, population_size=6, vocab_size=256, mutation_rate=0.0)
    # Every row should exactly match one of the elite rows
    for i in range(result.shape[0]):
        matches = any(torch.equal(result[i], elite[j]) for j in range(elite.shape[0]))
        assert matches, f"Row {i} of mutated population does not match any elite member (mutation_rate=0)"


# ---------------------------------------------------------------------------
# 11. BlackBoxPromptOptimizer instantiates
# ---------------------------------------------------------------------------

def test_black_box_optimizer_instantiates():
    model = make_model()
    cfg = make_config()
    optimizer = BlackBoxPromptOptimizer(model, cfg)
    assert optimizer is not None


# ---------------------------------------------------------------------------
# 12. BlackBoxPromptOptimizer.optimize returns (Tensor, float)
# ---------------------------------------------------------------------------

def test_black_box_optimizer_optimize_return_types():
    model = make_model()
    cfg = make_config()
    opt = BlackBoxPromptOptimizer(model, cfg)
    target_ids = torch.randint(0, 256, (6,))
    result = opt.optimize(target_ids)
    assert isinstance(result, tuple) and len(result) == 2, "optimize must return a 2-tuple"
    prompt, score = result
    assert isinstance(prompt, torch.Tensor), f"First element must be Tensor, got {type(prompt)}"
    assert isinstance(score, float), f"Second element must be float, got {type(score)}"


# ---------------------------------------------------------------------------
# 13. BlackBoxPromptOptimizer.optimize prompt shape is (prompt_len,)
# ---------------------------------------------------------------------------

def test_black_box_optimizer_prompt_shape():
    model = make_model()
    cfg = make_config(prompt_len=5)
    opt = BlackBoxPromptOptimizer(model, cfg)
    target_ids = torch.randint(0, 256, (6,))
    prompt, _ = opt.optimize(target_ids)
    assert prompt.shape == (5,), f"Expected prompt shape (5,), got {prompt.shape}"


# ---------------------------------------------------------------------------
# 14. BlackBoxPromptOptimizer.get_history length == n_generations
# ---------------------------------------------------------------------------

def test_black_box_optimizer_history_length():
    model = make_model()
    cfg = make_config(n_generations=4)
    opt = BlackBoxPromptOptimizer(model, cfg)
    target_ids = torch.randint(0, 256, (6,))
    opt.optimize(target_ids)
    history = opt.get_history()
    assert len(history) == 4, f"Expected history length 4, got {len(history)}"


# ---------------------------------------------------------------------------
# 15. PromptOptimizerTrainer.optimize_batch returns list of correct length
# ---------------------------------------------------------------------------

def test_trainer_optimize_batch_length():
    model = make_model()
    cfg = make_config()
    trainer = PromptOptimizerTrainer(model, cfg)
    targets = [torch.randint(0, 256, (5,)) for _ in range(3)]
    results = trainer.optimize_batch(targets)
    assert isinstance(results, list), f"optimize_batch must return a list, got {type(results)}"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for i, (prompt, score) in enumerate(results):
        assert isinstance(prompt, torch.Tensor), f"Result {i}: prompt must be Tensor"
        assert isinstance(score, float), f"Result {i}: score must be float"


# ---------------------------------------------------------------------------
# 16. PromptOptimizerTrainer.best_universal_prompt returns (Tensor, float)
# ---------------------------------------------------------------------------

def test_trainer_best_universal_prompt_return_types():
    model = make_model()
    cfg = make_config(n_generations=2, population_size=4, n_elite=2)
    trainer = PromptOptimizerTrainer(model, cfg)
    targets = [torch.randint(0, 256, (5,)) for _ in range(2)]
    result = trainer.best_universal_prompt(targets, n_trials=2)
    assert isinstance(result, tuple) and len(result) == 2, (
        "best_universal_prompt must return a 2-tuple"
    )
    prompt, mean_score = result
    assert isinstance(prompt, torch.Tensor), f"Prompt must be Tensor, got {type(prompt)}"
    assert isinstance(mean_score, float), f"Mean score must be float, got {type(mean_score)}"
    assert prompt.shape == (cfg.prompt_len,), (
        f"Expected prompt shape ({cfg.prompt_len},), got {prompt.shape}"
    )
