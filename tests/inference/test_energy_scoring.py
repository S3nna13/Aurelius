"""Tests for energy-based reranking and Langevin dynamics sampling."""

from __future__ import annotations

import pytest
import torch

from src.inference.energy_scoring import (
    EnergyBasedGenerator,
    EnergyConfig,
    EnergyReranker,
    compute_fluency_score,
    compute_sequence_energy,
    langevin_refine,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
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


@pytest.fixture(scope="module")
def model(small_cfg):
    torch.manual_seed(42)
    m = AureliusTransformer(small_cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def energy_cfg():
    return EnergyConfig(
        n_candidates=2,
        mcmc_steps=2,
        step_size=0.01,
        temperature=1.0,
        energy_weight=1.0,
        fluency_weight=1.0,
    )


@pytest.fixture(scope="module")
def reranker(model, energy_cfg):
    return EnergyReranker(model, energy_cfg)


@pytest.fixture(scope="module")
def sample_ids(small_cfg):
    """Small batch of token ids for testing: shape (2, 6)."""
    torch.manual_seed(0)
    return torch.randint(0, small_cfg.vocab_size, (2, 6))


@pytest.fixture(scope="module")
def candidates(small_cfg):
    """List of 3 candidate tensors, each shape (1, 6)."""
    torch.manual_seed(1)
    return [torch.randint(0, small_cfg.vocab_size, (1, 6)) for _ in range(3)]


def _byte_encode(text: str) -> list[int]:
    """Encode text to bytes (ints 0-255)."""
    return list(text.encode("utf-8"))


def _byte_decode(ids: list[int]) -> str:
    """Decode bytes to text, replacing errors."""
    return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Test 1: EnergyConfig defaults
# ---------------------------------------------------------------------------


def test_energy_config_defaults():
    cfg = EnergyConfig()
    assert cfg.n_candidates == 8
    assert cfg.mcmc_steps == 10
    assert cfg.step_size == 0.01
    assert cfg.temperature == 1.0
    assert cfg.energy_weight == 1.0
    assert cfg.fluency_weight == 1.0


# ---------------------------------------------------------------------------
# Test 2: compute_sequence_energy returns shape (B,)
# ---------------------------------------------------------------------------


def test_compute_sequence_energy_shape(model, sample_ids):
    energy = compute_sequence_energy(model, sample_ids)
    B = sample_ids.shape[0]
    assert energy.shape == (B,), f"Expected ({B},), got {energy.shape}"


# ---------------------------------------------------------------------------
# Test 3: compute_sequence_energy values are finite floats
# ---------------------------------------------------------------------------


def test_compute_sequence_energy_finite(model, sample_ids):
    energy = compute_sequence_energy(model, sample_ids)
    assert torch.isfinite(energy).all(), "Energy values contain non-finite numbers"


# ---------------------------------------------------------------------------
# Test 4: compute_fluency_score returns shape (B,)
# ---------------------------------------------------------------------------


def test_compute_fluency_score_shape(model, sample_ids):
    fluency = compute_fluency_score(model, sample_ids)
    B = sample_ids.shape[0]
    assert fluency.shape == (B,), f"Expected ({B},), got {fluency.shape}"


# ---------------------------------------------------------------------------
# Test 5: compute_fluency_score values <= 0 (negative entropy)
# ---------------------------------------------------------------------------


def test_compute_fluency_score_nonpositive(model, sample_ids):
    fluency = compute_fluency_score(model, sample_ids)
    assert (fluency <= 0).all(), f"Fluency scores should be <= 0 (negative entropy), got {fluency}"


# ---------------------------------------------------------------------------
# Test 6: EnergyReranker.score returns shape (N,)
# ---------------------------------------------------------------------------


def test_reranker_score_shape(reranker, candidates):
    scores = reranker.score(candidates)
    N = len(candidates)
    assert scores.shape == (N,), f"Expected ({N},), got {scores.shape}"


# ---------------------------------------------------------------------------
# Test 7: EnergyReranker.rerank returns same number of candidates
# ---------------------------------------------------------------------------


def test_reranker_rerank_same_count(reranker, candidates):
    reranked = reranker.rerank(candidates)
    assert len(reranked) == len(candidates)


# ---------------------------------------------------------------------------
# Test 8: EnergyReranker.rerank first element has highest score
# ---------------------------------------------------------------------------


def test_reranker_rerank_first_is_best(reranker, candidates):
    reranked = reranker.rerank(candidates)
    scores_after = reranker.score(reranked)
    # First element score should be >= all others
    assert scores_after[0] >= scores_after[1:].max() - 1e-6


# ---------------------------------------------------------------------------
# Test 9: EnergyReranker.select_best returns tensor with correct shape
# ---------------------------------------------------------------------------


def test_reranker_select_best_shape(reranker, candidates):
    best = reranker.select_best(candidates)
    # Each candidate is (1, 6) -- best should be (1, 6)
    assert best.shape == candidates[0].shape


# ---------------------------------------------------------------------------
# Test 10: langevin_refine returns same shape as input
# ---------------------------------------------------------------------------


def test_langevin_refine_same_shape(model, small_cfg, energy_cfg):
    torch.manual_seed(7)
    input_ids = torch.randint(0, small_cfg.vocab_size, (1, 6))
    refined = langevin_refine(model, input_ids, energy_cfg)
    assert refined.shape == input_ids.shape


# ---------------------------------------------------------------------------
# Test 11: langevin_refine output dtype is long
# ---------------------------------------------------------------------------


def test_langevin_refine_dtype_long(model, small_cfg, energy_cfg):
    torch.manual_seed(8)
    input_ids = torch.randint(0, small_cfg.vocab_size, (1, 6))
    refined = langevin_refine(model, input_ids, energy_cfg)
    assert refined.dtype == torch.long, f"Expected torch.long, got {refined.dtype}"


# ---------------------------------------------------------------------------
# Test 12: EnergyBasedGenerator.generate_candidates returns list of strings
# ---------------------------------------------------------------------------


def test_generator_candidates_is_list_of_strings(model, energy_cfg):
    gen = EnergyBasedGenerator(model, energy_cfg, _byte_encode, _byte_decode)
    results = gen.generate_candidates("hello", max_new_tokens=4)
    assert isinstance(results, list)
    assert all(isinstance(s, str) for s in results)


# ---------------------------------------------------------------------------
# Test 13: EnergyBasedGenerator.generate_candidates returns n_candidates strings
# ---------------------------------------------------------------------------


def test_generator_candidates_count(model, energy_cfg):
    gen = EnergyBasedGenerator(model, energy_cfg, _byte_encode, _byte_decode)
    results = gen.generate_candidates("hi", max_new_tokens=4)
    assert len(results) == energy_cfg.n_candidates


# ---------------------------------------------------------------------------
# Test 14: EnergyBasedGenerator.generate_best returns a single string
# ---------------------------------------------------------------------------


def test_generator_generate_best_returns_string(model, energy_cfg):
    gen = EnergyBasedGenerator(model, energy_cfg, _byte_encode, _byte_decode)
    result = gen.generate_best("test", max_new_tokens=4)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Bonus tests for extra coverage
# ---------------------------------------------------------------------------


def test_energy_config_custom_values():
    cfg = EnergyConfig(n_candidates=4, mcmc_steps=5, step_size=0.1, temperature=0.8)
    assert cfg.n_candidates == 4
    assert cfg.mcmc_steps == 5
    assert cfg.step_size == 0.1
    assert cfg.temperature == 0.8


def test_compute_sequence_energy_single_batch(model, small_cfg):
    ids = torch.randint(0, small_cfg.vocab_size, (1, 8))
    energy = compute_sequence_energy(model, ids)
    assert energy.shape == (1,)
    assert energy.item() >= 0.0  # Cross-entropy is non-negative


def test_reranker_select_best_is_highest_scored(reranker, candidates):
    scores = reranker.score(candidates)
    best = reranker.select_best(candidates)
    best_idx = scores.argmax().item()
    assert torch.allclose(best, candidates[best_idx])


def test_langevin_refine_tokens_in_vocab(model, small_cfg, energy_cfg):
    torch.manual_seed(9)
    input_ids = torch.randint(0, small_cfg.vocab_size, (1, 5))
    refined = langevin_refine(model, input_ids, energy_cfg)
    assert (refined >= 0).all()
    assert (refined < small_cfg.vocab_size).all()
