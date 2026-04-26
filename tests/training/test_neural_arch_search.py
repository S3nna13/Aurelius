"""Tests for differentiable neural architecture search (DARTS-style)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.neural_arch_search import (
    ArchitectureWeights,
    MixedOperation,
    NASConfig,
    NASFFNCandidate,
    NASSearcher,
    gumbel_softmax_sample,
)

D_MODEL = 64
B = 2
T = 8


# --- NASConfig ---


def test_nas_config_defaults_have_correct_keys():
    cfg = NASConfig()
    assert "d_ff_mult" in cfg.search_space
    assert "n_heads" in cfg.search_space
    assert "activation" in cfg.search_space


# --- ArchitectureWeights ---


def test_arch_weights_get_probs_valid_probabilities():
    cfg = NASConfig()
    aw = ArchitectureWeights(cfg.search_space)
    probs = aw.get_probs()
    for key, p in probs.items():
        assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)
        assert (p >= 0).all()


def test_arch_weights_get_best_returns_correct_keys():
    cfg = NASConfig()
    aw = ArchitectureWeights(cfg.search_space)
    best = aw.get_best()
    assert set(best.keys()) == set(cfg.search_space.keys())


def test_arch_weights_entropy_scalar_and_nonnegative():
    cfg = NASConfig()
    aw = ArchitectureWeights(cfg.search_space)
    ent = aw.entropy()
    assert ent.dim() == 0  # scalar
    assert ent.item() >= 0.0


# --- gumbel_softmax_sample ---


def test_gumbel_softmax_output_shape_matches_input():
    logits = torch.randn(5)
    out = gumbel_softmax_sample(logits, temperature=1.0)
    assert out.shape == logits.shape


def test_gumbel_softmax_hard_gives_one_hot():
    logits = torch.randn(5)
    out = gumbel_softmax_sample(logits, temperature=0.1, hard=True)
    # One-hot: max value should be 1.0, and sum should be 1.0
    assert torch.isclose(out.max(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)


def test_gumbel_softmax_probabilities_sum_to_one():
    logits = torch.randn(8)
    out = gumbel_softmax_sample(logits, temperature=1.0)
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-5)


# --- MixedOperation ---


def test_mixed_operation_forward_output_shape():
    ops = [nn.Linear(D_MODEL, D_MODEL) for _ in range(3)]
    arch_logits = torch.nn.Parameter(torch.zeros(3))
    mixed = MixedOperation(ops, arch_logits)
    x = torch.randn(B, T, D_MODEL)
    out = mixed(x, temperature=1.0)
    assert out.shape == (B, T, D_MODEL)


# --- NASFFNCandidate ---


@pytest.mark.parametrize("activation", ["swiglu", "gelu", "relu"])
def test_nas_ffn_candidate_forward(activation):
    candidate = NASFFNCandidate(D_MODEL, D_MODEL * 4, activation)
    x = torch.randn(B, T, D_MODEL)
    out = candidate(x)
    assert out.shape == (B, T, D_MODEL)


# --- NASSearcher ---


def test_nas_searcher_build_candidates_nonempty():
    cfg = NASConfig()
    searcher = NASSearcher(cfg, D_MODEL)
    candidates = searcher.build_candidates()
    assert len(candidates) > 0
    assert isinstance(candidates, torch.nn.ModuleList)


def test_nas_searcher_search_step_output():
    cfg = NASConfig()
    searcher = NASSearcher(cfg, D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    output, info = searcher.search_step(x)
    assert output.shape == (B, T, D_MODEL)
    assert "entropy" in info
    assert "temperature" in info
    assert "best_config" in info


def test_nas_searcher_anneal_temperature_decreases():
    cfg = NASConfig(temperature=1.0, anneal_rate=0.01, min_temperature=0.1)
    searcher = NASSearcher(cfg, D_MODEL)
    initial_temp = searcher.temperature
    searcher.anneal_temperature()
    assert searcher.temperature < initial_temp


def test_nas_searcher_get_best_architecture_returns_dict():
    cfg = NASConfig()
    searcher = NASSearcher(cfg, D_MODEL)
    best = searcher.get_best_architecture()
    assert isinstance(best, dict)
    assert set(best.keys()) == set(cfg.search_space.keys())
