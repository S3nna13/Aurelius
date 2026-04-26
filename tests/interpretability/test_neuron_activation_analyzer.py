"""Tests for src/interpretability/neuron_activation_analyzer.py"""

from __future__ import annotations

import pytest
import torch

from src.interpretability.neuron_activation_analyzer import (
    NeuronActivationAnalyzer,
)


@pytest.fixture
def analyzer() -> NeuronActivationAnalyzer:
    return NeuronActivationAnalyzer()


def _make_activations(batch: int = 2, seq: int = 4, d_model: int = 8) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch, seq, d_model)


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_returns_list(analyzer):
    acts = _make_activations()
    stats = analyzer.compute_stats(acts, layer_idx=0)
    assert isinstance(stats, list)


def test_compute_stats_length_equals_d_model(analyzer):
    d_model = 16
    acts = _make_activations(d_model=d_model)
    stats = analyzer.compute_stats(acts, layer_idx=0)
    assert len(stats) == d_model


def test_compute_stats_layer_idx_preserved(analyzer):
    acts = _make_activations()
    stats = analyzer.compute_stats(acts, layer_idx=3)
    assert all(s.layer_idx == 3 for s in stats)


def test_compute_stats_neuron_idx_sequential(analyzer):
    d_model = 8
    acts = _make_activations(d_model=d_model)
    stats = analyzer.compute_stats(acts, layer_idx=0)
    assert [s.neuron_idx for s in stats] == list(range(d_model))


def test_compute_stats_fields_are_float(analyzer):
    acts = _make_activations()
    stats = analyzer.compute_stats(acts, layer_idx=0)
    for s in stats:
        assert isinstance(s.mean, float)
        assert isinstance(s.std, float)
        assert isinstance(s.max_val, float)


def test_compute_stats_top_tokens_length(analyzer):
    acts = _make_activations(batch=2, seq=4)
    stats = analyzer.compute_stats(acts, layer_idx=0)
    for s in stats:
        assert len(s.top_tokens) == 5  # min(5, 2*4=8) = 5


def test_compute_stats_invalid_shape_raises(analyzer):
    acts = torch.randn(4, 8)  # 2-D, not 3-D
    with pytest.raises(ValueError):
        analyzer.compute_stats(acts, layer_idx=0)


# ---------------------------------------------------------------------------
# find_dead_neurons
# ---------------------------------------------------------------------------


def test_find_dead_neurons_returns_list(analyzer):
    acts = _make_activations()
    stats = analyzer.compute_stats(acts, layer_idx=0)
    dead = analyzer.find_dead_neurons(stats)
    assert isinstance(dead, list)


def test_find_dead_neurons_detects_zero_neuron(analyzer):
    # Create activations where neuron 2 is always zero
    acts = torch.randn(2, 4, 8)
    acts[:, :, 2] = 0.0
    stats = analyzer.compute_stats(acts, layer_idx=0)
    dead = analyzer.find_dead_neurons(stats, threshold=0.01)
    assert 2 in dead


def test_find_dead_neurons_no_dead_in_normal_activations(analyzer):
    torch.manual_seed(0)
    acts = torch.randn(4, 8, 32) * 2  # large activations, nothing dead
    stats = analyzer.compute_stats(acts, layer_idx=0)
    dead = analyzer.find_dead_neurons(stats, threshold=0.001)
    # All neurons should have max_val > 0.001
    assert len(dead) == 0


# ---------------------------------------------------------------------------
# find_monosemantic_neurons
# ---------------------------------------------------------------------------


def test_find_monosemantic_returns_list(analyzer):
    acts = _make_activations()
    stats = analyzer.compute_stats(acts, layer_idx=0)
    mono = analyzer.find_monosemantic_neurons(stats)
    assert isinstance(mono, list)


def test_find_monosemantic_detects_constant_neuron(analyzer):
    # A neuron that always activates with the same value = monosemantic
    acts = torch.zeros(3, 5, 8)
    acts[:, :, 1] = 5.0  # neuron 1: constant, std=0, mean=5 => std/mean=0 < 0.9
    stats = analyzer.compute_stats(acts, layer_idx=0)
    mono = analyzer.find_monosemantic_neurons(stats, mono_threshold=0.9)
    assert 1 in mono
