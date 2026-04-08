"""Tests for pretraining mix utilities."""

import pytest
import torch

from src.training.pretraining_mix import (
    CorpusSource,
    allocate_source_tokens,
    normalize_mix_weights,
    source_probabilities,
    temperature_mix,
)


def make_sources():
    return [
        CorpusSource("web", weight=3.0),
        CorpusSource("code", weight=1.0),
        CorpusSource("math", weight=2.0, max_fraction=0.5),
    ]


def test_normalize_mix_weights_sums_to_one():
    probs = normalize_mix_weights(torch.tensor([1.0, 2.0, 3.0]))
    assert probs.sum().item() == pytest.approx(1.0)


def test_temperature_mix_changes_sharpness():
    cold = temperature_mix(torch.tensor([3.0, 1.0]), temperature=0.5)
    hot = temperature_mix(torch.tensor([3.0, 1.0]), temperature=2.0)
    assert cold[0].item() > hot[0].item()


def test_source_probabilities_returns_named_probs():
    probs = source_probabilities(make_sources())
    assert set(probs) == {"web", "code", "math"}


def test_source_probabilities_respect_caps():
    probs = source_probabilities(make_sources())
    assert probs["math"] <= 0.5 + 1e-6


def test_allocate_source_tokens_respects_budget():
    allocation = allocate_source_tokens(make_sources(), total_tokens=100)
    assert sum(allocation.values()) == 100


def test_normalize_mix_weights_rejects_negative_weights():
    with pytest.raises(ValueError):
        normalize_mix_weights(torch.tensor([1.0, -1.0]))


def test_temperature_mix_rejects_bad_temperature():
    with pytest.raises(ValueError):
        temperature_mix(torch.tensor([1.0, 2.0]), temperature=0.0)
