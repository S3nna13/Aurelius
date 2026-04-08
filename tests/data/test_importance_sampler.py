import pytest
import torch
from torch.utils.data import WeightedRandomSampler
from src.data.importance_sampler import (
    ImportanceSamplerConfig, ImportanceWeightedSampler, DynamicCurriculumSampler
)

def test_set_weights_normalizes():
    sampler = ImportanceWeightedSampler(n_examples=10)
    ppls = torch.ones(10) * 5.0
    sampler.set_weights(ppls)
    # With uniform ppls, weights should be uniform
    weights = sampler.weights
    assert torch.allclose(weights, torch.ones(10), atol=0.01)

def test_set_weights_upweights_hard():
    """High-perplexity examples should get higher weights."""
    sampler = ImportanceWeightedSampler(n_examples=5, cfg=ImportanceSamplerConfig(temperature=1.0))
    ppls = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0])  # last is hard
    sampler.set_weights(ppls)
    weights = sampler.weights
    assert weights[-1] > weights[0]  # hard example has higher weight

def test_set_weights_min_floor():
    sampler = ImportanceWeightedSampler(n_examples=5, cfg=ImportanceSamplerConfig(min_weight=0.1))
    ppls = torch.tensor([1.0, 1.0, 1.0, 1.0, 1000.0])
    sampler.set_weights(ppls)
    weights = sampler.weights
    assert (weights > 0).all()  # no zero weights

def test_get_sampler_returns_weighted_random():
    sampler = ImportanceWeightedSampler(n_examples=10)
    sampler.set_weights(torch.ones(10))
    ws = sampler.get_sampler()
    assert isinstance(ws, WeightedRandomSampler)

def test_update_weight_ema():
    sampler = ImportanceWeightedSampler(n_examples=5)
    ppls = torch.ones(5) * 2.0
    sampler.set_weights(ppls)
    sampler.update_weight(0, 10.0)  # example 0 becomes harder
    # Weight of example 0 should now be higher
    assert sampler.weights[0] > sampler.weights[1]

def test_dynamic_curriculum_temperature_decay():
    sched = DynamicCurriculumSampler(
        n_examples=10,
        total_steps=100,
        start_temperature=10.0,
        end_temperature=1.0,
    )
    t0 = sched.get_temperature(0)
    t50 = sched.get_temperature(50)
    t100 = sched.get_temperature(100)
    assert t0 == pytest.approx(10.0)
    assert t100 == pytest.approx(1.0)
    assert t0 > t50 > t100

def test_dynamic_curriculum_step():
    sched = DynamicCurriculumSampler(n_examples=5, total_steps=10)
    ppls = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ws = sched.step(ppls)
    assert isinstance(ws, WeightedRandomSampler)
    assert sched._step == 1

def test_high_temperature_is_uniform():
    """Very high temperature -> near-uniform weights."""
    sampler = ImportanceWeightedSampler(
        n_examples=5,
        cfg=ImportanceSamplerConfig(temperature=1000.0, min_weight=0.0)
    )
    ppls = torch.tensor([1.0, 10.0, 100.0, 1000.0, 10000.0])
    sampler.set_weights(ppls)
    weights = sampler.weights
    # All should be close to 1.0 (uniform = n_examples/n_examples = 1)
    assert (weights - 1.0).abs().max() < 0.1
