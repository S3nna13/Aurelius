"""Tests for stratified sampler."""

from __future__ import annotations

from src.data.sampler import StratifiedSampler


class TestStratifiedSampler:
    def test_sample_preserves_distribution(self):
        labels = [0, 0, 0, 1, 1, 1]
        sampler = StratifiedSampler(labels)
        sampled = sampler.sample(4)
        sampled_labels = [labels[i] for i in sampled]
        assert 0 < sum(sampled_labels) < len(sampled)

    def test_sample_respects_count(self):
        labels = [0] * 10 + [1] * 10
        sampler = StratifiedSampler(labels)
        sampled = sampler.sample(5)
        assert len(sampled) <= 5

    def test_deterministic_seed(self):
        labels = [0, 0, 1, 1]
        s1 = StratifiedSampler(labels).sample(4, seed=42)
        s2 = StratifiedSampler(labels).sample(4, seed=42)
        assert s1 == s2
