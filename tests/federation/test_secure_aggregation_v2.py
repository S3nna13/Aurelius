"""Tests for secure_aggregation_v2 — privacy-preserving model aggregation."""
from __future__ import annotations

import torch

from src.federation.secure_aggregation_v2 import (
    SecureAggregatorV2,
    aggregate_with_clipping,
    add_clip_noise,
)


class TestAggregateWithClipping:
    def test_simple_average(self):
        deltas = [
            {"w": torch.tensor([1.0, 2.0])},
            {"w": torch.tensor([3.0, 4.0])},
        ]
        result = aggregate_with_clipping(deltas, clip_norm=10.0)
        assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))

    def test_clipping_limits_outliers(self):
        big = {"w": torch.tensor([100.0, 200.0])}
        small = {"w": torch.tensor([1.0, 2.0])}
        result = aggregate_with_clipping([big, small], clip_norm=5.0)
        # Both should be clipped to norm <= 5 before averaging
        assert result["w"].norm().item() < 100.0

    def test_empty_list_returns_empty(self):
        assert aggregate_with_clipping([]) == {}

    def test_single_client_returns_clipped(self):
        delta = {"w": torch.tensor([10.0, 0.0])}
        result = aggregate_with_clipping([delta], clip_norm=5.0)
        assert result["w"].norm().item() <= 5.0


class TestAddClipNoise:
    def test_noise_changes_values(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        noisy = add_clip_noise(x, noise_scale=1.0)
        assert not torch.allclose(x, noisy)

    def test_zero_noise_no_change(self):
        x = torch.tensor([5.0, 5.0])
        noisy = add_clip_noise(x, noise_scale=0.0)
        assert torch.allclose(x, noisy)

    def test_deterministic_seed(self):
        x = torch.tensor([1.0, 1.0, 1.0, 1.0])
        r1 = add_clip_noise(x, noise_scale=0.5, seed=42)
        r2 = add_clip_noise(x, noise_scale=0.5, seed=42)
        assert torch.allclose(r1, r2)


class TestSecureAggregatorV2:
    def test_aggregate_rejects_below_threshold(self):
        agg = SecureAggregatorV2(min_clients=3)
        deltas = [
            {"w": torch.tensor([1.0])},
            {"w": torch.tensor([2.0])},
        ]
        result = agg.aggregate(deltas)
        assert result is None  # below threshold

    def test_aggregate_meets_threshold(self):
        agg = SecureAggregatorV2(min_clients=2, clip_norm=10.0, noise_scale=0.0)
        deltas = [
            {"w": torch.tensor([4.0, 6.0])},
            {"w": torch.tensor([2.0, 2.0])},
        ]
        result = agg.aggregate(deltas)
        assert result is not None
        assert torch.allclose(result["w"], torch.tensor([3.0, 4.0]))

    def test_stats_track_calls(self):
        agg = SecureAggregatorV2(min_clients=1, noise_scale=0.0, clip_norm=10.0)
        deltas = [{"w": torch.tensor([1.0, 2.0])}]
        agg.aggregate(deltas)
        stats = agg.get_stats()
        assert stats["total_aggregations"] == 1
        assert stats["skipped_low_clients"] == 0
