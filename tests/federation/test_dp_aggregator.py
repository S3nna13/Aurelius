"""Tests for DP aggregator."""

from __future__ import annotations

import math

from src.federation.dp_aggregator import (
    DP_AGGREGATOR_REGISTRY,
    DPAggregationResult,
    DPAggregator,
    DPConfig,
)


def _agg(**kw) -> DPAggregator:
    return DPAggregator(DPConfig(**kw))


def test_dp_config_defaults() -> None:
    c = DPConfig()
    assert c.noise_multiplier == 1.0
    assert c.max_grad_norm == 1.0
    assert c.delta == 1e-5
    assert c.target_epsilon == 1.0


def test_dp_config_override() -> None:
    c = DPConfig(noise_multiplier=2.0, max_grad_norm=0.5)
    assert c.noise_multiplier == 2.0
    assert c.max_grad_norm == 0.5


def test_clip_gradient_noop_when_below_norm() -> None:
    a = _agg()
    grad = [0.3, 0.4]  # norm = 0.5
    clipped = a.clip_gradient(grad, max_norm=1.0)
    assert clipped == [0.3, 0.4]


def test_clip_gradient_scales_when_above_norm() -> None:
    a = _agg()
    grad = [3.0, 4.0]  # norm = 5.0
    clipped = a.clip_gradient(grad, max_norm=1.0)
    norm = math.sqrt(sum(x * x for x in clipped))
    assert abs(norm - 1.0) < 1e-9


def test_clip_gradient_scales_to_custom_max() -> None:
    a = _agg()
    grad = [3.0, 4.0]
    clipped = a.clip_gradient(grad, max_norm=2.5)
    norm = math.sqrt(sum(x * x for x in clipped))
    assert abs(norm - 2.5) < 1e-9


def test_clip_gradient_zero_vector() -> None:
    a = _agg()
    assert a.clip_gradient([0.0, 0.0, 0.0], 1.0) == [0.0, 0.0, 0.0]


def test_clip_gradient_preserves_direction() -> None:
    a = _agg()
    grad = [6.0, 8.0]  # direction (0.6, 0.8)
    clipped = a.clip_gradient(grad, max_norm=1.0)
    assert abs(clipped[0] - 0.6) < 1e-9
    assert abs(clipped[1] - 0.8) < 1e-9


def test_add_gaussian_noise_length_preserved() -> None:
    a = _agg()
    out = a.add_gaussian_noise([0.0] * 5, std=0.1, seed=42)
    assert len(out) == 5


def test_add_gaussian_noise_reproducible_with_seed() -> None:
    a = _agg()
    out1 = a.add_gaussian_noise([0.0, 0.0, 0.0], std=0.1, seed=123)
    out2 = a.add_gaussian_noise([0.0, 0.0, 0.0], std=0.1, seed=123)
    assert out1 == out2


def test_add_gaussian_noise_zero_std() -> None:
    a = _agg()
    out = a.add_gaussian_noise([1.0, 2.0, 3.0], std=0.0, seed=1)
    assert out == [1.0, 2.0, 3.0]


def test_aggregate_result_type() -> None:
    a = _agg()
    result = a.aggregate([[0.1, 0.2], [0.3, 0.4]])
    assert isinstance(result, DPAggregationResult)


def test_aggregate_length_matches_input_dim() -> None:
    a = _agg()
    result = a.aggregate([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    assert len(result.aggregated) == 3
    assert len(result.noise_added) == 3


def test_aggregate_num_clients() -> None:
    a = _agg()
    result = a.aggregate([[0.1], [0.2], [0.3], [0.4]])
    assert result.num_clients == 4


def test_aggregate_empty_returns_zero_client() -> None:
    a = _agg()
    result = a.aggregate([])
    assert result.num_clients == 0
    assert result.aggregated == []
    assert result.epsilon_used == 0.0


def test_aggregate_epsilon_used_positive() -> None:
    a = _agg()
    result = a.aggregate([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])
    assert result.epsilon_used > 0.0


def test_aggregate_epsilon_decreases_with_more_clients() -> None:
    a = _agg()
    r1 = a.aggregate([[0.1], [0.2]])
    r2 = a.aggregate([[0.1], [0.2], [0.3], [0.4]])
    assert r2.epsilon_used < r1.epsilon_used


def test_aggregate_clips_before_averaging() -> None:
    # With huge grads, after clipping each ~unit norm, noise=0 would still
    # bound the aggregated norm. We can only sanity-check finite results.
    a = _agg(noise_multiplier=0.0)
    result = a.aggregate([[100.0, 0.0], [0.0, 100.0]])
    # Each clipped to unit norm [1,0] and [0,1]; average = [0.5, 0.5]
    # Noise std = 0.0 * 1.0 / 2 = 0
    assert abs(result.aggregated[0] - 0.5) < 1e-9
    assert abs(result.aggregated[1] - 0.5) < 1e-9


def test_aggregate_zero_noise_multiplier_gives_zero_noise() -> None:
    a = _agg(noise_multiplier=0.0)
    result = a.aggregate([[0.1, 0.2], [0.3, 0.4]])
    assert result.noise_added == [0.0, 0.0]


def test_privacy_budget_remaining_full_at_start() -> None:
    a = _agg(target_epsilon=2.0)
    assert abs(a.privacy_budget_remaining(0, 10) - 2.0) < 1e-9


def test_privacy_budget_remaining_zero_at_end() -> None:
    a = _agg(target_epsilon=2.0)
    assert a.privacy_budget_remaining(10, 10) == 0.0


def test_privacy_budget_remaining_half() -> None:
    a = _agg(target_epsilon=2.0)
    assert abs(a.privacy_budget_remaining(5, 10) - 1.0) < 1e-9


def test_privacy_budget_remaining_zero_total() -> None:
    a = _agg(target_epsilon=2.0)
    assert a.privacy_budget_remaining(0, 0) == 2.0


def test_privacy_budget_remaining_clamps_negative() -> None:
    a = _agg(target_epsilon=2.0)
    assert a.privacy_budget_remaining(20, 10) == 0.0


def test_dp_aggregation_result_frozen() -> None:
    r = DPAggregationResult(aggregated=[0.1], noise_added=[0.0], epsilon_used=0.5, num_clients=2)
    try:
        r.num_clients = 3  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("DPAggregationResult should be frozen")


def test_registry_has_default() -> None:
    assert "default" in DP_AGGREGATOR_REGISTRY
    assert DP_AGGREGATOR_REGISTRY["default"] is DPAggregator


def test_aggregator_default_config() -> None:
    a = DPAggregator()
    assert a.config.noise_multiplier == 1.0


def test_aggregate_single_client() -> None:
    a = _agg(noise_multiplier=0.0)
    result = a.aggregate([[0.3, 0.4]])  # norm 0.5 -> unchanged
    assert result.num_clients == 1
    assert abs(result.aggregated[0] - 0.3) < 1e-9
    assert abs(result.aggregated[1] - 0.4) < 1e-9
