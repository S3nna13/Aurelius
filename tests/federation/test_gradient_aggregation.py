"""Tests for src/federation/gradient_aggregation.py."""

from __future__ import annotations

import math
import pytest

from src.federation.gradient_aggregation import (
    AggregationStrategy,
    GradientAggregator,
    GRADIENT_AGGREGATOR,
)


# ---------------------------------------------------------------------------
# AggregationStrategy enum
# ---------------------------------------------------------------------------

class TestAggregationStrategy:
    def test_enum_count(self):
        assert len(AggregationStrategy) == 4

    def test_fedavg_value(self):
        assert AggregationStrategy.FEDAVG == "fedavg"

    def test_fedmedian_value(self):
        assert AggregationStrategy.FEDMEDIAN == "fedmedian"

    def test_krum_value(self):
        assert AggregationStrategy.KRUM == "krum"

    def test_trimmed_mean_value(self):
        assert AggregationStrategy.TRIMMED_MEAN == "trimmed_mean"

    def test_str_subclass(self):
        assert isinstance(AggregationStrategy.FEDAVG, str)


# ---------------------------------------------------------------------------
# GradientAggregator – FedAvg
# ---------------------------------------------------------------------------

class TestFedAvg:
    def test_equal_weights_simple_mean(self):
        agg = GradientAggregator()
        grads = [[1.0, 2.0], [3.0, 4.0]]
        result = agg.fedavg(grads)
        assert abs(result[0] - 2.0) < 1e-9
        assert abs(result[1] - 3.0) < 1e-9

    def test_three_clients_equal(self):
        agg = GradientAggregator()
        grads = [[0.0], [3.0], [6.0]]
        result = agg.fedavg(grads)
        assert abs(result[0] - 3.0) < 1e-9

    def test_weighted_fedavg(self):
        agg = GradientAggregator()
        grads = [[0.0], [10.0]]
        weights = [1.0, 9.0]
        result = agg.fedavg(grads, weights=weights)
        # (0*0.1 + 10*0.9) = 9.0
        assert abs(result[0] - 9.0) < 1e-9

    def test_weighted_normalised(self):
        agg = GradientAggregator()
        grads = [[0.0], [100.0]]
        weights = [1.0, 99.0]  # sum = 100
        result = agg.fedavg(grads, weights=weights)
        assert abs(result[0] - 99.0) < 1e-9

    def test_single_gradient(self):
        agg = GradientAggregator()
        result = agg.fedavg([[5.0, 6.0]])
        assert result == [5.0, 6.0]

    def test_returns_list(self):
        agg = GradientAggregator()
        assert isinstance(agg.fedavg([[1.0]]), list)

    def test_empty_returns_empty(self):
        agg = GradientAggregator()
        assert agg.fedavg([]) == []

    def test_equal_weights_none_matches_uniform(self):
        agg = GradientAggregator()
        grads = [[2.0, 4.0], [4.0, 8.0]]
        r1 = agg.fedavg(grads, weights=None)
        r2 = agg.fedavg(grads, weights=[1.0, 1.0])
        for a, b in zip(r1, r2):
            assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# GradientAggregator – FedMedian
# ---------------------------------------------------------------------------

class TestFedMedian:
    def test_odd_count_median(self):
        agg = GradientAggregator()
        grads = [[1.0], [3.0], [5.0]]
        result = agg.fedmedian(grads)
        assert abs(result[0] - 3.0) < 1e-9

    def test_even_count_median(self):
        agg = GradientAggregator()
        grads = [[1.0], [3.0], [5.0], [7.0]]
        result = agg.fedmedian(grads)
        assert abs(result[0] - 4.0) < 1e-9

    def test_multidim(self):
        agg = GradientAggregator()
        grads = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        result = agg.fedmedian(grads)
        assert abs(result[0] - 2.0) < 1e-9
        assert abs(result[1] - 20.0) < 1e-9

    def test_single_element(self):
        agg = GradientAggregator()
        result = agg.fedmedian([[7.0, 8.0]])
        assert result == [7.0, 8.0]

    def test_returns_list(self):
        agg = GradientAggregator()
        assert isinstance(agg.fedmedian([[1.0]]), list)

    def test_empty_returns_empty(self):
        agg = GradientAggregator()
        assert agg.fedmedian([]) == []

    def test_two_elements_avg(self):
        agg = GradientAggregator()
        result = agg.fedmedian([[0.0], [10.0]])
        assert abs(result[0] - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# GradientAggregator – Trimmed Mean
# ---------------------------------------------------------------------------

class TestTrimmedMean:
    def test_no_trim_is_mean(self):
        agg = GradientAggregator(trim_fraction=0.0)
        grads = [[1.0], [2.0], [3.0], [4.0]]
        result = agg.trimmed_mean(grads, trim_fraction=0.0)
        assert abs(result[0] - 2.5) < 1e-9

    def test_trim_removes_extremes(self):
        agg = GradientAggregator()
        # with trim 0.25 on 4 elements => trim 1 from each end => keep [2, 3]
        grads = [[1.0], [2.0], [3.0], [100.0]]
        result = agg.trimmed_mean(grads, trim_fraction=0.25)
        assert abs(result[0] - 2.5) < 1e-9

    def test_uses_instance_trim_fraction_when_none(self):
        agg = GradientAggregator(trim_fraction=0.25)
        grads = [[1.0], [2.0], [3.0], [100.0]]
        result = agg.trimmed_mean(grads)
        assert abs(result[0] - 2.5) < 1e-9

    def test_returns_list(self):
        agg = GradientAggregator()
        assert isinstance(agg.trimmed_mean([[1.0]]), list)

    def test_empty_returns_empty(self):
        agg = GradientAggregator()
        assert agg.trimmed_mean([]) == []

    def test_multidim_trim(self):
        agg = GradientAggregator()
        grads = [[0.0, 100.0], [1.0, 2.0], [2.0, 3.0], [1000.0, 4.0]]
        result = agg.trimmed_mean(grads, trim_fraction=0.25)
        # dim 0: sorted=[0,1,2,1000], trim 1 each => [1, 2] => mean=1.5
        # dim 1: sorted=[2,3,4,100], trim 1 each => [3, 4] => mean=3.5
        assert abs(result[0] - 1.5) < 1e-9
        assert abs(result[1] - 3.5) < 1e-9


# ---------------------------------------------------------------------------
# GradientAggregator – Krum
# ---------------------------------------------------------------------------

class TestKrum:
    def test_returns_list(self):
        agg = GradientAggregator()
        grads = [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [100.0, 100.0]]
        result = agg.krum(grads, f=1)
        assert isinstance(result, list)

    def test_same_dim_as_input(self):
        agg = GradientAggregator()
        grads = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [100.0, 100.0, 100.0]]
        result = agg.krum(grads, f=1)
        assert len(result) == 3

    def test_filters_outlier(self):
        agg = GradientAggregator()
        # Three close gradients, one far outlier
        grads = [[1.0], [1.1], [0.9], [1000.0]]
        result = agg.krum(grads, f=1)
        # Should be close to ~1.0
        assert result[0] < 10.0

    def test_single_gradient(self):
        agg = GradientAggregator()
        result = agg.krum([[5.0, 6.0]], f=0)
        assert len(result) == 2

    def test_empty_returns_empty(self):
        agg = GradientAggregator()
        assert agg.krum([], f=1) == []

    def test_two_identical_gradients(self):
        agg = GradientAggregator()
        grads = [[1.0, 2.0], [1.0, 2.0]]
        result = agg.krum(grads, f=0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# GradientAggregator – aggregate dispatch
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_dispatch_fedavg(self):
        agg = GradientAggregator(strategy=AggregationStrategy.FEDAVG)
        grads = [[1.0, 2.0], [3.0, 4.0]]
        result = agg.aggregate(grads)
        assert abs(result[0] - 2.0) < 1e-9

    def test_dispatch_fedmedian(self):
        agg = GradientAggregator(strategy=AggregationStrategy.FEDMEDIAN)
        grads = [[1.0], [2.0], [3.0]]
        result = agg.aggregate(grads)
        assert abs(result[0] - 2.0) < 1e-9

    def test_dispatch_trimmed_mean(self):
        agg = GradientAggregator(strategy=AggregationStrategy.TRIMMED_MEAN, trim_fraction=0.25)
        grads = [[1.0], [2.0], [3.0], [100.0]]
        result = agg.aggregate(grads)
        assert abs(result[0] - 2.5) < 1e-9

    def test_dispatch_krum(self):
        agg = GradientAggregator(strategy=AggregationStrategy.KRUM)
        grads = [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [100.0, 100.0]]
        result = agg.aggregate(grads)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_aggregate_returns_list(self):
        agg = GradientAggregator()
        result = agg.aggregate([[1.0]])
        assert isinstance(result, list)

    def test_aggregate_with_weights_fedavg(self):
        agg = GradientAggregator(strategy=AggregationStrategy.FEDAVG)
        grads = [[0.0], [10.0]]
        result = agg.aggregate(grads, weights=[1.0, 9.0])
        assert abs(result[0] - 9.0) < 1e-9


# ---------------------------------------------------------------------------
# GRADIENT_AGGREGATOR singleton
# ---------------------------------------------------------------------------

class TestGradientAggregatorSingleton:
    def test_exists(self):
        assert GRADIENT_AGGREGATOR is not None

    def test_is_gradient_aggregator(self):
        assert isinstance(GRADIENT_AGGREGATOR, GradientAggregator)

    def test_default_strategy(self):
        assert GRADIENT_AGGREGATOR.strategy == AggregationStrategy.FEDAVG
