"""Tests for src/evaluation/metric_aggregator.py — ≥28 test cases."""

import dataclasses
import pytest

from src.evaluation.metric_aggregator import (
    MetricAggregator,
    MetricScore,
    MetricWeight,
    METRIC_AGGREGATOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# MetricWeight / MetricScore dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_metric_weight_frozen(self):
        mw = MetricWeight(name="accuracy", weight=0.5)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            mw.weight = 1.0  # type: ignore[misc]

    def test_metric_score_frozen(self):
        ms = MetricScore(name="f1", value=0.8)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ms.value = 0.0  # type: ignore[misc]

    def test_metric_score_default_sample_count(self):
        ms = MetricScore(name="acc", value=0.9)
        assert ms.sample_count == 0

    def test_metric_score_custom_sample_count(self):
        ms = MetricScore(name="acc", value=0.9, sample_count=100)
        assert ms.sample_count == 100


# ---------------------------------------------------------------------------
# MetricAggregator.add_score / basic retrieval
# ---------------------------------------------------------------------------

class TestAddScore:
    def test_add_single_score(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.9))
        report = agg.report()
        assert "acc" in report["scores"]
        assert report["scores"]["acc"] == pytest.approx(0.9)

    def test_add_overwrites_same_name(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.5))
        agg.add_score(MetricScore(name="acc", value=0.9))
        report = agg.report()
        assert report["scores"]["acc"] == pytest.approx(0.9)

    def test_add_multiple_scores(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.8))
        agg.add_score(MetricScore(name="f1", value=0.7))
        report = agg.report()
        assert len(report["scores"]) == 2


# ---------------------------------------------------------------------------
# MetricAggregator.weighted_average
# ---------------------------------------------------------------------------

class TestWeightedAverage:
    def test_weighted_average_empty_returns_zero(self):
        agg = MetricAggregator()
        assert agg.weighted_average() == pytest.approx(0.0)

    def test_weighted_average_equal_weights_single(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.7))
        assert agg.weighted_average() == pytest.approx(0.7)

    def test_weighted_average_equal_weights_multiple(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.8))
        agg.add_score(MetricScore(name="f1", value=0.6))
        # equal weight 1.0 each → (0.8 + 0.6) / 2 = 0.7
        assert agg.weighted_average() == pytest.approx(0.7)

    def test_weighted_average_custom_weights(self):
        weights = [MetricWeight(name="acc", weight=2.0), MetricWeight(name="f1", weight=1.0)]
        agg = MetricAggregator(weights=weights)
        agg.add_score(MetricScore(name="acc", value=0.9))
        agg.add_score(MetricScore(name="f1", value=0.6))
        # (0.9*2 + 0.6*1) / (2+1) = 2.4/3 = 0.8
        assert agg.weighted_average() == pytest.approx(0.8)

    def test_weighted_average_unknown_metric_uses_weight_1(self):
        weights = [MetricWeight(name="known", weight=3.0)]
        agg = MetricAggregator(weights=weights)
        agg.add_score(MetricScore(name="known", value=1.0))
        agg.add_score(MetricScore(name="unknown", value=0.0))
        # (1.0*3 + 0.0*1) / (3+1) = 3/4 = 0.75
        assert agg.weighted_average() == pytest.approx(0.75)

    def test_weighted_average_no_weights_init(self):
        agg = MetricAggregator(weights=None)
        agg.add_score(MetricScore(name="a", value=0.5))
        agg.add_score(MetricScore(name="b", value=0.5))
        assert agg.weighted_average() == pytest.approx(0.5)

    def test_weighted_average_single_custom_weight(self):
        weights = [MetricWeight(name="m", weight=5.0)]
        agg = MetricAggregator(weights=weights)
        agg.add_score(MetricScore(name="m", value=0.4))
        assert agg.weighted_average() == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# MetricAggregator.report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_has_scores_key(self):
        agg = MetricAggregator()
        assert "scores" in agg.report()

    def test_report_has_weighted_average_key(self):
        agg = MetricAggregator()
        assert "weighted_average" in agg.report()

    def test_report_has_weights_used_key(self):
        agg = MetricAggregator()
        assert "weights_used" in agg.report()

    def test_report_weights_used_reflects_actual_weights(self):
        weights = [MetricWeight(name="acc", weight=2.0)]
        agg = MetricAggregator(weights=weights)
        agg.add_score(MetricScore(name="acc", value=0.8))
        report = agg.report()
        assert report["weights_used"]["acc"] == pytest.approx(2.0)

    def test_report_weights_used_unknown_metric_is_1(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="mystery", value=0.5))
        report = agg.report()
        assert report["weights_used"]["mystery"] == pytest.approx(1.0)

    def test_report_weighted_average_matches_method(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="a", value=0.3))
        agg.add_score(MetricScore(name="b", value=0.7))
        report = agg.report()
        assert report["weighted_average"] == pytest.approx(agg.weighted_average())


# ---------------------------------------------------------------------------
# MetricAggregator.reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_scores(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.9))
        agg.reset()
        assert agg.report()["scores"] == {}

    def test_reset_makes_weighted_average_zero(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.9))
        agg.reset()
        assert agg.weighted_average() == pytest.approx(0.0)

    def test_reset_then_add_works(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="acc", value=0.9))
        agg.reset()
        agg.add_score(MetricScore(name="f1", value=0.5))
        assert agg.report()["scores"] == {"f1": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# MetricAggregator.best
# ---------------------------------------------------------------------------

class TestBest:
    def test_best_returns_top_n(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="a", value=0.3))
        agg.add_score(MetricScore(name="b", value=0.9))
        agg.add_score(MetricScore(name="c", value=0.6))
        top = agg.best(n=2)
        assert len(top) == 2
        assert top[0].value == pytest.approx(0.9)
        assert top[1].value == pytest.approx(0.6)

    def test_best_sorted_descending(self):
        agg = MetricAggregator()
        for i, v in enumerate([0.1, 0.5, 0.9, 0.3]):
            agg.add_score(MetricScore(name=f"m{i}", value=v))
        top = agg.best(n=4)
        values = [s.value for s in top]
        assert values == sorted(values, reverse=True)

    def test_best_n_greater_than_available_returns_all(self):
        agg = MetricAggregator()
        agg.add_score(MetricScore(name="a", value=0.5))
        agg.add_score(MetricScore(name="b", value=0.7))
        top = agg.best(n=10)
        assert len(top) == 2

    def test_best_default_n_is_3(self):
        agg = MetricAggregator()
        for i in range(5):
            agg.add_score(MetricScore(name=f"m{i}", value=float(i) / 10))
        top = agg.best()
        assert len(top) == 3

    def test_best_empty(self):
        agg = MetricAggregator()
        assert agg.best(n=3) == []


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in METRIC_AGGREGATOR_REGISTRY

    def test_registry_default_is_metric_aggregator(self):
        assert METRIC_AGGREGATOR_REGISTRY["default"] is MetricAggregator
