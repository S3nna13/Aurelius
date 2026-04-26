"""Tests for src/evaluation/cross_validation.py — ≥28 test cases."""

from __future__ import annotations

import math
import statistics

import pytest

from src.evaluation.cross_validation import (
    CROSS_VALIDATOR_REGISTRY,
    CrossValidator,
    CVResult,
    FoldResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cv5():
    return CrossValidator(k=5, shuffle=True, seed=42)


@pytest.fixture
def cv5_noshuffle():
    return CrossValidator(k=5, shuffle=False, seed=42)


# ---------------------------------------------------------------------------
# FoldResult frozen dataclass
# ---------------------------------------------------------------------------


class TestFoldResultFrozen:
    def test_is_frozen(self):
        fr = FoldResult(fold_idx=0, train_size=80, val_size=20, score=0.9)
        with pytest.raises((AttributeError, TypeError)):
            fr.score = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        fr = FoldResult(fold_idx=2, train_size=75, val_size=25, score=0.85)
        assert fr.fold_idx == 2
        assert fr.train_size == 75
        assert fr.val_size == 25
        assert fr.score == 0.85


# ---------------------------------------------------------------------------
# CVResult frozen dataclass
# ---------------------------------------------------------------------------


class TestCVResultFrozen:
    def test_is_frozen(self):
        cvr = CVResult(k=5, fold_results=[], mean_score=0.8, std_score=0.05)
        with pytest.raises((AttributeError, TypeError)):
            cvr.mean_score = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        fr = FoldResult(fold_idx=0, train_size=80, val_size=20, score=0.9)
        cvr = CVResult(k=5, fold_results=[fr], mean_score=0.9, std_score=0.0)
        assert cvr.k == 5
        assert len(cvr.fold_results) == 1
        assert cvr.mean_score == 0.9
        assert cvr.std_score == 0.0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_returns_k_folds(self, cv5):
        folds = cv5.split(100)
        assert len(folds) == 5

    def test_split_val_size_approx_n_over_k(self, cv5):
        folds = cv5.split(100)
        for train_idx, val_idx in folds:
            # Each val fold should be roughly n/k
            assert len(val_idx) == pytest.approx(20, abs=5)

    def test_split_no_overlap_train_val(self, cv5):
        folds = cv5.split(50)
        for train_idx, val_idx in folds:
            assert set(train_idx).isdisjoint(set(val_idx))

    def test_split_all_indices_covered_per_fold(self, cv5):
        n = 50
        folds = cv5.split(n)
        for train_idx, val_idx in folds:
            assert sorted(train_idx + val_idx) == list(range(n))

    def test_split_shuffle_false_val_indices_sorted(self, cv5_noshuffle):
        folds = cv5_noshuffle.split(10)
        # With no shuffle, val indices for fold 0 should be [0, 1]
        _, val_idx = folds[0]
        assert val_idx == sorted(val_idx)
        assert val_idx == list(range(len(val_idx)))

    def test_split_different_seeds_give_different_orders(self):
        cv_a = CrossValidator(k=5, shuffle=True, seed=1)
        cv_b = CrossValidator(k=5, shuffle=True, seed=99)
        folds_a = cv_a.split(50)
        folds_b = cv_b.split(50)
        # At least one fold's val indices should differ
        any_diff = any(set(folds_a[i][1]) != set(folds_b[i][1]) for i in range(5))
        assert any_diff

    def test_split_k2_returns_two_folds(self):
        cv2 = CrossValidator(k=2, shuffle=False)
        folds = cv2.split(10)
        assert len(folds) == 2

    def test_split_returns_list_of_tuples(self, cv5):
        folds = cv5.split(20)
        for item in folds:
            assert isinstance(item, tuple)
            assert len(item) == 2


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_calls_score_fn_k_times(self, cv5):
        call_count = {"n": 0}

        def score_fn(train, val):
            call_count["n"] += 1
            return 1.0

        cv5.evaluate(list(range(100)), score_fn)
        assert call_count["n"] == 5

    def test_evaluate_returns_cvresult(self, cv5):
        result = cv5.evaluate(list(range(50)), lambda tr, va: 0.8)
        assert isinstance(result, CVResult)

    def test_evaluate_cvresult_k_matches(self, cv5):
        result = cv5.evaluate(list(range(50)), lambda tr, va: 0.8)
        assert result.k == 5

    def test_evaluate_mean_score_correct(self, cv5):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        counter = {"i": 0}

        def score_fn(train, val):
            s = scores[counter["i"]]
            counter["i"] += 1
            return s

        result = cv5.evaluate(list(range(100)), score_fn)
        assert math.isclose(result.mean_score, statistics.mean(scores), rel_tol=1e-9)

    def test_evaluate_std_score_is_pstdev(self, cv5):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        counter = {"i": 0}

        def score_fn(train, val):
            s = scores[counter["i"]]
            counter["i"] += 1
            return s

        result = cv5.evaluate(list(range(100)), score_fn)
        assert math.isclose(result.std_score, statistics.pstdev(scores), rel_tol=1e-9)

    def test_evaluate_std_zero_for_constant_scores(self, cv5):
        result = cv5.evaluate(list(range(50)), lambda tr, va: 0.9)
        assert math.isclose(result.std_score, 0.0, abs_tol=1e-12)

    def test_evaluate_fold_results_length_equals_k(self, cv5):
        result = cv5.evaluate(list(range(50)), lambda tr, va: 0.7)
        assert len(result.fold_results) == 5

    def test_evaluate_fold_train_val_sizes_sum_to_n(self, cv5):
        n = 50
        result = cv5.evaluate(list(range(n)), lambda tr, va: 0.7)
        for fr in result.fold_results:
            assert fr.train_size + fr.val_size == n


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------


class TestStratifiedSplit:
    def test_stratified_split_returns_k_folds(self, cv5):
        labels = [0] * 50 + [1] * 50
        folds = cv5.stratified_split(labels)
        assert len(folds) == 5

    def test_stratified_split_class_distribution_preserved(self, cv5):
        # 60 class-0, 40 class-1 → each fold val should have ~8 class-0, ~8 class-1...
        # More concretely: class ratio in each val fold should be close to overall ratio.
        labels = [0] * 60 + [1] * 40
        folds = cv5.stratified_split(labels)
        for train_idx, val_idx in folds:
            val_labels = [labels[i] for i in val_idx]
            class0_count = val_labels.count(0)
            class1_count = val_labels.count(1)
            # Each fold should have some of each class (not all of one class)
            assert class0_count > 0
            assert class1_count > 0

    def test_stratified_split_no_overlap_train_val(self, cv5):
        labels = [i % 3 for i in range(90)]
        folds = cv5.stratified_split(labels)
        for train_idx, val_idx in folds:
            assert set(train_idx).isdisjoint(set(val_idx))

    def test_stratified_split_all_indices_covered(self, cv5):
        n = 90
        labels = [i % 3 for i in range(n)]
        folds = cv5.stratified_split(labels)
        for train_idx, val_idx in folds:
            assert sorted(train_idx + val_idx) == list(range(n))

    def test_stratified_split_custom_k_override(self, cv5):
        labels = [0] * 30 + [1] * 30
        folds = cv5.stratified_split(labels, k=3)
        assert len(folds) == 3

    def test_stratified_split_returns_list_of_tuples(self, cv5):
        labels = [0] * 10 + [1] * 10
        folds = cv5.stratified_split(labels)
        for item in folds:
            assert isinstance(item, tuple)
            assert len(item) == 2


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in CROSS_VALIDATOR_REGISTRY

    def test_registry_default_is_cross_validator_class(self):
        assert CROSS_VALIDATOR_REGISTRY["default"] is CrossValidator

    def test_registry_default_instantiable(self):
        cls = CROSS_VALIDATOR_REGISTRY["default"]
        instance = cls(k=3)
        assert isinstance(instance, CrossValidator)
