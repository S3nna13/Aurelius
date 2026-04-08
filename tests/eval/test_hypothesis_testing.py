"""Tests for statistical hypothesis testing module."""

import pytest

from src.eval.hypothesis_testing import (
    ModelComparisonSuite,
    TestResult,
    bootstrap_test,
    mcnemar_test,
    paired_t_test,
    permutation_test,
    wilcoxon_signed_rank_test,
)


# 1. TestResult.significant property
def test_test_result_significant_property():
    r_true = TestResult(statistic=2.0, p_value=0.01, reject_null=True, alpha=0.05)
    r_false = TestResult(statistic=0.5, p_value=0.5, reject_null=False, alpha=0.05)
    assert r_true.significant is True
    assert r_false.significant is False


# 2. bootstrap_test with identical data → p_value ≈ 1.0
def test_bootstrap_test_identical():
    data = [0.5] * 20
    result = bootstrap_test(data, data, n_bootstrap=100, alpha=0.05, seed=42)
    assert isinstance(result, TestResult)
    assert result.p_value > 0.3  # should be high


# 3. bootstrap_test with clearly different data → p_value < alpha
def test_bootstrap_test_different():
    data_a = [0.9] * 30
    data_b = [0.1] * 30
    result = bootstrap_test(data_a, data_b, n_bootstrap=100, alpha=0.05, seed=42)
    assert isinstance(result, TestResult)
    assert result.p_value < 0.05
    assert result.reject_null is True


# 4. mcnemar_test returns TestResult
def test_mcnemar_test_shape():
    correct_a = [True, False, True, True, False]
    correct_b = [False, True, True, False, True]
    result = mcnemar_test(correct_a, correct_b)
    assert isinstance(result, TestResult)
    assert hasattr(result, "p_value")
    assert hasattr(result, "statistic")


# 5. mcnemar_test symmetric — swapping A/B gives same p_value
def test_mcnemar_test_symmetric():
    correct_a = [True, False, True, True, False, False, True]
    correct_b = [False, True, True, False, True, True, False]
    result_ab = mcnemar_test(correct_a, correct_b)
    result_ba = mcnemar_test(correct_b, correct_a)
    assert result_ab.p_value == pytest.approx(result_ba.p_value, abs=1e-9)


# 6. permutation_test with identical data → large p_value
def test_permutation_test_identical():
    data = [0.5] * 20
    result = permutation_test(data, data, n_permutations=100, alpha=0.05, seed=42)
    assert isinstance(result, TestResult)
    assert result.p_value > 0.3


# 7. permutation_test with clearly different data → small p_value
def test_permutation_test_different():
    data_a = [0.9] * 30
    data_b = [0.1] * 30
    result = permutation_test(data_a, data_b, n_permutations=100, alpha=0.05, seed=42)
    assert isinstance(result, TestResult)
    assert result.p_value < 0.05
    assert result.reject_null is True


# 8. paired_t_test returns TestResult
def test_paired_t_test_shape():
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55]
    result = paired_t_test(scores_a, scores_b)
    assert isinstance(result, TestResult)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")


# 9. paired_t_test with same data → p_value ≈ 1.0
def test_paired_t_test_zero_diff():
    data = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = paired_t_test(data, data)
    assert isinstance(result, TestResult)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert result.reject_null is False


# 10. wilcoxon_signed_rank_test returns TestResult
def test_wilcoxon_test_shape():
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55]
    result = wilcoxon_signed_rank_test(scores_a, scores_b)
    assert isinstance(result, TestResult)
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")


# 11. ModelComparisonSuite.compare has bootstrap, permutation, t_test keys
def test_model_comparison_suite_keys():
    suite = ModelComparisonSuite(alpha=0.05)
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65, 0.72]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55, 0.62]
    results = suite.compare(scores_a, scores_b)
    assert "bootstrap" in results
    assert "permutation" in results
    assert "t_test" in results
    assert "mcnemar" not in results


# 12. ModelComparisonSuite.compare includes mcnemar when correct_a/b given
def test_model_comparison_mcnemar_included():
    suite = ModelComparisonSuite(alpha=0.05)
    scores_a = [0.6, 0.7, 0.8, 0.55, 0.65]
    scores_b = [0.5, 0.6, 0.7, 0.45, 0.55]
    correct_a = [True, True, True, False, True]
    correct_b = [False, True, True, True, False]
    results = suite.compare(scores_a, scores_b, correct_a=correct_a, correct_b=correct_b)
    assert "mcnemar" in results
    assert isinstance(results["mcnemar"], TestResult)
