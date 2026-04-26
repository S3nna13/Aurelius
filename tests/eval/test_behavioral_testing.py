"""Tests for src/eval/behavioral_testing.py"""

import pytest

from src.eval.behavioral_testing import (
    BehavioralTestSuite,
    DirectionalTest,
    InvarianceTest,
    MinimumFunctionalityTest,
    PerturbationGenerator,
    TestResult,
)

# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------


def test_test_result_pass_rate():
    result = TestResult(test_name="foo", passed=7, total=10)
    assert result.pass_rate == pytest.approx(0.7)


def test_test_result_failed_count():
    result = TestResult(test_name="foo", passed=3, total=10)
    assert result.failed == 7


def test_test_result_zero_total():
    result = TestResult(test_name="foo", passed=0, total=0)
    assert result.pass_rate == 0.0
    assert result.failed == 0


# ---------------------------------------------------------------------------
# MinimumFunctionalityTest
# ---------------------------------------------------------------------------


def test_mft_generate_cases_count():
    cases = [("hello world", 1), ("goodbye world", 0), ("neutral text", 1)]
    mft = MinimumFunctionalityTest(cases)
    generated = mft.generate_cases()
    assert len(generated) == 3


def test_mft_evaluate_perfect_model():
    cases = [("positive text", 1), ("another positive", 1)]
    mft = MinimumFunctionalityTest(cases)
    result = mft.evaluate(model_fn=lambda text: 1)
    assert result.pass_rate == pytest.approx(1.0)
    assert result.passed == 2
    assert result.total == 2


def test_mft_evaluate_wrong_model():
    cases = [("positive text", 1), ("another positive", 1)]
    mft = MinimumFunctionalityTest(cases)
    result = mft.evaluate(model_fn=lambda text: 0)
    assert result.pass_rate == pytest.approx(0.0)
    assert result.passed == 0


# ---------------------------------------------------------------------------
# InvarianceTest
# ---------------------------------------------------------------------------


def test_invariance_test_case_count():
    pairs = [
        ("I love this movie", "I really love this movie"),
        ("Great film", "Great film!"),
        ("Terrible experience", "Terrible experience indeed"),
    ]
    inv_test = InvarianceTest(base_cases=pairs, expected_label=1)
    cases = inv_test.generate_cases()
    # Should return 2x the number of pairs (original + perturbed)
    assert len(cases) == 2 * len(pairs)


# ---------------------------------------------------------------------------
# DirectionalTest
# ---------------------------------------------------------------------------


def test_directional_test_evaluate():
    # model returns second char parsed as int if possible, else 0
    case_pairs = [
        ("negative text", 0, "positive text", 1),
        ("bad review", 0, "good review", 1),
    ]
    directional = DirectionalTest(case_pairs=case_pairs)

    def model_fn(text: str) -> int:
        return 1 if text.startswith("positive") or text.startswith("good") else 0

    result = directional.evaluate(model_fn=model_fn)
    assert result.passed == 4
    assert result.total == 4
    assert result.pass_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PerturbationGenerator
# ---------------------------------------------------------------------------


def test_perturbation_random_deletion_shorter():
    gen = PerturbationGenerator()
    text = "the quick brown fox jumps over the lazy dog"
    perturbed = gen.random_deletion(text, p=0.5, seed=42)
    original_words = text.split()
    perturbed_words = perturbed.split()
    assert len(perturbed_words) < len(original_words)
    assert len(perturbed_words) >= 1


def test_perturbation_add_typo():
    gen = PerturbationGenerator()
    text = "This is a wonderful sentence"
    perturbed = gen.add_typo(text, seed=42)
    assert perturbed != text


# ---------------------------------------------------------------------------
# BehavioralTestSuite
# ---------------------------------------------------------------------------


def test_behavioral_suite_run_returns_all():
    mft = MinimumFunctionalityTest([("hello", 1), ("world", 0)])
    inv = InvarianceTest([("text a", "text b")], expected_label=1)
    suite = BehavioralTestSuite(tests=[mft, inv])
    results = suite.run(model_fn=lambda text: 1)
    assert "mft" in results
    assert "invariance" in results
    assert len(results) == 2


def test_behavioral_suite_overall_pass_rate():
    # mft: 2 cases, both correct → 2 passed
    # inv: 2 cases (1 pair), both predict 1, expected 1 → 2 passed
    mft = MinimumFunctionalityTest([("a", 1), ("b", 1)])
    inv = InvarianceTest([("x", "y")], expected_label=1)
    suite = BehavioralTestSuite(tests=[mft, inv])
    results = suite.run(model_fn=lambda text: 1)
    overall = suite.overall_pass_rate(results)
    # 4 total cases, all passed → 1.0
    assert overall == pytest.approx(1.0)
