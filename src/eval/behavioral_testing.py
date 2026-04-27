"""Behavioral testing (CheckList-style) for NLP models."""

from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class TestCase:
    input_text: str
    expected_label: int | None = None
    metadata: dict = field(default_factory=dict)


# Prevent pytest from trying to collect this helper dataclass as a test class.
TestCase.__test__ = False


@dataclass
class TestResult:
    test_name: str
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def failed(self) -> int:
        return self.total - self.passed


# Prevent pytest from trying to collect this helper dataclass as a test class.
TestResult.__test__ = False


class BehavioralTest(ABC):
    name: str = "behavioral_test"

    @abstractmethod
    def generate_cases(self) -> list[TestCase]: ...

    def evaluate(
        self,
        model_fn: Callable[[str], int],
        cases: list[TestCase] | None = None,
    ) -> TestResult:
        if cases is None:
            cases = self.generate_cases()

        passed = 0
        total = 0
        for case in cases:
            if case.expected_label is None:
                continue
            total += 1
            prediction = model_fn(case.input_text)
            if prediction == case.expected_label:
                passed += 1

        return TestResult(test_name=self.name, passed=passed, total=total)


class InvarianceTest(BehavioralTest):
    """Tests that output is the same under perturbations."""

    name = "invariance"

    def __init__(self, base_cases: list[tuple[str, str]], expected_label: int) -> None:
        self.base_cases = base_cases
        self.expected_label = expected_label

    def generate_cases(self) -> list[TestCase]:
        cases: list[TestCase] = []
        for original, perturbed in self.base_cases:
            cases.append(TestCase(input_text=original, expected_label=self.expected_label))
            cases.append(TestCase(input_text=perturbed, expected_label=self.expected_label))
        return cases


class DirectionalTest(BehavioralTest):
    """Tests that output changes in expected direction when input changes."""

    name = "directional"

    def __init__(self, case_pairs: list[tuple[str, int, str, int]]) -> None:
        # Each tuple: (input_a, label_a, input_b, label_b)
        self.case_pairs = case_pairs

    def generate_cases(self) -> list[TestCase]:
        cases: list[TestCase] = []
        for input_a, label_a, input_b, label_b in self.case_pairs:
            cases.append(TestCase(input_text=input_a, expected_label=label_a))
            cases.append(TestCase(input_text=input_b, expected_label=label_b))
        return cases


class MinimumFunctionalityTest(BehavioralTest):
    """Tests basic capabilities the model MUST get right."""

    name = "mft"

    def __init__(self, cases: list[tuple[str, int]]) -> None:
        self._cases = cases

    def generate_cases(self) -> list[TestCase]:
        return [TestCase(input_text=text, expected_label=label) for text, label in self._cases]


class PerturbationGenerator:
    """Generates text perturbations for robustness testing."""

    def random_deletion(self, text: str, p: float = 0.1, seed: int = 42) -> str:
        rng = random.Random(seed)  # noqa: S311
        words = text.split()
        if len(words) == 1:
            return text
        kept = [w for w in words if rng.random() > p]
        if not kept:
            # Keep at least one word
            kept = [rng.choice(words)]
        return " ".join(kept)

    def random_swap(self, text: str, n: int = 1, seed: int = 42) -> str:
        rng = random.Random(seed)  # noqa: S311
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n):
            idx = rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)

    def add_typo(self, text: str, seed: int = 42) -> str:
        rng = random.Random(seed)  # noqa: S311
        words = text.split()
        eligible = [i for i, w in enumerate(words) if len(w) > 3]
        if not eligible:
            return text
        idx = rng.choice(eligible)
        word = list(words[idx])
        char_idx = rng.randint(0, len(word) - 2)
        word[char_idx], word[char_idx + 1] = word[char_idx + 1], word[char_idx]
        words[idx] = "".join(word)
        return " ".join(words)

    def add_punctuation(self, text: str) -> str:
        if text and text[-1] in string.punctuation:
            return text
        return text + "."


class BehavioralTestSuite:
    """Runs multiple tests and aggregates results."""

    def __init__(self, tests: list[BehavioralTest]) -> None:
        self.tests = tests

    def run(self, model_fn: Callable[[str], int]) -> dict[str, TestResult]:
        return {test.name: test.evaluate(model_fn) for test in self.tests}

    def summary(self, results: dict[str, TestResult]) -> dict[str, float]:
        return {name: result.pass_rate for name, result in results.items()}

    def overall_pass_rate(self, results: dict[str, TestResult]) -> float:
        total_cases = sum(r.total for r in results.values())
        if total_cases == 0:
            return 0.0
        weighted_passed = sum(r.passed for r in results.values())
        return weighted_passed / total_cases
