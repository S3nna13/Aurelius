"""Tests for silo comparator."""

from __future__ import annotations

import numpy as np
import pytest

from src.federation.silo_comparator import SiloComparator


class TestSiloComparator:
    def test_cosine_similarity_identical(self):
        sc = SiloComparator()
        w = {"a": np.array([1.0, 2.0, 3.0])}
        sc.snapshot("s1", w)
        sc.snapshot("s2", w)
        assert sc.cosine_similarity("s1", "s2") == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        sc = SiloComparator()
        sc.snapshot("s1", {"a": np.array([1.0, 0.0])})
        sc.snapshot("s2", {"a": np.array([0.0, 1.0])})
        assert sc.cosine_similarity("s1", "s2") == pytest.approx(0.0)

    def test_cosine_similarity_missing(self):
        sc = SiloComparator()
        assert sc.cosine_similarity("x", "y") == 0.0

    def test_weight_divergence(self):
        sc = SiloComparator()
        sc.snapshot("s1", {"a": np.array([1.0, 1.0])})
        sc.snapshot("s2", {"a": np.array([2.0, 2.0])})
        assert sc.weight_divergence("s1", "s2") == pytest.approx(2.0)

    def test_weight_divergence_missing(self):
        sc = SiloComparator()
        assert sc.weight_divergence("x", "y") == float("inf")
