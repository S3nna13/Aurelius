"""Tests for src/eval/representation_similarity.py"""

from __future__ import annotations

import pytest
import torch
from aurelius.eval.representation_similarity import (
    CKAAnalyzer,
    LayerSimilarityReport,
    ProcrustesAnalyzer,
    RSAAnalyzer,
)

# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

torch.manual_seed(0)

N = 30  # number of examples
D = 16  # feature dimension
N_LAYERS = 4


@pytest.fixture()
def rng_reprs():
    """Two independent random representation matrices."""
    X = torch.randn(N, D)
    Y = torch.randn(N, D)
    return X, Y


@pytest.fixture()
def layer_reprs():
    """N_LAYERS random representation matrices (may have different d)."""
    return [torch.randn(N, D) for _ in range(N_LAYERS)]


# ===========================================================================
# CKAAnalyzer
# ===========================================================================


class TestCKAAnalyzer:
    def test_cka_identical_is_one(self):
        """CKA of identical representations should equal 1.0."""
        cka = CKAAnalyzer()
        X = torch.randn(N, D)
        result = cka.linear_cka(X, X)
        assert abs(result - 1.0) < 1e-5, f"Expected 1.0, got {result}"

    def test_cka_random_in_unit_interval(self, rng_reprs):
        """CKA of random unrelated representations should be in [0, 1]."""
        cka = CKAAnalyzer()
        X, Y = rng_reprs
        result = cka.linear_cka(X, Y)
        assert 0.0 <= result <= 1.0, f"CKA {result} out of [0, 1]"

    def test_cka_symmetric(self, rng_reprs):
        """CKA(X, Y) == CKA(Y, X)."""
        cka = CKAAnalyzer()
        X, Y = rng_reprs
        assert abs(cka.linear_cka(X, Y) - cka.linear_cka(Y, X)) < 1e-6

    def test_layer_cka_matrix_shape(self, layer_reprs):
        """layer_cka_matrix returns (n_layers, n_layers) tensor."""
        cka = CKAAnalyzer()
        mat = cka.layer_cka_matrix(layer_reprs)
        assert mat.shape == (N_LAYERS, N_LAYERS)

    def test_layer_cka_matrix_symmetric(self, layer_reprs):
        """CKA matrix is symmetric."""
        cka = CKAAnalyzer()
        mat = cka.layer_cka_matrix(layer_reprs)
        assert torch.allclose(mat, mat.T, atol=1e-6), "CKA matrix is not symmetric"

    def test_layer_cka_matrix_diagonal_is_one(self, layer_reprs):
        """Diagonal entries of CKA matrix should be 1.0."""
        cka = CKAAnalyzer()
        mat = cka.layer_cka_matrix(layer_reprs)
        for i in range(N_LAYERS):
            assert abs(mat[i, i].item() - 1.0) < 1e-5


# ===========================================================================
# RSAAnalyzer
# ===========================================================================


class TestRSAAnalyzerEuclidean:
    def test_rdm_shape(self):
        """rdm returns (N, N) tensor."""
        rsa = RSAAnalyzer(distance_fn="euclidean")
        X = torch.randn(N, D)
        rdm = rsa.rdm(X)
        assert rdm.shape == (N, N)

    def test_rdm_diagonal_zero(self):
        """Diagonal of euclidean RDM is 0 (distance to self)."""
        rsa = RSAAnalyzer(distance_fn="euclidean")
        X = torch.randn(N, D)
        rdm = rsa.rdm(X)
        assert torch.allclose(rdm.diagonal(), torch.zeros(N), atol=1e-5)

    def test_rdm_symmetric(self):
        """RDM is symmetric."""
        rsa = RSAAnalyzer(distance_fn="euclidean")
        X = torch.randn(N, D)
        rdm = rsa.rdm(X)
        assert torch.allclose(rdm, rdm.T, atol=1e-5)

    def test_rsa_correlation_range(self, rng_reprs):
        """rsa_correlation result is in [-1, 1]."""
        rsa = RSAAnalyzer(distance_fn="euclidean")
        X, Y = rng_reprs
        r = rsa.rsa_correlation(X, Y)
        assert -1.0 <= r <= 1.0, f"rsa_correlation {r} out of [-1, 1]"

    def test_rsa_correlation_identical(self):
        """rsa_correlation of identical representations is 1.0."""
        rsa = RSAAnalyzer(distance_fn="euclidean")
        X = torch.randn(N, D)
        r = rsa.rsa_correlation(X, X)
        assert abs(r - 1.0) < 1e-5, f"Expected 1.0, got {r}"


# ===========================================================================
# ProcrustesAnalyzer
# ===========================================================================


class TestProcrustesAnalyzer:
    def test_procrustes_keys(self, rng_reprs):
        """procrustes returns dict with 'residual' and 'similarity' keys."""
        proc = ProcrustesAnalyzer()
        X, Y = rng_reprs
        result = proc.procrustes(X, Y)
        assert "residual" in result
        assert "similarity" in result

    def test_procrustes_residual_nonnegative(self, rng_reprs):
        """Residual Frobenius norm is non-negative."""
        proc = ProcrustesAnalyzer()
        X, Y = rng_reprs
        result = proc.procrustes(X, Y)
        assert result["residual"] >= 0.0

    def test_procrustes_similarity_atmost_one(self, rng_reprs):
        """similarity can be <= 1 (no hard lower bound for very different spaces)."""
        proc = ProcrustesAnalyzer()
        X, Y = rng_reprs
        result = proc.procrustes(X, Y)
        assert result["similarity"] <= 1.0 + 1e-6

    def test_procrustes_identical_similarity_one(self):
        """Procrustes similarity of identical representations should be 1.0."""
        proc = ProcrustesAnalyzer()
        X = torch.randn(N, D)
        result = proc.procrustes(X, X)
        # residual should be ~0 → similarity ~1
        assert abs(result["similarity"] - 1.0) < 1e-4


# ===========================================================================
# LayerSimilarityReport
# ===========================================================================


class TestLayerSimilarityReport:
    def test_cka_profile_length(self, layer_reprs):
        """compute_cka_profile returns tensor of length n_layers - 1."""
        report = LayerSimilarityReport(n_layers=N_LAYERS)
        profile = report.compute_cka_profile(layer_reprs)
        assert profile.shape == (N_LAYERS - 1,)

    def test_find_similar_layers_pairs_ordered(self, layer_reprs):
        """find_similar_layers returns pairs with i < j."""
        report = LayerSimilarityReport(n_layers=N_LAYERS)
        # Use a low threshold so we're likely to get at least some pairs.
        pairs = report.find_similar_layers(layer_reprs, threshold=0.0)
        for i, j in pairs:
            assert i < j, f"Pair ({i}, {j}) is not ordered with i < j"

    def test_find_similar_layers_identical(self):
        """find_similar_layers with identical layers and threshold=0.9 returns all pairs."""
        X = torch.randn(N, D)
        reprs = [X.clone() for _ in range(N_LAYERS)]
        report = LayerSimilarityReport(n_layers=N_LAYERS)
        pairs = report.find_similar_layers(reprs, threshold=0.9)
        expected = N_LAYERS * (N_LAYERS - 1) // 2
        assert len(pairs) == expected

    def test_layer_change_rate_length(self, layer_reprs):
        """layer_change_rate returns a list of length n_layers - 1."""
        report = LayerSimilarityReport(n_layers=N_LAYERS)
        rates = report.layer_change_rate(layer_reprs)
        assert len(rates) == N_LAYERS - 1
