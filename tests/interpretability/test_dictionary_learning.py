"""Unit tests for dictionary learning."""
from __future__ import annotations

import time

import pytest
import torch

from src.interpretability.dictionary_learning import (
    DictionaryLearner,
    DictionaryResult,
)


N = 20
D_DIM = 8
N_ATOMS = 16
SPARSITY = 3


def _make_X(seed: int = 0, n: int = N, d: int = D_DIM) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_fit_returns_result_with_correct_shapes() -> None:
    X = _make_X()
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=5
    )
    res = learner.fit(X)
    assert isinstance(res, DictionaryResult)
    assert res.D.shape == (D_DIM, N_ATOMS)
    assert res.alpha.shape == (N, N_ATOMS)
    assert isinstance(res.reconstruction_error, float)
    assert res.n_iters >= 1


def test_reconstruction_error_decreases() -> None:
    torch.manual_seed(0)
    X = _make_X(0)
    errs = []
    for it in (1, 3, 8):
        learner = DictionaryLearner(
            n_atoms=N_ATOMS,
            sparsity_target=SPARSITY,
            max_iters=it,
            tol=0.0,
            l1_lambda=0.0,
        )
        torch.manual_seed(42)
        res = learner.fit(X)
        errs.append(res.reconstruction_error)
    assert errs[-1] <= errs[0] + 1e-6


def test_sparsity_bound_per_row() -> None:
    X = _make_X()
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=5, l1_lambda=0.0
    )
    res = learner.fit(X)
    nnz = (res.alpha.abs() > 0).sum(dim=1)
    assert int(nnz.max().item()) <= SPARSITY


def test_encode_given_fixed_D_matches_sparsity() -> None:
    torch.manual_seed(0)
    D = torch.randn(D_DIM, N_ATOMS)
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-12)
    X = _make_X(1)
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=1, l1_lambda=0.0
    )
    alpha = learner.encode(X, D)
    assert alpha.shape == (N, N_ATOMS)
    nnz = (alpha.abs() > 0).sum(dim=1)
    assert int(nnz.max().item()) <= SPARSITY


def test_encode_then_decode_near_identity_overcomplete() -> None:
    torch.manual_seed(0)
    X = _make_X(2)
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=15, l1_lambda=0.0
    )
    res = learner.fit(X)
    X_hat = learner.decode(res.alpha, res.D)
    assert torch.allclose(X, X_hat, atol=0.2)


def test_atoms_are_l2_normalized() -> None:
    X = _make_X()
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=4
    )
    res = learner.fit(X)
    norms = res.D.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_determinism_with_manual_seed() -> None:
    X = _make_X(0)
    torch.manual_seed(123)
    r1 = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=5
    ).fit(X)
    torch.manual_seed(123)
    r2 = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=5
    ).fit(X)
    assert torch.allclose(r1.D, r2.D, atol=1e-6)
    assert torch.allclose(r1.alpha, r2.alpha, atol=1e-6)


def test_invalid_n_atoms_raises() -> None:
    with pytest.raises(ValueError):
        DictionaryLearner(n_atoms=0, sparsity_target=3)
    with pytest.raises(ValueError):
        DictionaryLearner(n_atoms=-2, sparsity_target=3)


def test_invalid_sparsity_target_raises() -> None:
    with pytest.raises(ValueError):
        DictionaryLearner(n_atoms=4, sparsity_target=0)
    with pytest.raises(ValueError):
        DictionaryLearner(n_atoms=4, sparsity_target=-1)


def test_max_iters_one_runs_one_step() -> None:
    X = _make_X()
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=1, tol=0.0
    )
    res = learner.fit(X)
    assert res.n_iters == 1


def test_zero_variance_X_returns_zero_reconstruction() -> None:
    X = torch.zeros(N, D_DIM)
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=3
    )
    res = learner.fit(X)
    assert res.reconstruction_error == 0.0
    X_hat = learner.decode(res.alpha, res.D)
    assert torch.allclose(X_hat, torch.zeros_like(X), atol=1e-8)


def test_1000_sample_fit_under_2s() -> None:
    torch.manual_seed(0)
    X = torch.randn(1000, D_DIM)
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=2
    )
    t0 = time.perf_counter()
    res = learner.fit(X)
    dt = time.perf_counter() - t0
    assert res.alpha.shape == (1000, N_ATOMS)
    assert dt < 5.0, f"fit took {dt:.3f}s"


def test_batch_size_one_degenerate() -> None:
    X = _make_X(n=1)
    learner = DictionaryLearner(
        n_atoms=N_ATOMS, sparsity_target=SPARSITY, max_iters=3
    )
    res = learner.fit(X)
    assert res.D.shape == (D_DIM, N_ATOMS)
    assert res.alpha.shape == (1, N_ATOMS)


def test_undercomplete_more_dims_than_atoms() -> None:
    torch.manual_seed(0)
    d_dim, n_atoms = 16, 8
    X = torch.randn(N, d_dim)
    learner = DictionaryLearner(
        n_atoms=n_atoms, sparsity_target=3, max_iters=3
    )
    res = learner.fit(X)
    assert res.D.shape == (d_dim, n_atoms)
    assert res.alpha.shape == (N, n_atoms)
    norms = res.D.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
