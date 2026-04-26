"""Integration tests for dictionary_learning exposure via src.interpretability."""

from __future__ import annotations

import torch

import src.interpretability as interp
from src.interpretability import DictionaryLearner, DictionaryResult


def test_exposed_at_package_level() -> None:
    assert hasattr(interp, "DictionaryLearner")
    assert hasattr(interp, "DictionaryResult")


def test_prior_entries_intact() -> None:
    for name in (
        "AttributionNode",
        "AttributionEdge",
        "AttributionGraph",
        "AttributionGraphBuilder",
    ):
        assert hasattr(interp, name), f"missing prior export: {name}"


def test_small_end_to_end_pipeline() -> None:
    torch.manual_seed(7)
    # build a synthetic overcomplete signal from a known dictionary
    d_dim, k = 8, 16
    D_true = torch.randn(d_dim, k)
    D_true = D_true / D_true.norm(dim=0, keepdim=True).clamp_min(1e-12)
    N = 24
    alpha_true = torch.zeros(N, k)
    for i in range(N):
        idx = torch.randperm(k)[:3]
        alpha_true[i, idx] = torch.randn(3)
    X = alpha_true @ D_true.T

    learner = DictionaryLearner(n_atoms=k, sparsity_target=3, max_iters=15, l1_lambda=0.0, tol=1e-6)
    res = learner.fit(X)
    assert isinstance(res, DictionaryResult)
    assert res.D.shape == (d_dim, k)
    assert res.alpha.shape == (N, k)

    # encode / decode round-trip using fit artefacts
    alpha_re = learner.encode(X, res.D)
    X_hat = learner.decode(alpha_re, res.D)
    # overcomplete, known support size matches: should reconstruct well
    rel = (X - X_hat).norm() / X.norm().clamp_min(1e-12)
    assert rel.item() < 0.5

    # sparsity honored on re-encode
    nnz = (alpha_re.abs() > 0).sum(dim=1)
    assert int(nnz.max().item()) <= 3
