"""Tests for src/training/influence_functions_v2.py.

All tests use tiny nn.Linear(4, 1) models with MSE loss to stay fast.
Batches are small tensors (shape [2, 4] inputs, [2, 1] targets).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.influence_functions_v2 import (
    DataInfInfluence,
    GradientSimilarity,
    HessianVectorProduct,
    LiSSAInfluence,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D_IN = 4
D_OUT = 1
BATCH_SIZE = 2
N_BATCHES = 4


def _make_model(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(D_IN, D_OUT)


def _loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """MSE loss: batch is a tuple (x, y) packed as a list/tuple."""
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def _make_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(BATCH_SIZE, D_IN)
    y = torch.randn(BATCH_SIZE, D_OUT)
    return x, y


def _make_batches(n: int = N_BATCHES) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [_make_batch(seed=i) for i in range(n)]


def _make_param_vector(model: nn.Module, fill: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        name: torch.full_like(p, fill)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _compute_test_grad(model: nn.Module) -> dict[str, torch.Tensor]:
    """Compute a real gradient to use as test_grad."""
    batch = _make_batch(seed=99)
    model.zero_grad()
    loss = _loss_fn(model, batch)
    loss.backward()
    grad = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    model.zero_grad()
    return grad


# ---------------------------------------------------------------------------
# Tests: HessianVectorProduct
# ---------------------------------------------------------------------------


class TestHessianVectorProduct:
    def test_hvp_returns_dict_with_param_keys(self):
        """hvp should return a dict with the same keys as model named parameters."""
        model = _make_model()
        hvp = HessianVectorProduct(model, _loss_fn)
        batch = _make_batch()
        vector = _make_param_vector(model)

        result = hvp.hvp(batch, vector)

        expected_keys = {name for name, p in model.named_parameters() if p.requires_grad}
        assert set(result.keys()) == expected_keys

    def test_hvp_output_has_same_shape_as_input_vector(self):
        """hvp output tensors should have the same shapes as the input vector."""
        model = _make_model()
        hvp = HessianVectorProduct(model, _loss_fn)
        batch = _make_batch()
        vector = _make_param_vector(model)

        result = hvp.hvp(batch, vector)

        for name, v in vector.items():
            assert result[name].shape == v.shape, (
                f"Shape mismatch for '{name}': got {result[name].shape}, expected {v.shape}"
            )

    def test_hvp_output_is_finite(self):
        """All values in the hvp output should be finite."""
        model = _make_model()
        hvp_obj = HessianVectorProduct(model, _loss_fn)
        batch = _make_batch()
        vector = _make_param_vector(model)

        result = hvp_obj.hvp(batch, vector)

        for name, tensor in result.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in hvp output for param '{name}'"
            )


# ---------------------------------------------------------------------------
# Tests: LiSSAInfluence
# ---------------------------------------------------------------------------


class TestLiSSAInfluence:
    def test_estimate_returns_dict_with_param_keys(self):
        """LiSSA estimate should return a dict with all model parameter keys."""
        model = _make_model(seed=1)
        batches = _make_batches()
        test_grad = _compute_test_grad(model)

        lissa = LiSSAInfluence(model, damping=0.01, scale=10.0, n_iterations=3, n_samples=1)
        result = lissa.estimate(batches, _loss_fn, test_grad)

        expected_keys = {name for name, p in model.named_parameters() if p.requires_grad}
        assert set(result.keys()) == expected_keys

    def test_estimate_output_has_correct_shapes(self):
        """LiSSA estimate tensors should match parameter shapes."""
        model = _make_model(seed=2)
        batches = _make_batches()
        test_grad = _compute_test_grad(model)

        lissa = LiSSAInfluence(model, damping=0.01, scale=10.0, n_iterations=3, n_samples=1)
        result = lissa.estimate(batches, _loss_fn, test_grad)

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert result[name].shape == p.shape, (
                    f"Shape mismatch for '{name}': got {result[name].shape}, "
                    f"expected {p.shape}"
                )

    def test_estimate_is_finite(self):
        """LiSSA estimate values should all be finite."""
        model = _make_model(seed=3)
        batches = _make_batches()
        test_grad = _compute_test_grad(model)

        lissa = LiSSAInfluence(model, damping=0.1, scale=10.0, n_iterations=5, n_samples=2)
        result = lissa.estimate(batches, _loss_fn, test_grad)

        for name, tensor in result.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in LiSSA estimate for param '{name}'"
            )


# ---------------------------------------------------------------------------
# Tests: DataInfInfluence
# ---------------------------------------------------------------------------


class TestDataInfInfluence:
    def test_compute_diagonal_hessian_returns_param_shape_dict(self):
        """compute_diagonal_hessian should return a dict matching parameter shapes."""
        model = _make_model(seed=4)
        batches = _make_batches()

        datainf = DataInfInfluence(model, damping=0.01)
        diag_H = datainf.compute_diagonal_hessian(batches, _loss_fn)

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in diag_H, f"Missing key '{name}' in diagonal Hessian"
                assert diag_H[name].shape == p.shape, (
                    f"Shape mismatch for '{name}': got {diag_H[name].shape}, "
                    f"expected {p.shape}"
                )

    def test_diagonal_hessian_values_are_nonnegative(self):
        """Diagonal Hessian (mean of squared grads) should be non-negative."""
        model = _make_model(seed=5)
        batches = _make_batches()

        datainf = DataInfInfluence(model, damping=0.01)
        diag_H = datainf.compute_diagonal_hessian(batches, _loss_fn)

        for name, tensor in diag_H.items():
            assert (tensor >= 0).all(), (
                f"Negative values found in diagonal Hessian for param '{name}'"
            )

    def test_estimate_returns_finite_values(self):
        """DataInf estimate should produce all-finite values."""
        model = _make_model(seed=6)
        batches = _make_batches()
        test_grad = _compute_test_grad(model)

        datainf = DataInfInfluence(model, damping=0.01)
        result = datainf.estimate(batches, _loss_fn, test_grad)

        for name, tensor in result.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in DataInf estimate for param '{name}'"
            )

    def test_damping_prevents_division_by_zero(self):
        """Even with near-zero diagonal, damping should keep estimate finite."""
        model = _make_model(seed=7)

        # Single batch of zeros — all gradients will be zero → diag_H all zero
        zero_batch = (torch.zeros(BATCH_SIZE, D_IN), torch.zeros(BATCH_SIZE, D_OUT))
        batches = [zero_batch]
        test_grad = _make_param_vector(model, fill=1.0)

        datainf = DataInfInfluence(model, damping=1e-3)
        result = datainf.estimate(batches, _loss_fn, test_grad)

        for name, tensor in result.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values when diagonal is near-zero for param '{name}'"
            )


# ---------------------------------------------------------------------------
# Tests: GradientSimilarity
# ---------------------------------------------------------------------------


class TestGradientSimilarity:
    def test_compute_train_grads_returns_correct_length(self):
        """compute_train_grads should return one dict per batch."""
        model = _make_model(seed=8)
        batches = _make_batches(n=5)

        gs = GradientSimilarity(model)
        train_grads = gs.compute_train_grads(batches, _loss_fn)

        assert len(train_grads) == len(batches), (
            f"Expected {len(batches)} gradient dicts, got {len(train_grads)}"
        )

    def test_influence_scores_returns_correct_length(self):
        """influence_scores should return one score per train gradient."""
        model = _make_model(seed=9)
        batches = _make_batches(n=6)
        test_grad = _compute_test_grad(model)

        gs = GradientSimilarity(model)
        train_grads = gs.compute_train_grads(batches, _loss_fn)
        scores = gs.influence_scores(train_grads, test_grad)

        assert len(scores) == len(batches), (
            f"Expected {len(batches)} scores, got {len(scores)}"
        )

    def test_influence_scores_are_finite(self):
        """All influence scores should be finite floats."""
        model = _make_model(seed=10)
        batches = _make_batches(n=4)
        test_grad = _compute_test_grad(model)

        gs = GradientSimilarity(model)
        train_grads = gs.compute_train_grads(batches, _loss_fn)
        scores = gs.influence_scores(train_grads, test_grad)

        for i, s in enumerate(scores):
            assert isinstance(s, float), f"Score {i} is not a float: {type(s)}"
            assert torch.isfinite(torch.tensor(s)), f"Score {i} is not finite: {s}"

    def test_top_k_influential_returns_k_indices(self):
        """top_k_influential should return exactly k indices."""
        model = _make_model(seed=11)
        batches = _make_batches(n=8)
        test_grad = _compute_test_grad(model)

        gs = GradientSimilarity(model)
        train_grads = gs.compute_train_grads(batches, _loss_fn)
        scores = gs.influence_scores(train_grads, test_grad)

        k = 3
        top_indices = gs.top_k_influential(scores, k=k)

        assert len(top_indices) == k, (
            f"Expected {k} indices, got {len(top_indices)}"
        )

    def test_top_k_influential_indices_are_valid(self):
        """top_k_influential indices should all be valid positions in the scores list."""
        model = _make_model(seed=12)
        batches = _make_batches(n=8)
        test_grad = _compute_test_grad(model)

        gs = GradientSimilarity(model)
        train_grads = gs.compute_train_grads(batches, _loss_fn)
        scores = gs.influence_scores(train_grads, test_grad)

        k = 4
        top_indices = gs.top_k_influential(scores, k=k)

        n = len(scores)
        for idx in top_indices:
            assert 0 <= idx < n, (
                f"Index {idx} is out of range [0, {n - 1}]"
            )
