"""Tests for src/training/influence_functions.py.

All tests use tiny nn.Linear models to stay fast and avoid AureliusTransformer
overhead.  A simple cross-entropy loss wrapper is used:
    loss_fn(model, inputs, targets) = F.cross_entropy(model(inputs), targets)

Sizes: n_train=8, test_batch=2, d_in=16, d_out=4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.influence_functions import (
    InfluenceConfig,
    compute_grad,
    compute_influence_scores,
    hvp,
    lissa_inverse_hvp,
    top_influential_examples,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D_IN = 16
D_OUT = 4
N_TRAIN = 8
TEST_BATCH = 2


def _make_model(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(D_IN, D_OUT)


def _loss_fn(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(inputs), targets)


def _make_train_dataset(seed: int = 42) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(seed)
    dataset = []
    for _ in range(N_TRAIN):
        x = torch.randn(1, D_IN)
        y = torch.randint(0, D_OUT, (1,))
        dataset.append((x, y))
    return dataset


def _make_test_batch(seed: int = 99) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(TEST_BATCH, D_IN)
    y = torch.randint(0, D_OUT, (TEST_BATCH,))
    return x, y


def _make_vector(model: nn.Module) -> list[torch.Tensor]:
    """Return a list of ones tensors matching model parameter shapes."""
    return [torch.ones_like(p) for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Tests: compute_grad
# ---------------------------------------------------------------------------


class TestComputeGrad:
    def test_returns_list_matching_param_structure(self):
        """compute_grad should return one tensor per parameter."""
        torch.manual_seed(0)
        model = _make_model()
        params = [p for p in model.parameters() if p.requires_grad]
        x, y = _make_test_batch()
        grads = compute_grad(model, _loss_fn, x, y, params)

        assert isinstance(grads, list)
        assert len(grads) == len(params)
        for g, p in zip(grads, params):
            assert g.shape == p.shape, f"grad shape {g.shape} != param shape {p.shape}"

    def test_returns_list_without_explicit_params(self):
        """compute_grad should default to all requires_grad parameters."""
        torch.manual_seed(1)
        model = _make_model()
        x, y = _make_test_batch()
        grads = compute_grad(model, _loss_fn, x, y)
        params = [p for p in model.parameters() if p.requires_grad]
        assert len(grads) == len(params)

    def test_grad_is_nonzero(self):
        """Gradients should be non-zero for a non-trivial loss."""
        torch.manual_seed(2)
        model = _make_model()
        x, y = _make_test_batch()
        grads = compute_grad(model, _loss_fn, x, y)
        total_norm = sum(g.norm().item() for g in grads)
        assert total_norm > 0.0, "All gradients are zero — something went wrong."

    def test_grad_tensors_are_detached(self):
        """Returned gradient tensors must not require grad (detached)."""
        torch.manual_seed(3)
        model = _make_model()
        x, y = _make_test_batch()
        grads = compute_grad(model, _loss_fn, x, y)
        for g in grads:
            assert not g.requires_grad, "Gradient tensor is still attached to graph."


# ---------------------------------------------------------------------------
# Tests: hvp
# ---------------------------------------------------------------------------


class TestHvp:
    def test_returns_list_same_structure_as_vector(self):
        """hvp output should be a list matching the input vector structure."""
        torch.manual_seed(0)
        model = _make_model()
        x, y = _make_test_batch()
        v = _make_vector(model)
        result = hvp(model, _loss_fn, x, y, v)

        assert isinstance(result, list)
        assert len(result) == len(v)
        for r, vi in zip(result, v):
            assert r.shape == vi.shape

    def test_hvp_approximately_linear(self):
        """(H + λI)(α v) ≈ α (H + λI)(v) for scalar α."""
        torch.manual_seed(4)
        model = _make_model()
        x, y = _make_test_batch()

        v = _make_vector(model)
        alpha = 3.0
        av = [alpha * vi for vi in v]

        hv_v = hvp(model, _loss_fn, x, y, v)
        hv_av = hvp(model, _loss_fn, x, y, av)

        for h1, h2 in zip(hv_v, hv_av):
            # h2 should equal alpha * h1
            assert torch.allclose(
                alpha * h1, h2, atol=1e-4
            ), f"HVP linearity failed: max diff = {(alpha * h1 - h2).abs().max()}"

    def test_hvp_returns_finite_values(self):
        """All elements of the HVP result must be finite."""
        torch.manual_seed(5)
        model = _make_model()
        x, y = _make_test_batch()
        v = _make_vector(model)
        result = hvp(model, _loss_fn, x, y, v)
        for r in result:
            assert torch.isfinite(r).all(), "HVP contains non-finite values."

    def test_hvp_damping_increases_norm(self):
        """Higher damping λ should increase the norm of (H + λI)v."""
        torch.manual_seed(6)
        model = _make_model()
        x, y = _make_test_batch()
        v = _make_vector(model)

        hv_low = hvp(model, _loss_fn, x, y, v, damping=0.001)
        hv_high = hvp(model, _loss_fn, x, y, v, damping=10.0)

        norm_low = sum(t.norm().item() for t in hv_low)
        norm_high = sum(t.norm().item() for t in hv_high)
        assert norm_high > norm_low, "Higher damping should increase HVP norm."


# ---------------------------------------------------------------------------
# Tests: lissa_inverse_hvp
# ---------------------------------------------------------------------------


class TestLissaInverseHvp:
    def test_returns_same_structure_as_vector(self):
        """lissa_inverse_hvp output should match the vector structure."""
        torch.manual_seed(0)
        model = _make_model()
        dataset = _make_train_dataset()
        train_inputs = [x for x, _ in dataset]
        train_targets = [y for _, y in dataset]
        v = _make_vector(model)
        config = InfluenceConfig(n_recursion_depth=3, top_k=5)

        result = lissa_inverse_hvp(model, _loss_fn, train_inputs, train_targets, v, config)

        assert isinstance(result, list)
        assert len(result) == len(v)
        for r, vi in zip(result, v):
            assert r.shape == vi.shape

    def test_returns_finite_values(self):
        """All elements of lissa_inverse_hvp must be finite."""
        torch.manual_seed(10)
        model = _make_model()
        dataset = _make_train_dataset()
        train_inputs = [x for x, _ in dataset]
        train_targets = [y for _, y in dataset]
        v = _make_vector(model)
        config = InfluenceConfig(n_recursion_depth=5, recursion_scale=0.05, damping=0.1)

        result = lissa_inverse_hvp(model, _loss_fn, train_inputs, train_targets, v, config)

        for r in result:
            assert torch.isfinite(r).all(), "lissa_inverse_hvp produced non-finite values."

    def test_empty_train_set_returns_copy_of_vector(self):
        """With no training data, lissa should return a copy of the vector."""
        torch.manual_seed(7)
        model = _make_model()
        v = _make_vector(model)
        config = InfluenceConfig(n_recursion_depth=5)

        result = lissa_inverse_hvp(model, _loss_fn, [], [], v, config)

        assert len(result) == len(v)
        for r, vi in zip(result, v):
            assert torch.allclose(r, vi), "Empty train set should return copy of vector."


# ---------------------------------------------------------------------------
# Tests: compute_influence_scores
# ---------------------------------------------------------------------------


class TestComputeInfluenceScores:
    def test_returns_n_train_shape(self):
        """compute_influence_scores must return a 1-D tensor of length n_train."""
        torch.manual_seed(0)
        model = _make_model()
        dataset = _make_train_dataset()
        test_x, test_y = _make_test_batch()
        config = InfluenceConfig(n_recursion_depth=3, n_samples=N_TRAIN, top_k=3)

        scores = compute_influence_scores(model, _loss_fn, dataset, test_x, test_y, config)

        assert scores.shape == (N_TRAIN,), f"Expected shape ({N_TRAIN},), got {scores.shape}"

    def test_returns_finite_values(self):
        """All influence scores must be finite."""
        torch.manual_seed(1)
        model = _make_model()
        dataset = _make_train_dataset()
        test_x, test_y = _make_test_batch()
        config = InfluenceConfig(n_recursion_depth=3, n_samples=N_TRAIN, top_k=3)

        scores = compute_influence_scores(model, _loss_fn, dataset, test_x, test_y, config)

        assert torch.isfinite(scores).all(), "Influence scores contain non-finite values."

    def test_identical_examples_same_score(self):
        """Two identical training examples must receive identical influence scores."""
        torch.manual_seed(20)
        model = _make_model()
        torch.manual_seed(30)
        x_rep = torch.randn(1, D_IN)
        y_rep = torch.randint(0, D_OUT, (1,))
        # Build dataset where first two entries are identical
        dataset = [(x_rep.clone(), y_rep.clone()), (x_rep.clone(), y_rep.clone())]
        # Pad to n_train
        torch.manual_seed(31)
        for _ in range(N_TRAIN - 2):
            x = torch.randn(1, D_IN)
            y = torch.randint(0, D_OUT, (1,))
            dataset.append((x, y))

        test_x, test_y = _make_test_batch()
        config = InfluenceConfig(n_recursion_depth=3, n_samples=N_TRAIN, top_k=3)

        scores = compute_influence_scores(model, _loss_fn, dataset, test_x, test_y, config)

        assert torch.isclose(scores[0], scores[1], atol=1e-5), (
            f"Identical examples have different influence scores: "
            f"{scores[0].item():.6f} vs {scores[1].item():.6f}"
        )

    def test_deterministic_with_same_seed(self):
        """Same seed → same influence scores (deterministic computation)."""
        torch.manual_seed(50)
        model = _make_model()
        dataset = _make_train_dataset()
        test_x, test_y = _make_test_batch()
        config = InfluenceConfig(n_recursion_depth=3, n_samples=N_TRAIN, top_k=3)

        import random
        random.seed(42)
        torch.manual_seed(50)
        scores_a = compute_influence_scores(model, _loss_fn, dataset, test_x, test_y, config)

        random.seed(42)
        torch.manual_seed(50)
        scores_b = compute_influence_scores(model, _loss_fn, dataset, test_x, test_y, config)

        assert torch.allclose(scores_a, scores_b, atol=1e-6), (
            "Influence scores are not deterministic with the same seed."
        )


# ---------------------------------------------------------------------------
# Tests: top_influential_examples
# ---------------------------------------------------------------------------


class TestTopInfluentialExamples:
    def _make_scores(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.tensor([-3.0, 2.5, -1.0, 4.0, 0.5, -2.0, 3.5, 1.0])

    def test_returns_k_indices(self):
        """top_influential_examples must return exactly k indices."""
        scores = self._make_scores()
        k = 3
        indices, values = top_influential_examples(scores, k=k)
        assert indices.shape == (k,), f"Expected {k} indices, got {indices.shape}"

    def test_returns_k_scores(self):
        """top_influential_examples must return exactly k score values."""
        scores = self._make_scores()
        k = 3
        indices, values = top_influential_examples(scores, k=k)
        assert values.shape == (k,), f"Expected {k} values, got {values.shape}"

    def test_harmful_returns_highest_scores(self):
        """mode='harmful' must return the k examples with highest scores."""
        scores = self._make_scores()
        k = 3
        indices, values = top_influential_examples(scores, k=k, mode="harmful")
        # Verify returned values are the top-k largest
        expected_top = torch.topk(scores, k).values
        assert torch.allclose(values, expected_top), (
            f"harmful mode values {values} != expected top-k {expected_top}"
        )

    def test_helpful_returns_lowest_scores(self):
        """mode='helpful' must return the k examples with lowest (most negative) scores."""
        scores = self._make_scores()
        k = 3
        indices, values = top_influential_examples(scores, k=k, mode="helpful")
        # Verify returned values are the k smallest
        expected_bottom = torch.topk(scores, k, largest=False).values
        assert torch.allclose(values, expected_bottom), (
            f"helpful mode values {values} != expected bottom-k {expected_bottom}"
        )

    def test_harmful_indices_match_values(self):
        """Indices returned for 'harmful' must correspond to the correct scores."""
        scores = self._make_scores()
        k = 3
        indices, values = top_influential_examples(scores, k=k, mode="harmful")
        for idx, val in zip(indices, values):
            assert torch.isclose(scores[idx], val), (
                f"Score at index {idx} is {scores[idx]}, but returned value is {val}"
            )

    def test_helpful_indices_match_values(self):
        """Indices returned for 'helpful' must correspond to the correct scores."""
        scores = self._make_scores()
        k = 3
        indices, values = top_influential_examples(scores, k=k, mode="helpful")
        for idx, val in zip(indices, values):
            assert torch.isclose(scores[idx], val), (
                f"Score at index {idx} is {scores[idx]}, but returned value is {val}"
            )

    def test_invalid_mode_raises(self):
        """Passing an unknown mode must raise ValueError."""
        scores = self._make_scores()
        with pytest.raises(ValueError, match="mode must be"):
            top_influential_examples(scores, k=2, mode="unknown")

    def test_k_larger_than_n_clamps(self):
        """Requesting k > len(scores) should return all available scores."""
        scores = torch.tensor([1.0, 2.0, 3.0])
        indices, values = top_influential_examples(scores, k=10, mode="harmful")
        assert indices.shape == (3,)
