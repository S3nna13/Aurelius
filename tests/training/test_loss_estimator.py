"""Tests for src/training/loss_estimator.py"""

import math

import torch
import torch.nn as nn

from src.training.loss_estimator import (
    LOSS_ESTIMATOR,
    LossEstimate,
    LossEstimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConstantLossModel(nn.Module):
    """Returns a constant scalar loss regardless of input."""

    def __init__(self, loss_value: float = 2.5):
        super().__init__()
        self._loss = loss_value
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.tensor(self._loss, requires_grad=False)


class TupleLossModel(nn.Module):
    """Returns (loss, logits) tuple — tests tuple unpacking path."""

    def __init__(self, loss_value: float = 1.0):
        super().__init__()
        self._loss = loss_value
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.tensor(self._loss, requires_grad=False), torch.zeros(2, 4)


class ErrorModel(nn.Module):
    """Always raises an exception — tests fallback path."""

    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        raise RuntimeError("intentional error")


def dummy_batch_fn(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(2, 8, dtype=torch.long)
    y = torch.zeros(2, 8, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# Tests: estimate()
# ---------------------------------------------------------------------------


def test_estimate_returns_dict_keyed_by_split():
    estimator = LossEstimator()
    model = ConstantLossModel(2.0)
    results = estimator.estimate(model, dummy_batch_fn, ["train", "val"], eval_iters=5)
    assert set(results.keys()) == {"train", "val"}


def test_estimate_result_is_loss_estimate():
    estimator = LossEstimator()
    model = ConstantLossModel(2.0)
    results = estimator.estimate(model, dummy_batch_fn, ["train"], eval_iters=5)
    assert isinstance(results["train"], LossEstimate)


def test_estimate_mean_loss_correct():
    estimator = LossEstimator()
    model = ConstantLossModel(3.0)
    results = estimator.estimate(model, dummy_batch_fn, ["train"], eval_iters=10)
    assert abs(results["train"].mean_loss - 3.0) < 1e-6


def test_estimate_n_batches_matches_eval_iters():
    estimator = LossEstimator()
    model = ConstantLossModel(1.0)
    results = estimator.estimate(model, dummy_batch_fn, ["train"], eval_iters=7)
    assert results["train"].n_batches == 7


def test_estimate_std_loss_is_zero_for_constant():
    estimator = LossEstimator()
    model = ConstantLossModel(2.0)
    results = estimator.estimate(model, dummy_batch_fn, ["train"], eval_iters=10)
    assert results["train"].std_loss < 1e-6


def test_estimate_model_restored_to_train_mode():
    estimator = LossEstimator()
    model = ConstantLossModel(1.0)
    model.train()
    estimator.estimate(model, dummy_batch_fn, ["train"], eval_iters=3)
    assert model.training


def test_estimate_tuple_output_uses_first_element():
    estimator = LossEstimator()
    model = TupleLossModel(loss_value=1.5)
    results = estimator.estimate(model, dummy_batch_fn, ["val"], eval_iters=5)
    assert abs(results["val"].mean_loss - 1.5) < 1e-6


def test_estimate_error_model_produces_invalid_mean():
    """When all batches fail, mean_loss should be not-a-number and n_batches == 0."""
    estimator = LossEstimator()
    model = ErrorModel()
    results = estimator.estimate(model, dummy_batch_fn, ["train"], eval_iters=5)
    assert math.isnan(results["train"].mean_loss)
    assert results["train"].n_batches == 0


def test_estimate_multiple_splits_independent():
    estimator = LossEstimator()
    model = ConstantLossModel(2.0)
    results = estimator.estimate(model, dummy_batch_fn, ["train", "val", "test"], eval_iters=5)
    assert len(results) == 3
    for split in ("train", "val", "test"):
        assert split in results


def test_estimate_split_name_stored_in_result():
    estimator = LossEstimator()
    model = ConstantLossModel(1.0)
    results = estimator.estimate(model, dummy_batch_fn, ["myval"], eval_iters=3)
    assert results["myval"].split == "myval"


# ---------------------------------------------------------------------------
# Tests: estimate_perplexity()
# ---------------------------------------------------------------------------


def test_perplexity_of_zero_loss():
    estimator = LossEstimator()
    assert abs(estimator.estimate_perplexity(0.0) - 1.0) < 1e-9


def test_perplexity_of_log2_is_two():
    estimator = LossEstimator()
    assert abs(estimator.estimate_perplexity(math.log(2)) - 2.0) < 1e-6


def test_perplexity_cap_at_exp20():
    estimator = LossEstimator()
    large_loss = 100.0
    result = estimator.estimate_perplexity(large_loss)
    assert abs(result - math.exp(20.0)) < 1.0


def test_perplexity_exactly_at_cap():
    estimator = LossEstimator()
    result = estimator.estimate_perplexity(20.0)
    assert abs(result - math.exp(20.0)) < 1e-6


# ---------------------------------------------------------------------------
# Tests: module-level singleton
# ---------------------------------------------------------------------------


def test_singleton_is_instance():
    assert isinstance(LOSS_ESTIMATOR, LossEstimator)
