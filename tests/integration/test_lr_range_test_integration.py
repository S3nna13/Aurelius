"""Integration tests for src.training.lr_range_test."""

from __future__ import annotations

import torch
import torch.nn as nn

import src.training as training
from src.training import LRRangeTest, LRRangeTestResult


def test_exposed_via_src_training() -> None:
    assert hasattr(training, "LRRangeTest")
    assert hasattr(training, "LRRangeTestResult")
    assert training.LRRangeTest is LRRangeTest
    assert training.LRRangeTestResult is LRRangeTestResult


def test_existing_training_entries_intact() -> None:
    for name in (
        "FSDPLite",
        "ShardSpec",
        "shard_tensor",
        "gather_tensor",
        "TokenDropout",
        "LossStats",
        "LossVarianceMonitor",
    ):
        assert hasattr(training, name), f"missing pre-existing export: {name}"


def test_tiny_nn_linear_train_step_fn() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    # Fixed synthetic regression target.
    xs = torch.randn(32, 4)
    w_true = torch.randn(4, 1)
    ys = xs @ w_true + 0.1 * torch.randn(32, 1)

    def train_step_fn(lr: float) -> float:
        optim = torch.optim.SGD(model.parameters(), lr=lr)
        optim.zero_grad()
        pred = model(xs)
        loss = ((pred - ys) ** 2).mean()
        loss.backward()
        optim.step()
        return float(loss.item())

    test = LRRangeTest(
        train_step_fn,
        lr_start=1e-6,
        lr_end=1e2,
        num_steps=40,
        smooth_factor=0.9,
        divergence_threshold=10.0,
    )
    result = test.run()
    assert isinstance(result, LRRangeTestResult)
    assert len(result.lrs) == len(result.losses)
    assert 0 < len(result.lrs) <= 40
    assert result.best_lr > 0
    assert result.best_lr_div_10 == result.best_lr / 10.0
    # Losses should be finite (or the sweep stopped on divergence).
    for loss in result.losses[:-1]:
        assert loss == loss  # not NaN
