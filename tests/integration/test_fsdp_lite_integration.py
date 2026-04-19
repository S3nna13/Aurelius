"""Integration tests for FSDPLite via the src.training package surface."""

from __future__ import annotations

import torch
import torch.nn as nn

import src.training as training_pkg
from src.training import FSDPLite, ShardSpec


def test_exposed_via_src_training():
    assert hasattr(training_pkg, "FSDPLite")
    assert hasattr(training_pkg, "ShardSpec")
    assert hasattr(training_pkg, "shard_tensor")
    assert hasattr(training_pkg, "gather_tensor")


def test_train_step_on_tiny_mlp():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    wrapped = FSDPLite(mlp, ShardSpec(world_size=2, rank=0))

    optim = torch.optim.SGD(wrapped.parameters(), lr=1e-2)
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))

    # Record initial shard state for the weight-0 param of the first linear.
    first_weight_shard = next(iter(wrapped.parameters())).clone()

    optim.zero_grad()
    logits = wrapped(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    optim.step()

    # After one step, at least one shard must have changed (nonzero grads were applied).
    updated = next(iter(wrapped.parameters()))
    assert not torch.equal(first_weight_shard, updated)
    assert torch.isfinite(loss).item()
