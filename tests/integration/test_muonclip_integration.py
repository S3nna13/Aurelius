"""Integration tests for MuonClip optimizer.

Verifies:
  1. "muonclip" is present in OPTIMIZER_REGISTRY exposed by src.optimizers.
  2. An optimizer constructed from the registry trains a tiny nn.Linear(4, 2)
     for 5 steps and produces a finite loss.
  3. Pre-existing OPTIMIZER_REGISTRY keys are not removed (regression guard).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import src.optimizers as optim_pkg
from src.optimizers import OPTIMIZER_REGISTRY, MuonClip


# ---------------------------------------------------------------------------
# 1. Registry contains "muonclip"
# ---------------------------------------------------------------------------

def test_muonclip_in_registry():
    assert "muonclip" in OPTIMIZER_REGISTRY, (
        "'muonclip' key missing from OPTIMIZER_REGISTRY"
    )
    assert OPTIMIZER_REGISTRY["muonclip"] is MuonClip


# ---------------------------------------------------------------------------
# 2. Construct from registry and run 5 training steps
# ---------------------------------------------------------------------------

def test_registry_construct_and_train():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    loss_fn = nn.MSELoss()

    OptimizerCls = OPTIMIZER_REGISTRY["muonclip"]
    opt = OptimizerCls(model.parameters(), lr=1e-3)

    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        opt.step()

    assert torch.isfinite(loss), f"Non-finite loss after 5 steps: {loss.item()}"


# ---------------------------------------------------------------------------
# 3. Pre-existing registry entries are still present (regression guard)
# ---------------------------------------------------------------------------

def test_existing_registry_entries_intact():
    """Ensure adding 'muonclip' did not remove any prior registry entry."""
    # These keys were present before this commit.
    prior_keys = ["schedule_free_adamw", "mars"]
    for key in prior_keys:
        assert key in OPTIMIZER_REGISTRY, (
            f"Pre-existing registry key '{key}' was removed — regression detected"
        )
