"""Integration tests for the MARS optimizer."""

from __future__ import annotations

import torch
import torch.nn as nn

import src.optimizers as optimizers
from src.optimizers import Mars


def test_mars_exposed_via_package():
    assert hasattr(optimizers, "Mars")
    assert optimizers.Mars is Mars


def test_prior_entries_intact():
    # Ensure additive append didn't remove existing exports.
    for name in ("Adafactor", "LAMB", "Lookahead", "Prodigy", "RAdam", "ScheduleFreeAdamW"):
        assert hasattr(optimizers, name), f"missing prior export {name}"


def test_train_tiny_mlp_loss_decreases():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    opt = Mars(model.parameters(), lr=3e-3, weight_decay=0.0)

    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    loss_fn = nn.MSELoss()

    opt.zero_grad()
    initial_loss = loss_fn(model(x), y).item()

    for _ in range(15):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    final_loss = loss_fn(model(x), y).item()
    assert final_loss < initial_loss
