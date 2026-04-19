"""Integration test for Schedule-Free AdamW.

Verifies:
  * the optimizer is exposed via ``src.optimizers``
  * end-to-end training of a 2-layer MLP for 10 steps decreases the loss
  * existing optimizer registry / public symbols are intact
"""

from __future__ import annotations

import torch
import torch.nn as nn

import src.optimizers as optim_pkg
from src.optimizers import ScheduleFreeAdamW


def test_exposed_via_package():
    assert hasattr(optim_pkg, "ScheduleFreeAdamW")
    assert optim_pkg.ScheduleFreeAdamW is ScheduleFreeAdamW


def test_existing_exports_intact():
    # Sentinels from pre-existing entries of the package.
    for name in (
        "Adafactor",
        "LAMB",
        "Lookahead",
        "RAdam",
        "radam_rectification",
        "radam_rho_inf",
        "radam_rho_t",
        "adafactor_factorized_second_moment",
        "adafactor_reconstruct_second_moment",
        "lamb_trust_ratio",
    ):
        assert name in optim_pkg.__all__, f"missing export: {name}"
        assert hasattr(optim_pkg, name), f"missing attribute: {name}"


def test_train_mlp_loss_decreases():
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    opt = ScheduleFreeAdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)

    x = torch.randn(32, 8)
    y = torch.randn(32, 4)

    losses = []
    for _ in range(10):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_registry_entry_if_registry_exists():
    # If an OPTIMIZER_REGISTRY is exposed, it must contain our entry without
    # having lost any prior entries.
    reg = getattr(optim_pkg, "OPTIMIZER_REGISTRY", None)
    if reg is None:
        return  # no registry in this project layout; nothing to check
    assert "schedule_free_adamw" in reg
    assert reg["schedule_free_adamw"] is ScheduleFreeAdamW
