"""Integration test for Prodigy: exposed via src.optimizers, trains tiny MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optimizers import Prodigy


def test_prodigy_trains_tiny_mlp_20_steps():
    torch.manual_seed(0)
    # Teacher model generates targets, then student learns them.
    teacher = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    torch.manual_seed(1)
    student = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))

    x = torch.randn(32, 8)
    with torch.no_grad():
        y = teacher(x).detach()

    opt = Prodigy(student.parameters(), lr=1.0)

    with torch.no_grad():
        loss0 = ((student(x) - y) ** 2).mean().item()

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = ((student(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Require final loss lower than initial, allowing Prodigy warmup in between.
    assert losses[-1] < loss0
