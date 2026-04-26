"""Integration tests: preference_ranking_loss exposed via src.alignment."""

from __future__ import annotations

import torch

import src.alignment as alignment_pkg
from src.alignment import (
    bradley_terry_loss,
    dpo_pair_loss,
    listnet_loss,
    margin_ranking_loss,
    ordinal_ranking_loss,
)


def test_exposed_via_package():
    for name in [
        "bradley_terry_loss",
        "margin_ranking_loss",
        "listnet_loss",
        "ordinal_ranking_loss",
        "dpo_pair_loss",
    ]:
        assert hasattr(alignment_pkg, name), f"{name} not exposed on src.alignment"


def test_one_pass_through_all_losses():
    torch.manual_seed(42)
    B, K = 4, 6

    # Reward-model-style pairwise
    chosen = torch.randn(B, requires_grad=True)
    rejected = torch.randn(B, requires_grad=True)
    bt = bradley_terry_loss(chosen, rejected)
    mr = margin_ranking_loss(chosen, rejected, margin=0.5)

    # Listwise
    pred = torch.randn(B, K, requires_grad=True)
    true = torch.randn(B, K)
    ranks = torch.stack([torch.randperm(K) for _ in range(B)])
    ln = listnet_loss(pred, true)
    od = ordinal_ranking_loss(pred, ranks)

    # DPO
    pc = torch.randn(B, requires_grad=True)
    pr = torch.randn(B, requires_grad=True)
    rc = torch.randn(B)
    rr = torch.randn(B)
    dp = dpo_pair_loss(pc, pr, rc, rr, beta=0.1)

    total = bt + mr + ln + od + dp
    total.backward()

    assert torch.isfinite(total)
    assert chosen.grad is not None and torch.isfinite(chosen.grad).all()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pc.grad is not None and torch.isfinite(pc.grad).all()
