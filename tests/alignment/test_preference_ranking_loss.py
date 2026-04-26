"""Tests for src.alignment.preference_ranking_loss."""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn.functional as F

from src.alignment.preference_ranking_loss import (
    bradley_terry_loss,
    dpo_pair_loss,
    listnet_loss,
    margin_ranking_loss,
    ordinal_ranking_loss,
)


def test_bradley_terry_zero_when_chosen_dominates():
    chosen = torch.tensor([50.0, 60.0, 70.0])
    rejected = torch.tensor([-50.0, -60.0, -70.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.item() < 1e-10


def test_bradley_terry_equals_log2_when_equal():
    chosen = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rejected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert math.isclose(loss.item(), math.log(2.0), abs_tol=1e-6)


def test_bradley_terry_gradient_flows():
    chosen = torch.randn(4, requires_grad=True)
    rejected = torch.randn(4, requires_grad=True)
    loss = bradley_terry_loss(chosen, rejected)
    loss.backward()
    assert chosen.grad is not None and torch.isfinite(chosen.grad).all()
    assert rejected.grad is not None and torch.isfinite(rejected.grad).all()
    assert (chosen.grad != 0).any()


def test_margin_ranking_respects_margin():
    chosen = torch.tensor([2.0, 2.0, 2.0])
    rejected = torch.tensor([0.0, 1.0, 5.0])  # diffs = 2, 1, -3
    # margin=1: losses = max(0, 1-2)=0, max(0,1-1)=0, max(0,1-(-3))=4 -> mean 4/3
    loss = margin_ranking_loss(chosen, rejected, margin=1.0)
    assert math.isclose(loss.item(), 4.0 / 3.0, abs_tol=1e-6)
    # margin=0: only negative diff contributes => max(0,0-(-3))=3 -> mean 1.0
    loss2 = margin_ranking_loss(chosen, rejected, margin=0.0)
    assert math.isclose(loss2.item(), 1.0, abs_tol=1e-6)
    # large margin: all contribute
    loss3 = margin_ranking_loss(chosen, rejected, margin=10.0)
    expected = ((10 - 2) + (10 - 1) + (10 - (-3))) / 3.0
    assert math.isclose(loss3.item(), expected, abs_tol=1e-6)


def test_listnet_hand_computed():
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    true = torch.tensor([[3.0, 2.0, 1.0]])
    # true prob
    tp = torch.softmax(true, dim=-1)
    lp = torch.log_softmax(pred, dim=-1)
    expected = -(tp * lp).sum(dim=-1).mean().item()
    got = listnet_loss(pred, true).item()
    assert math.isclose(got, expected, abs_tol=1e-5)


def test_listnet_identical_scores_equals_entropy():
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss = listnet_loss(scores, scores).item()
    p = torch.softmax(scores, dim=-1)
    expected = -(p * torch.log(p)).sum().item()
    assert math.isclose(loss, expected, abs_tol=1e-5)


def test_ordinal_ranking_loss_correctness():
    # Perfect ordering: predicted decreasing matches ranks increasing
    pred = torch.tensor([[3.0, 2.0, 1.0]])
    ranks = torch.tensor([[0, 1, 2]])
    # i=0 preferred over 1: softplus(-(3-2))=softplus(-1)
    # i=0 over 2: softplus(-(3-1))=softplus(-2)
    # i=1 over 2: softplus(-(2-1))=softplus(-1)
    expected = (F.softplus(torch.tensor(-1.0)) * 2 + F.softplus(torch.tensor(-2.0))) / 3.0
    got = ordinal_ranking_loss(pred, ranks)
    assert math.isclose(got.item(), expected.item(), abs_tol=1e-6)

    # Worst ordering: predicted flipped
    pred_bad = torch.tensor([[1.0, 2.0, 3.0]])
    loss_bad = ordinal_ranking_loss(pred_bad, ranks)
    assert loss_bad.item() > got.item()


def test_dpo_zero_when_policy_perfectly_prefers_chosen():
    # chosen_logratio - rejected_logratio very large
    policy_chosen = torch.tensor([100.0])
    policy_rejected = torch.tensor([-100.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss = dpo_pair_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)
    assert loss.item() < 1e-10


def test_dpo_gradient_flows_on_all_inputs():
    pc = torch.randn(4, requires_grad=True)
    pr = torch.randn(4, requires_grad=True)
    rc = torch.randn(4, requires_grad=True)
    rr = torch.randn(4, requires_grad=True)
    loss = dpo_pair_loss(pc, pr, rc, rr, beta=0.5)
    loss.backward()
    for name, t in [("pc", pc), ("pr", pr), ("rc", rc), ("rr", rr)]:
        assert t.grad is not None, f"{name} grad missing"
        assert torch.isfinite(t.grad).all(), f"{name} grad not finite"
        assert (t.grad != 0).any(), f"{name} grad all zero"


def test_dpo_equals_log2_when_beta_zero():
    pc = torch.randn(8)
    pr = torch.randn(8)
    rc = torch.randn(8)
    rr = torch.randn(8)
    loss = dpo_pair_loss(pc, pr, rc, rr, beta=0.0)
    assert math.isclose(loss.item(), math.log(2.0), abs_tol=1e-6)


def test_shape_mismatches_raise():
    with pytest.raises(ValueError):
        bradley_terry_loss(torch.randn(4), torch.randn(5))
    with pytest.raises(ValueError):
        margin_ranking_loss(torch.randn(4), torch.randn(5))
    with pytest.raises(ValueError):
        listnet_loss(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(ValueError):
        listnet_loss(torch.randn(3), torch.randn(3))  # not 2D
    with pytest.raises(ValueError):
        ordinal_ranking_loss(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(ValueError):
        dpo_pair_loss(torch.randn(4), torch.randn(5), torch.randn(4), torch.randn(4))


def test_nan_safe_large_differences():
    big = torch.tensor([1e6, -1e6, 1e6])
    small = torch.tensor([-1e6, 1e6, -1e6])
    # BT: one giant positive diff, one giant negative -> softplus handles it
    loss = bradley_terry_loss(big, small)
    assert torch.isfinite(loss)
    # DPO with big logratios
    loss2 = dpo_pair_loss(big, small, torch.zeros(3), torch.zeros(3), beta=1.0)
    assert torch.isfinite(loss2)
    # margin
    loss3 = margin_ranking_loss(big, small, margin=1.0)
    assert torch.isfinite(loss3)


def test_determinism():
    torch.manual_seed(0)
    a = torch.randn(16, requires_grad=False)
    b = torch.randn(16, requires_grad=False)
    l1 = bradley_terry_loss(a, b).item()
    l2 = bradley_terry_loss(a, b).item()
    assert l1 == l2

    pred = torch.randn(4, 5)
    true = torch.randn(4, 5)
    assert listnet_loss(pred, true).item() == listnet_loss(pred, true).item()

    ranks = torch.tensor([[0, 1, 2, 3, 4]] * 4)
    assert ordinal_ranking_loss(pred, ranks).item() == ordinal_ranking_loss(pred, ranks).item()


def test_all_losses_scalar_and_finite():
    torch.manual_seed(7)
    c = torch.randn(8)
    r = torch.randn(8)
    l1 = bradley_terry_loss(c, r)
    l2 = margin_ranking_loss(c, r, margin=0.5)
    pred = torch.randn(8, 5)
    true = torch.randn(8, 5)
    ranks = torch.stack([torch.randperm(5) for _ in range(8)])
    l3 = listnet_loss(pred, true)
    l4 = ordinal_ranking_loss(pred, ranks)
    l5 = dpo_pair_loss(torch.randn(8), torch.randn(8), torch.randn(8), torch.randn(8), beta=0.1)
    for l in (l1, l2, l3, l4, l5):  # noqa: E741
        assert l.dim() == 0  # noqa: E741
        assert torch.isfinite(l)  # noqa: E741


def test_large_batch_runtime_under_one_second():
    torch.manual_seed(0)
    B = 128
    c = torch.randn(B)
    r = torch.randn(B)
    pred = torch.randn(B, 16)
    true = torch.randn(B, 16)
    ranks = torch.stack([torch.randperm(16) for _ in range(B)])
    pc = torch.randn(B)
    pr = torch.randn(B)
    rc = torch.randn(B)
    rr = torch.randn(B)
    t0 = time.perf_counter()
    bradley_terry_loss(c, r)
    margin_ranking_loss(c, r)
    listnet_loss(pred, true)
    ordinal_ranking_loss(pred, ranks)
    dpo_pair_loss(pc, pr, rc, rr)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"took {elapsed:.3f}s"
