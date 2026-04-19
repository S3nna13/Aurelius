"""Unit tests for KTO v2 loss."""

from __future__ import annotations

import pytest
import torch

from src.alignment.kto_v2 import KTOv2Loss, kto_v2_loss_functional


def _mk(batch=4, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    policy = torch.randn(batch, generator=g, device=device, requires_grad=True)
    ref = torch.randn(batch, generator=g, device=device)
    is_des = torch.tensor([True, False, True, False][:batch], device=device)
    return policy, ref, is_des


def test_forward_returns_scalar_finite():
    loss_fn = KTOv2Loss()
    p, r, d = _mk()
    out = loss_fn(p, r, d)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_gradient_flows_to_policy():
    loss_fn = KTOv2Loss()
    p, r, d = _mk()
    out = loss_fn(p, r, d)
    out.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert (p.grad.abs() > 0).any()


def test_no_grad_flows_through_z_ref():
    loss_fn = KTOv2Loss(z_ref_ema=0.5)
    # Run once so z_ref has been seeded from a prior batch
    p0, r0, d0 = _mk(seed=1)
    _ = loss_fn(p0.detach().requires_grad_(True), r0, d0)

    assert not loss_fn.z_ref.requires_grad
    z_before = loss_fn.z_ref.clone()

    p, r, d = _mk(seed=2)
    loss = loss_fn(p, r, d)
    loss.backward()
    # z_ref was updated (not the same as before) and is still detached.
    assert not loss_fn.z_ref.requires_grad
    assert loss_fn.z_ref.grad is None
    # p.grad must not depend on z_ref's autograd graph — just assert it's finite.
    assert torch.isfinite(p.grad).all()
    # sanity: z_ref changed from prior value (EMA kicked in)
    assert not torch.equal(z_before, loss_fn.z_ref)


def test_preferred_vs_unpreferred_loss_ordering():
    """Policy that strongly prefers desirable and unfavours undesirable
    should yield lower loss than the reversed arrangement."""
    ref = torch.zeros(4)
    is_des = torch.tensor([True, True, False, False])

    good_policy = torch.tensor([5.0, 5.0, -5.0, -5.0])
    bad_policy = torch.tensor([-5.0, -5.0, 5.0, 5.0])

    loss_good = kto_v2_loss_functional(good_policy, ref, is_des, 0.5, 0.5, 0.0)
    loss_bad = kto_v2_loss_functional(bad_policy, ref, is_des, 0.5, 0.5, 0.0)

    assert loss_good.item() < loss_bad.item()
    # Good arrangement should be small; bad should be large.
    assert loss_good.item() < 0.1
    assert loss_bad.item() > 1.0


def test_all_desirable_batch():
    loss_fn = KTOv2Loss()
    p = torch.randn(8, requires_grad=True)
    r = torch.randn(8)
    d = torch.ones(8, dtype=torch.bool)
    out = loss_fn(p, r, d)
    out.backward()
    assert torch.isfinite(out)
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_all_undesirable_batch():
    loss_fn = KTOv2Loss()
    p = torch.randn(8, requires_grad=True)
    r = torch.randn(8)
    d = torch.zeros(8, dtype=torch.bool)
    out = loss_fn(p, r, d)
    out.backward()
    assert torch.isfinite(out)
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_z_ref_updates_via_ema():
    loss_fn = KTOv2Loss(z_ref_ema=0.9)
    assert loss_fn.z_ref.item() == 0.0

    p1 = torch.tensor([2.0, 2.0, 2.0, 2.0], requires_grad=True)
    r1 = torch.zeros(4)
    d1 = torch.tensor([True, False, True, False])
    _ = loss_fn(p1, r1, d1)
    # First batch: cold-start copy -> z_ref == mean log-ratio == 2.0
    assert loss_fn.z_ref.item() == pytest.approx(2.0, abs=1e-6)

    p2 = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
    r2 = torch.zeros(4)
    d2 = torch.tensor([True, False, True, False])
    _ = loss_fn(p2, r2, d2)
    # EMA: 0.9 * 2.0 + 0.1 * 0.0 == 1.8
    assert loss_fn.z_ref.item() == pytest.approx(1.8, abs=1e-6)


def test_reset_z_ref_clears_to_zero():
    loss_fn = KTOv2Loss()
    p, r, d = _mk()
    _ = loss_fn(p, r, d)
    assert loss_fn._z_ref_initialized.item() is True or bool(loss_fn._z_ref_initialized)
    loss_fn.reset_z_ref()
    assert loss_fn.z_ref.item() == 0.0
    assert not bool(loss_fn._z_ref_initialized)


def test_asymmetric_betas_differ():
    # Asymmetric content: desirable samples have log-ratio 2.0, undesirable -0.5.
    p = torch.tensor([2.0, 2.0, -0.5, -0.5])
    r = torch.zeros(4)
    d = torch.tensor([True, True, False, False])
    l1 = kto_v2_loss_functional(p, r, d, beta_d=0.1, beta_u=0.5, z_ref=0.0)
    l2 = kto_v2_loss_functional(p, r, d, beta_d=0.5, beta_u=0.1, z_ref=0.0)
    assert not torch.allclose(l1, l2)


def test_shape_mismatch_raises():
    loss_fn = KTOv2Loss()
    p = torch.randn(4)
    with pytest.raises(ValueError):
        loss_fn(p, torch.randn(5), torch.tensor([True, False, True, False]))
    with pytest.raises(ValueError):
        loss_fn(p, torch.randn(4), torch.tensor([True, False, True]))
    with pytest.raises(ValueError):
        # non-bool is_desirable
        loss_fn(p, torch.randn(4), torch.tensor([1, 0, 1, 0]))
    with pytest.raises(ValueError):
        # non-1D policy
        loss_fn(torch.randn(2, 2), torch.randn(2, 2), torch.ones(2, 2, dtype=torch.bool))


def test_determinism_with_seeded_input():
    torch.manual_seed(42)
    a = KTOv2Loss(beta_desirable=0.2, beta_undesirable=0.3, z_ref_ema=0.8)
    torch.manual_seed(42)
    b = KTOv2Loss(beta_desirable=0.2, beta_undesirable=0.3, z_ref_ema=0.8)

    g = torch.Generator().manual_seed(7)
    p = torch.randn(16, generator=g)
    r = torch.randn(16, generator=torch.Generator().manual_seed(8))
    d = torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(9)).bool()

    la = a(p.clone().requires_grad_(True), r, d)
    lb = b(p.clone().requires_grad_(True), r, d)
    assert torch.equal(la.detach(), lb.detach())
    assert torch.equal(a.z_ref, b.z_ref)


def test_lambda_d_zero_zeros_desirable_contribution():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    r = torch.zeros(4)
    d_all = torch.ones(4, dtype=torch.bool)
    loss = kto_v2_loss_functional(p, r, d_all, beta_d=0.3, beta_u=0.3, z_ref=0.0, lambda_d=0.0, lambda_u=1.0)
    assert loss.item() == 0.0


def test_lambda_u_zero_zeros_undesirable_contribution():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    r = torch.zeros(4)
    d_none = torch.zeros(4, dtype=torch.bool)
    loss = kto_v2_loss_functional(p, r, d_none, beta_d=0.3, beta_u=0.3, z_ref=0.0, lambda_d=1.0, lambda_u=0.0)
    assert loss.item() == 0.0


def test_nan_safe_on_extreme_logprobs():
    loss_fn = KTOv2Loss()
    # Very large magnitudes; softplus should remain finite.
    p = torch.tensor([50.0, -50.0, 100.0, -100.0], requires_grad=True)
    r = torch.tensor([-50.0, 50.0, -100.0, 100.0])
    d = torch.tensor([True, False, True, False])
    out = loss_fn(p, r, d)
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(p.grad).all()


def test_batch_one_degenerate():
    loss_fn = KTOv2Loss()
    p = torch.tensor([0.5], requires_grad=True)
    r = torch.tensor([0.1])
    d = torch.tensor([True])
    out = loss_fn(p, r, d)
    assert out.dim() == 0
    assert torch.isfinite(out)
    out.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()

    # And the undesirable-only singleton
    p2 = torch.tensor([0.5], requires_grad=True)
    d2 = torch.tensor([False])
    out2 = loss_fn(p2, r, d2)
    assert torch.isfinite(out2)
