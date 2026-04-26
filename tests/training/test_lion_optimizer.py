"""Tests for the Lion optimizer (EvoLved Sign Momentum)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.lion_optimizer import Lion

# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_construction_defaults():
    p = torch.zeros(3, requires_grad=True)
    opt = Lion([p])
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(1e-4)
    assert group["betas"] == (0.9, 0.99)
    assert group["weight_decay"] == 0.0


def test_construction_custom_args():
    p = torch.zeros(3, requires_grad=True)
    opt = Lion([p], lr=3e-4, betas=(0.95, 0.98), weight_decay=0.1)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["betas"] == (0.95, 0.98)
    assert group["weight_decay"] == pytest.approx(0.1)


def test_rejects_invalid_lr():
    p = torch.zeros(3, requires_grad=True)
    with pytest.raises(ValueError):
        Lion([p], lr=-0.1)


def test_rejects_invalid_betas():
    p = torch.zeros(3, requires_grad=True)
    with pytest.raises(ValueError):
        Lion([p], betas=(-0.1, 0.99))
    with pytest.raises(ValueError):
        Lion([p], betas=(0.9, 1.0))
    with pytest.raises(ValueError):
        Lion([p], betas=(1.0, 0.99))


def test_rejects_negative_weight_decay():
    p = torch.zeros(3, requires_grad=True)
    with pytest.raises(ValueError):
        Lion([p], weight_decay=-1e-3)


# ---------------------------------------------------------------------------
# Basic step behavior
# ---------------------------------------------------------------------------


def test_step_runs_without_error():
    p = torch.randn(4, requires_grad=True)
    p.grad = torch.randn_like(p)
    opt = Lion([p], lr=1e-3)
    opt.step()


def test_parameters_update_after_step():
    torch.manual_seed(0)
    p = torch.randn(4, requires_grad=True)
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt = Lion([p], lr=1e-2)
    opt.step()
    # At least one entry must have changed.
    assert not torch.allclose(p.detach(), p_before)


def test_update_direction_matches_sign_of_blended_gradient():
    """Core invariant: the step direction equals -lr * sign(c_t) (wd=0)."""
    torch.manual_seed(1)
    p = torch.randn(6, requires_grad=True)
    p_before = p.detach().clone()

    # On the first step exp_avg starts at 0, so c_t = (1-beta1)*g => sign(c_t)=sign(g).
    g = torch.tensor([1.0, -2.0, 0.5, -0.3, 3.0, -0.01])
    p.grad = g.clone()

    lr = 1e-2
    opt = Lion([p], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    opt.step()

    expected_delta = -lr * torch.sign(g)
    actual_delta = p.detach() - p_before
    assert torch.allclose(actual_delta, expected_delta, atol=1e-7)


def test_weight_decay_shrinks_params_with_zero_grad():
    p = torch.full((4,), 2.0, requires_grad=True)
    p_before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    lr = 0.1
    wd = 0.2
    opt = Lion([p], lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    opt.step()
    # When grad=0 on first step, sign(c_t)=sign(0)=0, so update = -lr*wd*theta.
    expected = p_before * (1 - lr * wd)
    assert torch.allclose(p.detach(), expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Parameter groups / state
# ---------------------------------------------------------------------------


def test_multiple_param_groups_with_different_lr():
    p1 = torch.tensor([1.0, 1.0], requires_grad=True)
    p2 = torch.tensor([1.0, 1.0], requires_grad=True)
    p1.grad = torch.tensor([1.0, 1.0])
    p2.grad = torch.tensor([1.0, 1.0])

    lr1, lr2 = 1e-2, 1e-3
    opt = Lion(
        [
            {"params": [p1], "lr": lr1},
            {"params": [p2], "lr": lr2},
        ]
    )
    before1 = p1.detach().clone()
    before2 = p2.detach().clone()
    opt.step()

    delta1 = (before1 - p1.detach()).abs().max().item()
    delta2 = (before2 - p2.detach()).abs().max().item()
    assert delta1 == pytest.approx(lr1, rel=1e-4, abs=1e-6)
    assert delta2 == pytest.approx(lr2, rel=1e-4, abs=1e-6)


def test_state_contains_exp_avg_after_first_step():
    p = torch.randn(3, 5, requires_grad=True)
    p.grad = torch.randn_like(p)
    opt = Lion([p])
    opt.step()
    state = opt.state[p]
    assert "exp_avg" in state
    assert state["exp_avg"].shape == p.shape
    assert state["exp_avg"].dtype == p.dtype


# ---------------------------------------------------------------------------
# Exact value invariants
# ---------------------------------------------------------------------------


def test_zero_betas_gives_pure_sign_update():
    """With beta1=beta2=0 the update is exactly -lr*(sign(g) + wd*theta)."""
    theta0 = torch.tensor([1.0, -2.0, 0.5])
    p = theta0.clone().requires_grad_(True)
    g = torch.tensor([0.2, -3.0, 0.0])
    p.grad = g.clone()

    lr = 0.05
    wd = 0.1
    opt = Lion([p], lr=lr, betas=(0.0, 0.0), weight_decay=wd)
    opt.step()

    expected = theta0 - lr * (torch.sign(g) + wd * theta0)
    assert torch.allclose(p.detach(), expected, atol=1e-7)


def test_momentum_buffer_accumulates_correctly_over_two_steps():
    torch.manual_seed(7)
    p = torch.zeros(3, requires_grad=True)
    beta1, beta2 = 0.9, 0.99
    opt = Lion([p], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)

    g1 = torch.tensor([1.0, -2.0, 0.5])
    g2 = torch.tensor([0.3, 1.0, -0.4])

    # Step 1: exp_avg starts at 0 -> exp_avg becomes (1-beta2)*g1.
    p.grad = g1.clone()
    opt.step()
    expected_m1 = (1 - beta2) * g1
    assert torch.allclose(opt.state[p]["exp_avg"], expected_m1, atol=1e-7)

    # Step 2: exp_avg becomes beta2*m1 + (1-beta2)*g2.
    p.grad = g2.clone()
    opt.step()
    expected_m2 = beta2 * expected_m1 + (1 - beta2) * g2
    assert torch.allclose(opt.state[p]["exp_avg"], expected_m2, atol=1e-7)


# ---------------------------------------------------------------------------
# Integration / robustness
# ---------------------------------------------------------------------------


def test_trains_tiny_linear_model_and_loss_decreases():
    torch.manual_seed(42)
    model = nn.Linear(8, 1, bias=True)
    opt = Lion(model.parameters(), lr=1e-2, weight_decay=0.0)

    x = torch.randn(32, 8)
    target = x.sum(dim=1, keepdim=True)  # easy linear target

    def loss_fn():
        return ((model(x) - target) ** 2).mean()

    initial_loss = loss_fn().item()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    final_loss = loss_fn().item()
    assert final_loss < initial_loss * 0.5


def test_grad_none_skips_param_without_crash():
    p1 = torch.randn(3, requires_grad=True)
    p2 = torch.randn(3, requires_grad=True)
    p1.grad = torch.randn_like(p1)
    # p2.grad is None.
    p2_before = p2.detach().clone()

    opt = Lion([p1, p2], lr=1e-2)
    opt.step()  # Must not raise.

    # p2 unchanged; no state allocated for it.
    assert torch.equal(p2.detach(), p2_before)
    assert len(opt.state.get(p2, {})) == 0


def test_closure_return_value_is_returned_from_step():
    p = torch.zeros(2, requires_grad=True)
    p.grad = torch.ones_like(p)
    opt = Lion([p], lr=1e-3)

    sentinel = torch.tensor(3.14159)

    def closure():
        return sentinel

    out = opt.step(closure)
    assert out is sentinel


def test_deterministic_across_identical_seeds():
    def run():
        torch.manual_seed(123)
        model = nn.Linear(4, 2)
        opt = Lion(model.parameters(), lr=1e-3)
        x = torch.randn(5, 4, generator=torch.Generator().manual_seed(0))
        target = torch.randn(5, 2, generator=torch.Generator().manual_seed(1))
        for _ in range(5):
            opt.zero_grad()
            loss = ((model(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    a = run()
    b = run()
    assert torch.equal(a, b)
