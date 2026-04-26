"""Tests for the Adan optimizer."""

from __future__ import annotations

import torch

from src.optimizers.adan import Adan

# ------------------------------------------------------------------------------- #
# Helpers                                                                          #
# ------------------------------------------------------------------------------- #


def make_param(value: float = 3.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor(value))


def quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    """L = p^2, minimum at 0."""
    return (p**2).sum()


def run_steps(p, opt, n=20):
    for _ in range(n):
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        opt.step()
    return loss.item()


# ------------------------------------------------------------------------------- #
# Tests                                                                            #
# ------------------------------------------------------------------------------- #


def test_step_reduces_loss_on_quadratic():
    """Optimizer should reduce loss on a simple quadratic."""
    p = make_param(3.0)
    opt = Adan([p], lr=1e-2, weight_decay=0.0)
    initial_loss = quadratic_loss(p).item()
    run_steps(p, opt, n=50)
    final_loss = quadratic_loss(p).item()
    assert final_loss < initial_loss


def test_state_keys_exist():
    """After first step, expected state keys should be present."""
    p = make_param(2.0)
    opt = Adan([p], lr=1e-3)
    opt.zero_grad()
    quadratic_loss(p).backward()
    opt.step()
    state = opt.state[p]
    assert "step" in state
    assert "exp_avg" in state
    assert "exp_avg_sq" in state
    assert "exp_avg_diff" in state
    assert "prev_grad" in state


def test_lr_zero_no_update():
    """With lr=0 the parameter should not change."""
    p = make_param(5.0)
    opt = Adan([p], lr=0.0, weight_decay=0.0)
    initial_val = p.data.clone()
    opt.zero_grad()
    quadratic_loss(p).backward()
    opt.step()
    assert torch.allclose(p.data, initial_val)


def test_weight_decay_shrinks_params():
    """Weight decay should push parameter magnitude toward zero."""
    p = make_param(10.0)
    opt = Adan([p], lr=1e-2, weight_decay=0.5, betas=(0.98, 0.92, 0.99))
    # Zero out the gradient update by setting grad to zero
    for _ in range(30):
        opt.zero_grad()
        # provide a small constant gradient to keep optimizer active
        p.grad = torch.zeros_like(p)
        p.grad.fill_(0.01)
        opt.step()
    assert abs(p.item()) < 10.0


def test_no_prox_mode_works():
    """no_prox mode should also converge on quadratic."""
    p = make_param(3.0)
    opt = Adan([p], lr=1e-2, weight_decay=0.01, no_prox=True)
    run_steps(p, opt, n=50)
    assert quadratic_loss(p).item() < 9.0  # improved from initial 9.0


def test_step_count_increments():
    """State step counter should increment by 1 each call to opt.step()."""
    p = make_param(1.0)
    opt = Adan([p], lr=1e-3)
    for i in range(1, 6):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
        assert opt.state[p]["step"] == i


def test_different_betas():
    """Optimizer should still converge with non-default betas."""
    p = make_param(3.0)
    opt = Adan([p], lr=5e-3, betas=(0.9, 0.8, 0.95), weight_decay=0.0)
    run_steps(p, opt, n=60)
    assert quadratic_loss(p).item() < 9.0


def test_grad_none_skipped():
    """Parameters without gradient should be skipped without error."""
    p1 = make_param(2.0)
    p2 = make_param(3.0)
    opt = Adan([p1, p2], lr=1e-3)
    opt.zero_grad()
    # Only compute grad for p1
    quadratic_loss(p1).backward()
    # p2.grad remains None
    p2_before = p2.data.clone()
    opt.step()
    assert torch.allclose(p2.data, p2_before)


def test_closure_support():
    """Optimizer should correctly use closure to re-evaluate loss."""
    p = make_param(2.0)
    opt = Adan([p], lr=1e-2, weight_decay=0.0)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert loss.item() > 0.0


def test_param_group_isolation():
    """Different param groups should use their own lr."""
    p1 = make_param(3.0)
    p2 = make_param(3.0)
    opt = Adan(
        [{"params": [p1], "lr": 1e-1}, {"params": [p2], "lr": 1e-5}],
        lr=1e-3,
        weight_decay=0.0,
    )
    run_steps(p1, opt, n=1)
    run_steps(p2, opt, n=1)
    # p1 should move more than p2 given much higher lr
    for _ in range(20):
        opt.zero_grad()
        (quadratic_loss(p1) + quadratic_loss(p2)).backward()
        opt.step()
    assert abs(p1.item()) < abs(p2.item())
