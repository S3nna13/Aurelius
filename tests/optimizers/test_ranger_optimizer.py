"""Tests for the Ranger optimizer (RAdam + LookAhead)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.optimizers.ranger_optimizer import Ranger


def quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    return (p**2).sum()


def test_instantiate_with_linear():
    model = nn.Linear(4, 3)
    opt = Ranger(model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer) or hasattr(opt, "step")
    assert len(opt.param_groups) == 1


def test_step_reduces_loss_on_quadratic():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))
    opt = Ranger([p], lr=0.1)
    start = quadratic_loss(p).item()
    last = None
    for _ in range(200):
        opt.zero_grad()
        last = quadratic_loss(p)
        last.backward()
        opt.step()
    assert last.item() < start * 0.1


def test_slow_weights_differ_from_fast_before_sync():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = Ranger([p], lr=0.1, k=5, alpha=0.5)
    # Step 1-3: before the first k=5 sync, slow should be initial values
    for _ in range(3):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    slow = opt.slow_params
    fast = p.detach()
    # Fast weights moved from initialization; slow weights still hold initial values
    assert not torch.allclose(slow[0], fast)
    # After the sync at step 5, slow catches up
    for _ in range(2):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    slow = opt.slow_params
    fast = p.detach()
    # After sync at step 5, slow == fast (they were just interpolated and fast was copied)
    assert torch.allclose(slow[0], fast)


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    m = nn.Linear(3, 2)
    opt = Ranger(m.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(3):
        opt.zero_grad()
        ((m(torch.randn(4, 3)) - torch.randn(4, 2)) ** 2).mean().backward()
        opt.step()
    sd = copy.deepcopy(opt.state_dict())

    m2 = nn.Linear(3, 2)
    m2.load_state_dict(m.state_dict())
    opt2 = Ranger(m2.parameters(), lr=0.01, weight_decay=0.01)
    opt2.load_state_dict(sd)

    torch.manual_seed(123)
    xin = torch.randn(4, 3)
    yin = torch.randn(4, 2)
    for opt_, mod in ((opt, m), (opt2, m2)):
        opt_.zero_grad()
        ((mod(xin) - yin) ** 2).mean().backward()
        opt_.step()
    for pa, pb in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(pa, pb)


def test_multiple_param_groups():
    a = nn.Parameter(torch.tensor([3.0]))
    b = nn.Parameter(torch.tensor([3.0]))
    opt = Ranger([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}])
    for _ in range(50):
        opt.zero_grad()
        (quadratic_loss(a) + quadratic_loss(b)).backward()
        opt.step()
    assert abs(3.0 - a.detach().item()) > abs(3.0 - b.detach().item())


def test_grad_none_skipped_without_crash():
    a = nn.Parameter(torch.tensor([1.0]))
    b = nn.Parameter(torch.tensor([2.0]))
    opt = Ranger([a, b], lr=0.1)
    (a**2).sum().backward()
    assert b.grad is None
    opt.step()


def test_determinism_under_manual_seed():
    def run():
        torch.manual_seed(42)
        m = nn.Linear(4, 2)
        opt = Ranger(m.parameters(), lr=0.01)
        x = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
        y = torch.randn(8, 2, generator=torch.Generator().manual_seed(1))
        for _ in range(10):
            opt.zero_grad()
            loss = ((m(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    a = run()
    b = run()
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb)


def test_repr():
    opt = Ranger([nn.Parameter(torch.tensor([1.0]))], lr=0.01)
    r = repr(opt)
    assert "Ranger" in r
    assert "RAdam" in r
    assert "LookAhead" in r


def test_zero_grad_sets_grad_to_none():
    p = nn.Parameter(torch.tensor([2.0]))
    opt = Ranger([p], lr=0.1)
    quadratic_loss(p).backward()
    assert p.grad is not None
    opt.zero_grad(set_to_none=True)
    assert p.grad is None


def test_closure_parameter_returns_loss():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([2.0]))
    opt = Ranger([p], lr=0.1)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert float(loss) == pytest.approx(4.0)
