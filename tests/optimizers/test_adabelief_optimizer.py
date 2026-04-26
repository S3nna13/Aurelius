"""Tests for the AdaBelief optimizer."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.optimizers.adabelief_optimizer import AdaBelief


def quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    return (p**2).sum()


def test_instantiate_with_linear():
    model = nn.Linear(4, 3)
    opt = AdaBelief(model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(opt.param_groups) == 1


def test_step_reduces_loss_on_quadratic():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))
    opt = AdaBelief([p], lr=0.1)
    start = quadratic_loss(p).item()
    last = None
    for _ in range(200):
        opt.zero_grad()
        last = quadratic_loss(p)
        last.backward()
        opt.step()
    assert last.item() < start * 0.1


def test_weight_decay_zero_vs_nonzero_differ():
    torch.manual_seed(0)
    p0 = nn.Parameter(torch.tensor([3.0, -2.0]))
    p1 = nn.Parameter(torch.tensor([3.0, -2.0]))
    o0 = AdaBelief([p0], lr=0.1, weight_decay=0.0)
    o1 = AdaBelief([p1], lr=0.1, weight_decay=0.1)
    for _ in range(20):
        for p, opt in ((p0, o0), (p1, o1)):
            opt.zero_grad()
            quadratic_loss(p).backward()
            opt.step()
    assert not torch.allclose(p0.detach(), p1.detach())


def test_fixed_decay_vs_decoupled_differ():
    torch.manual_seed(0)
    p0 = nn.Parameter(torch.tensor([3.0, -2.0]))
    p1 = nn.Parameter(torch.tensor([3.0, -2.0]))
    o0 = AdaBelief([p0], lr=0.1, weight_decay=0.1, fixed_decay=False)
    o1 = AdaBelief([p1], lr=0.1, weight_decay=0.1, fixed_decay=True)
    for _ in range(20):
        for p, opt in ((p0, o0), (p1, o1)):
            opt.zero_grad()
            quadratic_loss(p).backward()
            opt.step()
    assert not torch.allclose(p0.detach(), p1.detach())


def test_amsgrad_variant_works():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = AdaBelief([p], lr=0.1, amsgrad=True)
    for _ in range(20):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    m = nn.Linear(3, 2)
    opt = AdaBelief(m.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(3):
        opt.zero_grad()
        ((m(torch.randn(4, 3)) - torch.randn(4, 2)) ** 2).mean().backward()
        opt.step()
    sd = copy.deepcopy(opt.state_dict())

    m2 = nn.Linear(3, 2)
    m2.load_state_dict(m.state_dict())
    opt2 = AdaBelief(m2.parameters(), lr=0.01, weight_decay=0.01)
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


def test_invalid_lr_raises():
    p = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        AdaBelief([p], lr=-1e-3)


def test_invalid_betas_raises():
    p = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        AdaBelief([p], betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        AdaBelief([p], betas=(0.9, -0.1))
    with pytest.raises(ValueError):
        AdaBelief([p], betas=(0.9,))


def test_invalid_eps_and_wd_raise():
    p = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        AdaBelief([p], eps=-1e-8)
    with pytest.raises(ValueError):
        AdaBelief([p], weight_decay=-0.1)


def test_determinism_under_manual_seed():
    def run():
        torch.manual_seed(42)
        m = nn.Linear(4, 2)
        opt = AdaBelief(m.parameters(), lr=0.01)
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


def test_grad_none_skipped_without_crash():
    a = nn.Parameter(torch.tensor([1.0]))
    b = nn.Parameter(torch.tensor([2.0]))
    opt = AdaBelief([a, b], lr=0.1)
    (a**2).sum().backward()
    assert b.grad is None
    opt.step()


def test_closure_parameter_returns_loss():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([2.0]))
    opt = AdaBelief([p], lr=0.1)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert float(loss) == pytest.approx(4.0)


def test_sparse_gradient_raises():
    p = nn.Parameter(torch.zeros(4))
    opt = AdaBelief([p], lr=0.1)
    indices = torch.tensor([[0, 2]])  # shape (sparse_dim, nnz) = (1, 2)
    values = torch.tensor([1.0, 2.0])
    grad = torch.sparse_coo_tensor(indices, values, size=(4,))
    p.grad = grad
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


def test_multiple_param_groups():
    a = nn.Parameter(torch.tensor([3.0]))
    b = nn.Parameter(torch.tensor([3.0]))
    opt = AdaBelief([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}])
    for _ in range(50):
        opt.zero_grad()
        (quadratic_loss(a) + quadratic_loss(b)).backward()
        opt.step()
    assert abs(3.0 - a.detach().item()) > abs(3.0 - b.detach().item())
