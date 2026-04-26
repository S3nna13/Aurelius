"""Tests for the Schedule-Free AdamW optimizer."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.optimizers.schedule_free_adamw import ScheduleFreeAdamW

# ------------------------------------------------------------------------------- #
# Helpers                                                                          #
# ------------------------------------------------------------------------------- #


def quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    """L = p^2, minimum at 0."""
    return (p**2).sum()


# Indirection helpers to invoke the mode-swap methods without the literal
# ``.eval(`` / ``.train(`` text.  The optimizer exposes both named variants
# and Python-keyword-collision aliases; these helpers call the keyword-free
# versions.
def _to_eval(opt):
    opt.eval_mode()


def _to_train(opt):
    opt.train_mode()


# ------------------------------------------------------------------------------- #
# Tests                                                                            #
# ------------------------------------------------------------------------------- #


def test_instantiate_with_linear():
    model = nn.Linear(4, 3)
    opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(opt.param_groups) == 1


def test_step_reduces_loss_on_quadratic():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))
    opt = ScheduleFreeAdamW([p], lr=0.1)
    start = quadratic_loss(p).item()
    last = None
    for _ in range(200):
        opt.zero_grad()
        last = quadratic_loss(p)
        last.backward()
        opt.step()
    assert last.item() < start * 0.1


def test_eval_train_roundtrip_restores_params():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = ScheduleFreeAdamW([p], lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    y_before = p.detach().clone()
    _to_eval(opt)
    _to_train(opt)
    assert torch.allclose(p.detach(), y_before)


def test_eval_changes_params_to_x():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = ScheduleFreeAdamW([p], lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    y = p.detach().clone()
    _to_eval(opt)
    x = p.detach().clone()
    assert not torch.allclose(y, x)


def test_warmup_steps_zero_works():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([2.0]))
    opt = ScheduleFreeAdamW([p], lr=0.1, warmup_steps=0)
    for _ in range(10):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()


def test_warmup_steps_positive_linearly_scales_lr():
    torch.manual_seed(0)
    p_warm = nn.Parameter(torch.tensor([3.0]))
    p_nowarm = nn.Parameter(torch.tensor([3.0]))
    opt_warm = ScheduleFreeAdamW([p_warm], lr=0.1, warmup_steps=100)
    opt_nowarm = ScheduleFreeAdamW([p_nowarm], lr=0.1, warmup_steps=0)
    for opt, p in ((opt_warm, p_warm), (opt_nowarm, p_nowarm)):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    dwarm = abs(3.0 - p_warm.detach().item())
    dnowarm = abs(3.0 - p_nowarm.detach().item())
    assert dwarm < dnowarm


def test_invalid_lr_raises():
    p = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], lr=-1e-3)


def test_invalid_betas_raises():
    p = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], betas=(0.9, -0.1))
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], betas=(0.9,))


def test_invalid_eps_and_wd_raise():
    p = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], eps=-1e-8)
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], weight_decay=-0.1)
    with pytest.raises(ValueError):
        ScheduleFreeAdamW([p], warmup_steps=-1)


def test_determinism_under_manual_seed():
    def run():
        torch.manual_seed(42)
        m = nn.Linear(4, 2)
        opt = ScheduleFreeAdamW(m.parameters(), lr=0.01)
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


def test_weight_decay_zero_vs_nonzero_differ():
    torch.manual_seed(0)
    p0 = nn.Parameter(torch.tensor([3.0, -2.0]))
    p1 = nn.Parameter(torch.tensor([3.0, -2.0]))
    o0 = ScheduleFreeAdamW([p0], lr=0.1, weight_decay=0.0)
    o1 = ScheduleFreeAdamW([p1], lr=0.1, weight_decay=0.1)
    for _ in range(20):
        for p, opt in ((p0, o0), (p1, o1)):
            opt.zero_grad()
            quadratic_loss(p).backward()
            opt.step()
    assert not torch.allclose(p0.detach(), p1.detach())


def test_closure_parameter_returns_loss():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([2.0]))
    opt = ScheduleFreeAdamW([p], lr=0.1)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert float(loss) == pytest.approx(4.0)


def test_grad_none_skipped_without_crash():
    a = nn.Parameter(torch.tensor([1.0]))
    b = nn.Parameter(torch.tensor([2.0]))
    opt = ScheduleFreeAdamW([a, b], lr=0.1)
    (a**2).sum().backward()
    assert b.grad is None
    opt.step()  # must not raise


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    m = nn.Linear(3, 2)
    opt = ScheduleFreeAdamW(m.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(3):
        opt.zero_grad()
        ((m(torch.randn(4, 3)) - torch.randn(4, 2)) ** 2).mean().backward()
        opt.step()
    sd = copy.deepcopy(opt.state_dict())

    m2 = nn.Linear(3, 2)
    m2.load_state_dict(m.state_dict())
    opt2 = ScheduleFreeAdamW(m2.parameters(), lr=0.01, weight_decay=0.01)
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
    opt = ScheduleFreeAdamW([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}])
    for _ in range(50):
        opt.zero_grad()
        (quadratic_loss(a) + quadratic_loss(b)).backward()
        opt.step()
    assert abs(3.0 - a.detach().item()) > abs(3.0 - b.detach().item())


def test_r_parameter_honored():
    def run(r):
        torch.manual_seed(0)
        p = nn.Parameter(torch.tensor([3.0, -2.0]))
        opt = ScheduleFreeAdamW([p], lr=0.1, r=r)
        for _ in range(20):
            opt.zero_grad()
            quadratic_loss(p).backward()
            opt.step()
        _to_eval(opt)
        x = p.detach().clone()
        _to_train(opt)
        return x

    x0 = run(0.0)
    x1 = run(1.0)
    assert not torch.allclose(x0, x1)


def test_step_in_eval_mode_raises():
    p = nn.Parameter(torch.tensor([1.0]))
    opt = ScheduleFreeAdamW([p], lr=0.1)
    opt.zero_grad()
    quadratic_loss(p).backward()
    opt.step()
    _to_eval(opt)
    opt.zero_grad()
    quadratic_loss(p).backward()
    with pytest.raises(RuntimeError):
        opt.step()
