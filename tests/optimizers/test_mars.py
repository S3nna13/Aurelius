"""Tests for the MARS optimizer."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.optimizers.mars import Mars


def quadratic_loss(p: torch.Tensor) -> torch.Tensor:
    return (p ** 2).sum()


def test_instantiate_with_linear():
    model = nn.Linear(4, 2)
    opt = Mars(model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(opt.param_groups) == 1


def test_step_reduces_loss_on_quadratic():
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -2.0, 1.5]))
    opt = Mars([p], lr=1e-2, weight_decay=0.0)
    initial = quadratic_loss(p).item()
    for _ in range(80):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    final = quadratic_loss(p).item()
    assert final < initial


def test_first_step_uses_plain_gradient():
    """First step: no prev_grad, variance reduction term is zero."""
    p = nn.Parameter(torch.tensor([2.0, -1.0]))
    opt = Mars([p], lr=1e-3, gamma=0.5, weight_decay=0.0)
    opt.zero_grad()
    quadratic_loss(p).backward()
    grad_before = p.grad.detach().clone()
    opt.step()
    state = opt.state[p]
    # With variance reduction disabled on first step, exp_avg = (1-b1)*grad
    b1 = opt.param_groups[0]["betas"][0]
    expected = (1.0 - b1) * grad_before
    assert torch.allclose(state["exp_avg"], expected, atol=1e-6)
    # prev_grad should now be populated
    assert state["prev_grad"] is not None
    assert torch.allclose(state["prev_grad"], grad_before)


def test_second_step_uses_variance_reduced_estimator():
    """Second step with gamma>0: exp_avg should differ from the gamma=0 path."""
    torch.manual_seed(1)
    # Run with gamma>0
    p1 = nn.Parameter(torch.tensor([2.5, -1.3, 0.7]))
    opt1 = Mars([p1], lr=1e-3, gamma=0.3, weight_decay=0.0)
    # Run with gamma=0
    p2 = nn.Parameter(torch.tensor([2.5, -1.3, 0.7]))
    opt2 = Mars([p2], lr=1e-3, gamma=0.0, weight_decay=0.0)

    for step_i in range(2):
        for p, opt in [(p1, opt1), (p2, opt2)]:
            opt.zero_grad()
            # Use a loss that changes gradient across steps so diff != 0
            loss = (p ** 2).sum() + (p ** 3).sum()
            loss.backward()
            opt.step()

    # After 2 steps, the exp_avgs must differ because gamma>0 perturbed g_tilde
    assert not torch.allclose(opt1.state[p1]["exp_avg"], opt2.state[p2]["exp_avg"])


def test_determinism_with_seed():
    def run():
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(5))
        opt = Mars([p], lr=1e-3)
        for _ in range(10):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        return p.detach().clone()

    a = run()
    b = run()
    assert torch.allclose(a, b)


def test_invalid_lr_raises():
    p = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        Mars([p], lr=-1e-3)


def test_invalid_betas_raises():
    p = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        Mars([p], betas=(1.1, 0.99))
    with pytest.raises(ValueError):
        Mars([p], betas=(0.9, -0.1))
    with pytest.raises(ValueError):
        Mars([p], betas=(0.9,))


def test_invalid_gamma_raises():
    p = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        Mars([p], gamma=-0.1)


def test_weight_decay_decoupled():
    """With zero gradients, weight decay should still shrink the params."""
    p = nn.Parameter(torch.tensor([5.0, -5.0]))
    opt = Mars([p], lr=0.1, weight_decay=0.1, gamma=0.0)
    initial = p.detach().clone()
    for _ in range(5):
        opt.zero_grad()
        p.grad = torch.zeros_like(p)
        opt.step()
    assert p.abs().sum().item() < initial.abs().sum().item()


def test_state_dict_roundtrip():
    import copy
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(3))
    opt = Mars([p], lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        (p ** 2).sum().backward()
        opt.step()
    sd = copy.deepcopy(opt.state_dict())

    p2 = nn.Parameter(p.detach().clone())
    opt2 = Mars([p2], lr=1e-3)
    opt2.load_state_dict(sd)

    # State should match
    st1 = opt.state[p]
    st2 = opt2.state[p2]
    assert st1["step"] == st2["step"]
    assert torch.equal(st1["exp_avg"], st2["exp_avg"])
    assert torch.equal(st1["exp_avg_sq"], st2["exp_avg_sq"])
    assert torch.equal(st1["prev_grad"], st2["prev_grad"])

    # Stepping each with the same fresh gradient should produce the same update.
    g = torch.tensor([0.5, -0.3, 0.9])
    opt.zero_grad(); p.grad = g.clone(); opt.step()
    opt2.zero_grad(); p2.grad = g.clone(); opt2.step()
    assert torch.allclose(p, p2, atol=1e-6, rtol=1e-6)


def test_closure_supported():
    p = nn.Parameter(torch.tensor([2.0]))
    opt = Mars([p], lr=1e-3, weight_decay=0.0)

    def closure():
        opt.zero_grad()
        loss = quadratic_loss(p)
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert loss.item() == pytest.approx(4.0)


def test_none_grad_skipped():
    p1 = nn.Parameter(torch.tensor([2.0]))
    p2 = nn.Parameter(torch.tensor([3.0]))
    opt = Mars([p1, p2], lr=1e-2)
    opt.zero_grad()
    quadratic_loss(p1).backward()
    before = p2.detach().clone()
    opt.step()
    assert torch.allclose(p2.detach(), before)
    # p2 should have no state populated
    assert len(opt.state[p2]) == 0


def test_gamma_zero_matches_adamw():
    """With gamma=0, MARS should match torch.optim.AdamW exactly."""
    torch.manual_seed(123)
    init = torch.randn(8)

    p_mars = nn.Parameter(init.clone())
    p_adam = nn.Parameter(init.clone())

    opt_mars = Mars([p_mars], lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                   weight_decay=0.05, gamma=0.0)
    opt_adam = torch.optim.AdamW([p_adam], lr=1e-2, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.05)

    for step_i in range(6):
        # Same grad source
        g = torch.randn(8, generator=torch.Generator().manual_seed(step_i))
        opt_mars.zero_grad()
        opt_adam.zero_grad()
        p_mars.grad = g.clone()
        p_adam.grad = g.clone()
        opt_mars.step()
        opt_adam.step()

    assert torch.allclose(p_mars, p_adam, atol=1e-4, rtol=1e-4)


def test_multiple_param_groups():
    p1 = nn.Parameter(torch.tensor([3.0]))
    p2 = nn.Parameter(torch.tensor([3.0]))
    opt = Mars(
        [{"params": [p1], "lr": 1e-1},
         {"params": [p2], "lr": 1e-5}],
        lr=1e-3,
        weight_decay=0.0,
    )
    for _ in range(20):
        opt.zero_grad()
        (quadratic_loss(p1) + quadratic_loss(p2)).backward()
        opt.step()
    # p1 with larger lr should be much closer to zero than p2
    assert abs(p1.item()) < abs(p2.item())


def test_invalid_eps_raises():
    p = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        Mars([p], eps=-1e-8)


def test_invalid_weight_decay_raises():
    p = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        Mars([p], weight_decay=-0.1)
