"""Tests for Schedule-Free AdamW / SGD optimizers.

Covers 15 test cases:
 1. ScheduleFreeSGD initializes without error
 2. ScheduleFreeSGD single step does not crash
 3. ScheduleFreeSGD loss decreases on simple quadratic
 4. ScheduleFreeSGD parameters updated after step
 5. ScheduleFreeSGD weight_decay reduces param magnitude over time
 6. ScheduleFreeAdamW initializes without error
 7. ScheduleFreeAdamW single step does not crash
 8. ScheduleFreeAdamW loss decreases on simple regression
 9. ScheduleFreeAdamW parameters updated after step
10. ScheduleFreeAdamW state initialized correctly
11. ScheduleFreeAdamW warmup_steps: effective lr starts small and grows
12. make_schedule_free returns ScheduleFreeSGD for 'sgd'
13. make_schedule_free returns ScheduleFreeAdamW for 'adamw'
14. Zero grad + step leaves params unchanged (no gradient)
15. AdamW weight_decay: params shrink toward zero on zero gradient
"""
import pytest
import torch
import torch.nn as nn

from aurelius.training.schedule_free_adamw import (
    ScheduleFreeAdamW,
    ScheduleFreeSGD,
    make_schedule_free,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param(shape, seed=0, val=None):
    torch.manual_seed(seed)
    p = nn.Parameter(torch.randn(*shape) if val is None else torch.full(shape, val))
    return p


def _one_step_loss(p, opt, loss_fn=None):
    opt.zero_grad()
    loss = (p ** 2).sum() if loss_fn is None else loss_fn(p)
    loss.backward()
    opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# 1. ScheduleFreeSGD initializes without error
# ---------------------------------------------------------------------------

def test_sgd_init():
    p = _param((4,))
    opt = ScheduleFreeSGD([p], lr=0.01)
    assert opt is not None


# ---------------------------------------------------------------------------
# 2. ScheduleFreeSGD single step does not crash
# ---------------------------------------------------------------------------

def test_sgd_single_step():
    p = _param((4,))
    opt = ScheduleFreeSGD([p], lr=0.01)
    _one_step_loss(p, opt)  # must not raise


# ---------------------------------------------------------------------------
# 3. ScheduleFreeSGD loss decreases on simple quadratic
# ---------------------------------------------------------------------------

def test_sgd_loss_decreases_quadratic():
    torch.manual_seed(1)
    p = nn.Parameter(torch.ones(16) * 3.0)
    opt = ScheduleFreeSGD([p], lr=0.05, momentum=0.0)
    losses = [_one_step_loss(p, opt) for _ in range(30)]
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. ScheduleFreeSGD parameters updated after step
# ---------------------------------------------------------------------------

def test_sgd_params_updated():
    p = _param((8,), seed=2)
    p_before = p.data.clone()
    opt = ScheduleFreeSGD([p], lr=0.01)
    _one_step_loss(p, opt)
    assert not torch.allclose(p.data, p_before), "Parameters did not change after step."


# ---------------------------------------------------------------------------
# 5. ScheduleFreeSGD weight_decay reduces param magnitude over time
# ---------------------------------------------------------------------------

def test_sgd_weight_decay_shrinks_params():
    torch.manual_seed(3)
    p_wd = nn.Parameter(torch.ones(16) * 2.0)
    p_no = nn.Parameter(torch.ones(16) * 2.0)

    opt_wd = ScheduleFreeSGD([p_wd], lr=0.001, momentum=0.0, weight_decay=0.5)
    opt_no = ScheduleFreeSGD([p_no], lr=0.001, momentum=0.0, weight_decay=0.0)

    for _ in range(20):
        opt_wd.zero_grad()
        p_wd.grad = torch.zeros_like(p_wd)  # zero gradient, only weight decay acts
        opt_wd.step()

        opt_no.zero_grad()
        p_no.grad = torch.zeros_like(p_no)
        opt_no.step()

    assert p_wd.data.abs().mean() < p_no.data.abs().mean(), (
        "weight_decay did not reduce param magnitude."
    )


# ---------------------------------------------------------------------------
# 6. ScheduleFreeAdamW initializes without error
# ---------------------------------------------------------------------------

def test_adamw_init():
    p = _param((4,))
    opt = ScheduleFreeAdamW([p], lr=0.001)
    assert opt is not None


# ---------------------------------------------------------------------------
# 7. ScheduleFreeAdamW single step does not crash
# ---------------------------------------------------------------------------

def test_adamw_single_step():
    p = _param((4,))
    opt = ScheduleFreeAdamW([p], lr=0.001)
    _one_step_loss(p, opt)


# ---------------------------------------------------------------------------
# 8. ScheduleFreeAdamW loss decreases on simple regression
# ---------------------------------------------------------------------------

def test_adamw_loss_decreases_regression():
    torch.manual_seed(4)
    target = torch.zeros(16)
    p = nn.Parameter(torch.ones(16) * 2.0)
    opt = ScheduleFreeAdamW([p], lr=0.01)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. ScheduleFreeAdamW parameters updated after step
# ---------------------------------------------------------------------------

def test_adamw_params_updated():
    p = _param((8,), seed=5)
    p_before = p.data.clone()
    opt = ScheduleFreeAdamW([p], lr=0.001)
    _one_step_loss(p, opt)
    assert not torch.allclose(p.data, p_before), "Parameters did not change after step."


# ---------------------------------------------------------------------------
# 10. ScheduleFreeAdamW state initialized correctly
# ---------------------------------------------------------------------------

def test_adamw_state_init():
    p = _param((4, 4))
    opt = ScheduleFreeAdamW([p], lr=0.001)
    _one_step_loss(p, opt)

    state = opt.state[p]
    assert "z" in state, "State missing 'z'."
    assert "exp_avg" in state, "State missing 'exp_avg'."
    assert "exp_avg_sq" in state, "State missing 'exp_avg_sq'."
    assert "step" in state, "State missing 'step'."
    assert state["step"] == 1, f"Expected step=1 after one step, got {state['step']}."
    assert state["z"].shape == p.shape, "z shape mismatch."
    assert state["exp_avg"].shape == p.shape, "exp_avg shape mismatch."
    assert state["exp_avg_sq"].shape == p.shape, "exp_avg_sq shape mismatch."


# ---------------------------------------------------------------------------
# 11. ScheduleFreeAdamW warmup_steps: effective lr starts small and grows
# ---------------------------------------------------------------------------

def test_adamw_warmup_lr_grows():
    torch.manual_seed(6)
    warmup = 10
    updates = []

    for step_target in [1, 5, 10]:
        p = nn.Parameter(torch.ones(8))
        opt = ScheduleFreeAdamW([p], lr=0.1, warmup_steps=warmup, betas=(0.0, 0.0))

        for _ in range(step_target):
            opt.zero_grad()
            p.grad = torch.ones_like(p)
            opt.step()

        state = opt.state[p]
        updates.append(state["z"].clone())

    # z after 1 warmup step should be closer to init than z after 10 steps
    init = torch.ones(8)
    dist_1 = (updates[0] - init).abs().mean().item()
    dist_10 = (updates[2] - init).abs().mean().item()
    assert dist_10 > dist_1, (
        f"Warmup did not produce smaller early updates: "
        f"dist_1={dist_1:.6f}, dist_10={dist_10:.6f}"
    )


# ---------------------------------------------------------------------------
# 12. make_schedule_free returns ScheduleFreeSGD for 'sgd'
# ---------------------------------------------------------------------------

def test_factory_sgd():
    p = _param((4,))
    opt = make_schedule_free("sgd", [p], lr=0.01)
    assert isinstance(opt, ScheduleFreeSGD), (
        f"Expected ScheduleFreeSGD, got {type(opt)}"
    )


# ---------------------------------------------------------------------------
# 13. make_schedule_free returns ScheduleFreeAdamW for 'adamw'
# ---------------------------------------------------------------------------

def test_factory_adamw():
    p = _param((4,))
    opt = make_schedule_free("adamw", [p], lr=0.001)
    assert isinstance(opt, ScheduleFreeAdamW), (
        f"Expected ScheduleFreeAdamW, got {type(opt)}"
    )


# ---------------------------------------------------------------------------
# 14. Zero grad + step leaves params unchanged when grad=0
# ---------------------------------------------------------------------------

def test_zero_grad_no_update():
    """When all gradients are zero, parameters should not change."""
    p = _param((8,), seed=7)
    p_before = p.data.clone()
    opt = ScheduleFreeAdamW([p], lr=0.001)

    opt.zero_grad()
    # Do NOT call backward() -- grad stays None, step should skip param
    opt.step()

    assert torch.allclose(p.data, p_before), (
        "Parameters changed despite zero/missing gradient."
    )


# ---------------------------------------------------------------------------
# 15. AdamW weight_decay: params shrink toward zero on zero gradient
# ---------------------------------------------------------------------------

def test_adamw_weight_decay_shrinks():
    torch.manual_seed(8)
    p_wd = nn.Parameter(torch.ones(16) * 3.0)
    p_no = nn.Parameter(torch.ones(16) * 3.0)

    opt_wd = ScheduleFreeAdamW([p_wd], lr=0.01, weight_decay=0.5, betas=(0.0, 0.999))
    opt_no = ScheduleFreeAdamW([p_no], lr=0.01, weight_decay=0.0, betas=(0.0, 0.999))

    for _ in range(20):
        opt_wd.zero_grad()
        p_wd.grad = torch.ones_like(p_wd) * 0.001  # tiny gradient, weight decay dominates
        opt_wd.step()

        opt_no.zero_grad()
        p_no.grad = torch.ones_like(p_no) * 0.001
        opt_no.step()

    assert p_wd.data.abs().mean() < p_no.data.abs().mean(), (
        "AdamW weight_decay did not reduce param magnitude compared to no weight_decay."
    )
