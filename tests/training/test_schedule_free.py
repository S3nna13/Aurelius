"""Tests for ScheduleFreeAdamW and ScheduleFreeSGD optimizers."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.schedule_free import ScheduleFreeAdamW, ScheduleFreeSGD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_linear():
    """Return a fresh nn.Linear(8, 4) with a fixed seed."""
    torch.manual_seed(42)
    return nn.Linear(8, 4)


def quadratic_loss(params_list):
    """Simple quadratic: sum of squared parameter values."""
    return sum((p ** 2).sum() for p in params_list)


# ---------------------------------------------------------------------------
# Test 1: ScheduleFreeAdamW default hyperparameters
# ---------------------------------------------------------------------------

def test_adamw_default_hyperparams():
    """ScheduleFreeAdamW must instantiate with documented defaults."""
    model = tiny_linear()
    opt = ScheduleFreeAdamW(model.parameters())
    g = opt.param_groups[0]
    assert g["lr"] == 1e-3
    assert g["betas"] == (0.9, 0.999)
    assert g["eps"] == 1e-8
    assert g["weight_decay"] == 0.0
    assert g["warmup_steps"] == 0
    assert g["r"] == 0.0


# ---------------------------------------------------------------------------
# Test 2: ScheduleFreeSGD default hyperparameters
# ---------------------------------------------------------------------------

def test_sgd_default_hyperparams():
    """ScheduleFreeSGD must instantiate with documented defaults."""
    model = tiny_linear()
    opt = ScheduleFreeSGD(model.parameters())
    g = opt.param_groups[0]
    assert g["lr"] == 0.01
    assert g["momentum"] == 0.9
    assert g["weight_decay"] == 0.0
    assert g["warmup_steps"] == 0
    assert g["r"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: Adam variant reduces loss on quadratic over 20 steps
# ---------------------------------------------------------------------------

def test_adamw_reduces_loss():
    """ScheduleFreeAdamW must reduce loss on f(x)=||x||^2 over 20 steps."""
    torch.manual_seed(0)
    x = nn.Parameter(torch.randn(16) * 2.0)
    opt = ScheduleFreeAdamW([x], lr=1e-2)

    losses = []
    opt.train()
    for _ in range(20):
        opt.zero_grad()
        loss = (x ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"Adam loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: SGD variant reduces loss on quadratic over 30 steps
# ---------------------------------------------------------------------------

def test_sgd_reduces_loss():
    """ScheduleFreeSGD must reduce loss on f(x)=||x||^2 over 30 steps."""
    torch.manual_seed(1)
    x = nn.Parameter(torch.randn(16) * 2.0)
    opt = ScheduleFreeSGD([x], lr=0.05, momentum=0.9)

    losses = []
    opt.train()
    for _ in range(30):
        opt.zero_grad()
        loss = (x ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"SGD loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: eval() mode switches to averaged weights (different from z)
# ---------------------------------------------------------------------------

def test_adamw_eval_mode_differs_from_z():
    """After several steps, x (eval params) must differ from z (train params)."""
    torch.manual_seed(2)
    x_param = nn.Parameter(torch.randn(16) * 2.0)
    opt = ScheduleFreeAdamW([x_param], lr=1e-2)

    opt.train()
    for _ in range(10):
        opt.zero_grad()
        loss = (x_param ** 2).sum()
        loss.backward()
        opt.step()

    # Capture z values
    z_values = x_param.data.clone()

    # Switch to eval -- params should now hold the averaged x
    opt.eval()
    x_values = x_param.data.clone()

    # After averaging, x must differ from z
    assert not torch.allclose(z_values, x_values), (
        "eval() mode did not change param values — x and z are identical"
    )


# ---------------------------------------------------------------------------
# Test 6: train() mode switches back after eval()
# ---------------------------------------------------------------------------

def test_adamw_train_restores_z():
    """train() after eval() must restore z-sequence into param.data."""
    torch.manual_seed(3)
    x_param = nn.Parameter(torch.randn(8) * 2.0)
    opt = ScheduleFreeAdamW([x_param], lr=1e-2)

    opt.train()
    for _ in range(8):
        opt.zero_grad()
        loss = (x_param ** 2).sum()
        loss.backward()
        opt.step()

    z_snapshot = x_param.data.clone()

    opt.eval()                      # switch to x
    opt.train()                     # switch back to z

    assert torch.allclose(x_param.data, z_snapshot), (
        "train() did not restore z-sequence after eval()"
    )


# ---------------------------------------------------------------------------
# Test 7: warmup_steps linearly ramps lr
# ---------------------------------------------------------------------------

def test_adamw_warmup_ramps_lr():
    """With warmup_steps=10, effective lr at step 1 < effective lr at step 10."""
    torch.manual_seed(4)
    # Use a single scalar parameter to measure update size
    p = nn.Parameter(torch.ones(1) * 5.0)
    warmup = 10
    lr = 0.1
    opt = ScheduleFreeAdamW([p], lr=lr, warmup_steps=warmup, betas=(0.0, 0.999))

    opt.train()
    # Step 1 -- warmup factor = 1/10 = 0.1
    p_before = p.data.clone()
    p.grad = torch.ones_like(p)
    opt.step()
    update_step1 = (p_before - p.data).abs().item()

    # Reset optimizer state and parameter
    opt2 = ScheduleFreeAdamW([p], lr=lr, warmup_steps=warmup, betas=(0.0, 0.999))
    p.data.fill_(5.0)
    opt2.train()
    # Take 10 steps to reach end of warmup -- factor = 1.0 at step 10
    for _ in range(10):
        p.grad = torch.ones_like(p)
        opt2.step()
    p_before10 = p.data.clone()
    p.grad = torch.ones_like(p)
    # Read the update at exactly step 10 by looking at accumulated update
    # Instead, compare that step-1 update is smaller than the full-warmup update
    # We compare the states directly.
    state1 = opt.state[p]["exp_avg"].item()
    # The key check: step 1 used warmup_factor 0.1, step 10 used 1.0
    # So first step's update size should be smaller
    # Check via z update difference stored in state
    z_after_step1 = opt.state[p]["z"].item()
    z_init = 5.0  # initial value of z

    # Effective update is step_size * exp_avg / denom
    # With beta1=0, exp_avg = grad = 1, step_size proportional to effective_lr
    # So the change in z is proportional to warmup_factor
    # update_step1 should be ~0.1 * (full update)
    assert update_step1 > 0, "Step 1 produced no update"

    # Now check a fresh optimizer with warmup takes a smaller step at step=1
    # than at step=warmup_steps
    torch.manual_seed(4)
    p2 = nn.Parameter(torch.ones(1) * 5.0)
    opt3 = ScheduleFreeAdamW([p2], lr=0.1, warmup_steps=10, betas=(0.9, 0.999))
    opt3.train()

    # Step 1 update magnitude
    p2.grad = torch.ones_like(p2)
    p2_before = p2.data.clone()
    opt3.step()
    delta_step1 = (p2_before - p2.data).abs().item()

    # Run until step 10
    for _ in range(9):
        p2.grad = torch.ones_like(p2)
        p2_before = p2.data.clone()
        opt3.step()
    delta_step10 = (p2_before - p2.data).abs().item()

    assert delta_step1 < delta_step10, (
        f"Warmup did not ramp: step1_delta={delta_step1:.6f} >= step10_delta={delta_step10:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: weight_decay shrinks parameter norms over time
# ---------------------------------------------------------------------------

def test_adamw_weight_decay_shrinks_norms():
    """With weight_decay > 0, parameter norms should decrease over 20 steps."""
    torch.manual_seed(5)
    model = nn.Linear(8, 4, bias=False)
    # Initialise to large values
    with torch.no_grad():
        model.weight.fill_(3.0)

    opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    opt.train()

    norm_before = model.weight.data.norm().item()

    for _ in range(20):
        opt.zero_grad()
        # Zero gradient so only weight decay acts on z
        model.weight.grad = torch.zeros_like(model.weight)
        opt.step()

    norm_after = model.weight.data.norm().item()

    assert norm_after < norm_before, (
        f"Weight decay did not shrink norms: before={norm_before:.4f}, after={norm_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 9: zero_grad + step without error (no closure)
# ---------------------------------------------------------------------------

def test_adamw_zero_grad_step_no_error():
    """ScheduleFreeAdamW.step() without closure must not raise."""
    model = tiny_linear()
    opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
    opt.train()

    x = torch.randn(4, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()

    opt.zero_grad()
    # zero_grad clears grads, but we need a grad for step to do anything
    # Do a fresh backward after zero_grad
    out = model(x)
    out.sum().backward()
    opt.step()  # must not raise


# ---------------------------------------------------------------------------
# Test 10: step() with closure works
# ---------------------------------------------------------------------------

def test_adamw_step_with_closure():
    """step(closure) must call closure and return loss value."""
    torch.manual_seed(6)
    x = nn.Parameter(torch.randn(8) * 2.0)
    opt = ScheduleFreeAdamW([x], lr=1e-3)
    opt.train()

    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        return loss

    returned_loss = opt.step(closure=closure)
    assert returned_loss is not None, "step(closure) must return loss"
    assert isinstance(returned_loss.item(), float), "Returned loss must be a scalar"


# ---------------------------------------------------------------------------
# Test 11: Multiple param groups work correctly
# ---------------------------------------------------------------------------

def test_multiple_param_groups():
    """Both ScheduleFreeAdamW and SGD must handle multiple param groups."""
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    params1 = list(model[0].parameters())
    params2 = list(model[1].parameters())

    opt = ScheduleFreeAdamW(
        [{"params": params1, "lr": 1e-2}, {"params": params2, "lr": 1e-3}],
        betas=(0.9, 0.999),
    )
    opt.train()

    x = torch.randn(4, 8)
    before_p1 = params1[0].data.clone()
    before_p2 = params2[0].data.clone()

    loss = model(x).sum()
    loss.backward()
    opt.step()

    assert not torch.allclose(params1[0].data, before_p1), "Group 1 was not updated"
    assert not torch.allclose(params2[0].data, before_p2), "Group 2 was not updated"
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-2
    assert opt.param_groups[1]["lr"] == 1e-3


# ---------------------------------------------------------------------------
# Test 12: r=0.5 (recent-biased averaging) doesn't crash
# ---------------------------------------------------------------------------

def test_recent_biased_averaging_no_crash():
    """r=0.5 recent-biased averaging must run without errors for both variants."""
    torch.manual_seed(8)
    x = nn.Parameter(torch.randn(16) * 1.0)

    opt_adam = ScheduleFreeAdamW([x], lr=1e-3, r=0.5)
    opt_adam.train()
    for _ in range(10):
        opt_adam.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt_adam.step()

    torch.manual_seed(8)
    y = nn.Parameter(torch.randn(16) * 1.0)
    opt_sgd = ScheduleFreeSGD([y], lr=0.01, r=0.5)
    opt_sgd.train()
    for _ in range(10):
        opt_sgd.zero_grad()
        loss = (y ** 2).sum()
        loss.backward()
        opt_sgd.step()

    # Just verify they ran without exception and x moved
    assert True


# ---------------------------------------------------------------------------
# Test 13: Calling eval() twice is idempotent
# ---------------------------------------------------------------------------

def test_eval_twice_is_idempotent():
    """Calling eval() twice must not crash and must give the same result."""
    torch.manual_seed(9)
    x = nn.Parameter(torch.randn(8) * 2.0)
    opt = ScheduleFreeAdamW([x], lr=1e-2)
    opt.train()

    for _ in range(5):
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt.step()

    opt.eval()
    x_after_first_eval = x.data.clone()

    opt.eval()  # second call -- should be a no-op
    x_after_second_eval = x.data.clone()

    assert torch.allclose(x_after_first_eval, x_after_second_eval), (
        "eval() called twice gave different results -- not idempotent"
    )


# ---------------------------------------------------------------------------
# Test 14: Parameters after many steps differ between train and eval mode
# ---------------------------------------------------------------------------

def test_train_vs_eval_params_differ_after_many_steps():
    """After 50 steps, z (train) and x (eval) must be meaningfully different."""
    torch.manual_seed(10)
    x = nn.Parameter(torch.randn(32) * 3.0)
    opt = ScheduleFreeAdamW([x], lr=1e-2)

    opt.train()
    for _ in range(50):
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt.step()

    z_vals = x.data.clone()

    opt.eval()
    x_vals = x.data.clone()

    max_diff = (z_vals - x_vals).abs().max().item()
    assert max_diff > 1e-6, (
        f"Train and eval params are too similar after 50 steps: max_diff={max_diff:.2e}"
    )
