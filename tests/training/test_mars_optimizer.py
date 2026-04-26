"""Tests for the MARS optimizer."""

import copy

import torch
import torch.nn as nn

from src.training.mars_optimizer import MARSOptimizer

# ---------------------------------------------------------------------------
# Test 1: Basic instantiation with default params
# ---------------------------------------------------------------------------


def test_instantiation_defaults():
    """MARSOptimizer must be constructable and store correct default hyperparams."""
    p = nn.Parameter(torch.randn(4, 4))
    opt = MARSOptimizer([p])

    group = opt.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"] == (0.9, 0.99)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0
    assert group["gamma"] == 0.025
    assert group["mars_type"] == "mars-adamw"


# ---------------------------------------------------------------------------
# Test 2: Single step updates parameters
# ---------------------------------------------------------------------------


def test_single_step_updates_params():
    """Parameters must change after a single MARS optimizer step."""
    p = nn.Parameter(torch.randn(8, 8))
    opt = MARSOptimizer([p], lr=1e-3)

    before = p.data.clone()
    p.grad = torch.ones_like(p)
    opt.step()

    assert not torch.allclose(p.data, before), (
        "Parameter was not updated after a single optimizer step."
    )


# ---------------------------------------------------------------------------
# Test 3: Loss decreases on simple quadratic over 100 steps
# ---------------------------------------------------------------------------


def test_quadratic_loss_decreases():
    """Loss must decrease when minimizing f(x) = ||x||^2 over 100 steps."""
    x = nn.Parameter(torch.randn(32) * 3.0)
    opt = MARSOptimizer([x], lr=1e-2, betas=(0.9, 0.99), gamma=0.025)

    initial_loss = None
    for _ in range(100):
        opt.zero_grad()
        loss = (x**2).sum()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()

    final_loss = (x**2).sum().item()
    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Weight decay shrinks params without gradients
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_params():
    """With weight_decay > 0 and zero gradients, parameters must shrink toward 0."""
    p = nn.Parameter(torch.ones(16, 16) * 3.0)
    opt = MARSOptimizer([p], lr=1e-2, weight_decay=0.1, gamma=0.0)

    # Zero gradients — only weight decay acts
    p.grad = torch.zeros_like(p)
    norm_before = p.data.norm().item()
    opt.step()
    norm_after = p.data.norm().item()

    assert norm_after < norm_before, (
        f"Weight decay did not shrink param: before={norm_before:.4f}, after={norm_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: gamma=0 gives standard Adam behaviour
# ---------------------------------------------------------------------------


def test_gamma_zero_equals_adam():
    """With gamma=0, MARS should produce updates identical to standard AdamW."""
    torch.manual_seed(42)
    p_mars = nn.Parameter(torch.randn(8, 8))
    torch.manual_seed(42)
    p_adam = nn.Parameter(p_mars.data.clone())

    opt_mars = MARSOptimizer(
        [p_mars], lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, gamma=0.0
    )
    opt_adam = torch.optim.AdamW([p_adam], lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)

    torch.manual_seed(0)
    for _ in range(5):
        g = torch.randn_like(p_mars)
        p_mars.grad = g.clone()
        p_adam.grad = g.clone()
        opt_mars.step()
        opt_adam.step()

    assert torch.allclose(p_mars.data, p_adam.data, atol=1e-6), (
        f"gamma=0 MARS diverged from AdamW. "
        f"Max diff: {(p_mars.data - p_adam.data).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 6: get_lr() returns correct learning rates
# ---------------------------------------------------------------------------


def test_get_lr_returns_correct_values():
    """get_lr() must return the lr for each parameter group."""
    p1 = nn.Parameter(torch.randn(4))
    p2 = nn.Parameter(torch.randn(4))

    opt = MARSOptimizer(
        [{"params": [p1], "lr": 1e-3}, {"params": [p2], "lr": 5e-4}],
        lr=1e-3,
    )

    lrs = opt.get_lr()
    assert lrs == [1e-3, 5e-4], f"Expected [1e-3, 5e-4], got {lrs}"


def test_get_lr_single_group():
    """get_lr() with a single param group returns a list with one element."""
    p = nn.Parameter(torch.randn(4))
    opt = MARSOptimizer([p], lr=2e-4)
    assert opt.get_lr() == [2e-4]


# ---------------------------------------------------------------------------
# Test 7: zero_variance_reduction() resets last_grad to None
# ---------------------------------------------------------------------------


def test_zero_variance_reduction_resets_last_grad():
    """zero_variance_reduction() must set last_grad to None for all params."""
    p = nn.Parameter(torch.randn(8))
    opt = MARSOptimizer([p], lr=1e-3)

    # Run two steps so that last_grad is populated
    for _ in range(2):
        p.grad = torch.randn_like(p)
        opt.step()

    # Verify last_grad is not None before reset
    assert opt.state[p]["last_grad"] is not None, (
        "last_grad should be a tensor after steps, not None"
    )

    opt.zero_variance_reduction()

    assert opt.state[p]["last_grad"] is None, (
        "last_grad should be None after zero_variance_reduction()"
    )


# ---------------------------------------------------------------------------
# Test 8: State dict save/load preserves optimizer state
# ---------------------------------------------------------------------------


def test_state_dict_save_load():
    """State dict round-trip must preserve optimizer state tensors."""
    p = nn.Parameter(torch.randn(8, 8))
    opt = MARSOptimizer([p], lr=1e-3, gamma=0.025)

    # Run a few steps to populate state
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()

    state_before = copy.deepcopy(opt.state_dict())

    # Create a fresh optimizer and load the saved state
    p2 = nn.Parameter(p.data.clone())
    opt2 = MARSOptimizer([p2], lr=1e-3, gamma=0.025)
    opt2.load_state_dict(state_before)

    state_after = opt2.state_dict()

    # Verify step count matches
    for key in state_before["state"]:
        assert state_before["state"][key]["step"] == state_after["state"][key]["step"], (
            f"Step count mismatch for state key {key}"
        )
        # exp_avg must be preserved
        assert torch.allclose(
            state_before["state"][key]["exp_avg"],
            state_after["state"][key]["exp_avg"],
            atol=1e-7,
        ), f"exp_avg mismatch after state_dict load for key {key}"


# ---------------------------------------------------------------------------
# Test 9: Works with parameter groups having different lrs
# ---------------------------------------------------------------------------


def test_parameter_groups_different_lrs():
    """Different param groups with distinct lrs must both update independently."""
    p1 = nn.Parameter(torch.ones(8, 8))
    p2 = nn.Parameter(torch.ones(8, 8))

    opt = MARSOptimizer(
        [{"params": [p1], "lr": 1e-2}, {"params": [p2], "lr": 1e-4}],
        betas=(0.9, 0.99),
        gamma=0.0,
    )

    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    opt.step()

    # Both must have changed
    assert not torch.allclose(p1.data, torch.ones_like(p1)), "p1 (lr=1e-2) was not updated"
    assert not torch.allclose(p2.data, torch.ones_like(p2)), "p2 (lr=1e-4) was not updated"

    # p1 should have moved more than p2 because its lr is larger
    delta1 = (p1.data - torch.ones_like(p1)).abs().mean().item()
    delta2 = (p2.data - torch.ones_like(p2)).abs().mean().item()
    assert delta1 > delta2, (
        f"Expected p1 (lr=1e-2) to move more than p2 (lr=1e-4), "
        f"but delta1={delta1:.6f} <= delta2={delta2:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 10: gradient clipping compatible
# ---------------------------------------------------------------------------


def test_gradient_clipping_compatible():
    """MARS optimizer must work correctly after torch.nn.utils.clip_grad_norm_."""
    model = nn.Linear(16, 16)
    opt = MARSOptimizer(model.parameters(), lr=1e-3, gamma=0.025)

    x = torch.randn(4, 16)
    before = {name: p.data.clone() for name, p in model.named_parameters()}

    loss = model(x).sum()
    loss.backward()

    # Apply gradient clipping before optimizer step
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert total_norm >= 0.0, "clip_grad_norm_ returned a negative norm"

    opt.step()

    changed = any(not torch.allclose(p.data, before[name]) for name, p in model.named_parameters())
    assert changed, "No parameters were updated after gradient clipping + optimizer step"

    # Ensure no NaNs or Infs in parameters
    for name, p in model.named_parameters():
        assert torch.isfinite(p.data).all(), (
            f"Parameter '{name}' contains non-finite values after clipped step"
        )
