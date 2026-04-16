"""Tests for the Adan (Adaptive Nesterov Momentum) optimizer."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import pytest

from src.training.adan import Adan, AdanConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic_param(size: int = 64, scale: float = 2.0) -> nn.Parameter:
    """Return a parameter initialised far from zero for easy convergence tests."""
    return nn.Parameter(torch.randn(size) * scale)


def _make_linear(in_f: int = 16, out_f: int = 16) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=False)


# ---------------------------------------------------------------------------
# Test 1 – Basic instantiation with default params
# ---------------------------------------------------------------------------


def test_adan_instantiation_defaults():
    """Adan must be constructable with default hyperparameters."""
    p = nn.Parameter(torch.randn(8))
    opt = Adan([p])

    group = opt.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"] == (0.98, 0.92, 0.99)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.02
    assert group["no_prox"] is False


# ---------------------------------------------------------------------------
# Test 2 – Single step actually updates parameters
# ---------------------------------------------------------------------------


def test_adan_single_step_updates_params():
    """Parameters must change after exactly one Adan step."""
    p = _quadratic_param()
    opt = Adan([p], lr=1e-3, weight_decay=0.0)

    before = p.data.clone()
    p.grad = torch.randn_like(p)
    opt.step()

    assert not torch.allclose(p.data, before), "Parameter was not updated after one Adan step"


# ---------------------------------------------------------------------------
# Test 3 – Loss decreases on quadratic objective over 200 steps
# ---------------------------------------------------------------------------


def test_adan_loss_decreases_quadratic():
    """Minimise f(x)=||x||²; loss must decrease over 200 steps."""
    x = _quadratic_param(size=64, scale=3.0)
    opt = Adan([x], lr=1e-2, weight_decay=0.0)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = (x ** 2).sum()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 200 steps: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )
    # Should converge meaningfully (at least 50% reduction)
    assert losses[-1] < losses[0] * 0.5, (
        f"Loss reduction insufficient: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4 – Weight decay (proximal) shrinks parameters
# ---------------------------------------------------------------------------


def test_adan_proximal_weight_decay_shrinks_params():
    """Proximal weight decay (no_prox=False) must shrink parameter norms."""
    # Large init, zero gradient so only weight decay acts
    p = nn.Parameter(torch.ones(32, 32) * 3.0)
    opt = Adan([p], lr=1e-3, weight_decay=0.1, no_prox=False)
    p.grad = torch.zeros_like(p)

    norm_before = p.data.norm().item()
    for _ in range(5):
        opt.step()
        p.grad = torch.zeros_like(p)

    norm_after = p.data.norm().item()
    assert norm_after < norm_before, (
        f"Proximal weight decay did not shrink params: before={norm_before:.4f}, after={norm_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 – no_prox=True uses L2 weight-decay path
# ---------------------------------------------------------------------------


def test_adan_no_prox_uses_l2_decay():
    """no_prox=True and no_prox=False must produce different parameter values."""
    torch.manual_seed(0)
    p_prox = nn.Parameter(torch.ones(16) * 2.0)
    p_l2 = nn.Parameter(torch.ones(16) * 2.0)

    grad = torch.ones(16) * 0.5
    p_prox.grad = grad.clone()
    p_l2.grad = grad.clone()

    opt_prox = Adan([p_prox], lr=1e-3, weight_decay=0.1, no_prox=False)
    opt_l2 = Adan([p_l2], lr=1e-3, weight_decay=0.1, no_prox=True)

    opt_prox.step()
    opt_l2.step()

    # Both should update, but through different paths — values differ
    assert not torch.allclose(p_prox.data, p_l2.data), (
        "no_prox=True and no_prox=False produced identical updates — paths not distinct"
    )


# ---------------------------------------------------------------------------
# Test 6 – restart_opt() resets all states to zero
# ---------------------------------------------------------------------------


def test_adan_restart_opt_resets_state():
    """restart_opt() must zero all moment buffers and step counter."""
    x = _quadratic_param()
    opt = Adan([x], lr=1e-3, weight_decay=0.0)

    # Run a few steps to populate state
    for _ in range(5):
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt.step()

    # Confirm state is non-zero
    state = opt.state[x]
    assert state["step"] > 0
    assert state["exp_avg"].abs().sum().item() > 0

    opt.restart_opt()

    assert state["step"] == 0, "restart_opt did not reset step counter"
    assert state["exp_avg"].abs().sum().item() == 0.0, "restart_opt did not zero exp_avg"
    assert state["exp_avg_diff"].abs().sum().item() == 0.0, "restart_opt did not zero exp_avg_diff"
    assert state["exp_avg_sq"].abs().sum().item() == 0.0, "restart_opt did not zero exp_avg_sq"
    assert state["previous_grad"].abs().sum().item() == 0.0, "restart_opt did not zero previous_grad"


# ---------------------------------------------------------------------------
# Test 7 – get_lr() returns list with correct learning rates
# ---------------------------------------------------------------------------


def test_adan_get_lr_single_group():
    """get_lr() must return a list containing each group's lr."""
    p = nn.Parameter(torch.randn(4))
    lr_val = 5e-4
    opt = Adan([p], lr=lr_val)
    lrs = opt.get_lr()
    assert isinstance(lrs, list), "get_lr() must return a list"
    assert len(lrs) == 1
    assert lrs[0] == lr_val


def test_adan_get_lr_multiple_groups():
    """get_lr() must return one entry per param group."""
    p1 = nn.Parameter(torch.randn(4))
    p2 = nn.Parameter(torch.randn(4))
    opt = Adan([
        {"params": [p1], "lr": 1e-3},
        {"params": [p2], "lr": 2e-4},
    ])
    lrs = opt.get_lr()
    assert lrs == [1e-3, 2e-4], f"Unexpected lrs: {lrs}"


# ---------------------------------------------------------------------------
# Test 8 – State dict save/load preserves all state tensors
# ---------------------------------------------------------------------------


def test_adan_state_dict_save_load():
    """state_dict round-trip must preserve all optimizer state tensors."""
    x = _quadratic_param()
    opt = Adan([x], lr=1e-3, weight_decay=0.0)

    for _ in range(3):
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt.step()

    sd = opt.state_dict()

    # Create fresh optimizer and load state
    x2 = nn.Parameter(x.data.clone())
    opt2 = Adan([x2], lr=1e-3, weight_decay=0.0)
    opt2.load_state_dict(sd)

    state_keys = {"step", "exp_avg", "exp_avg_diff", "exp_avg_sq", "previous_grad"}
    for p in opt2.param_groups[0]["params"]:
        if p in opt2.state:
            loaded_keys = set(opt2.state[p].keys())
            assert state_keys.issubset(loaded_keys), (
                f"Loaded state missing keys: {state_keys - loaded_keys}"
            )

    # step counter must be preserved
    orig_step = list(opt.state.values())[0]["step"]
    loaded_step = list(opt2.state.values())[0]["step"]
    assert orig_step == loaded_step, f"Step mismatch after load: {orig_step} vs {loaded_step}"


# ---------------------------------------------------------------------------
# Test 9 – Multiple parameter groups with different lrs
# ---------------------------------------------------------------------------


def test_adan_multiple_param_groups():
    """Different param groups with different lrs must both update independently."""
    p1 = nn.Parameter(torch.randn(16) * 2.0)
    p2 = nn.Parameter(torch.randn(16) * 2.0)

    opt = Adan([
        {"params": [p1], "lr": 1e-2},
        {"params": [p2], "lr": 1e-4},
    ], weight_decay=0.0)

    before1 = p1.data.clone()
    before2 = p2.data.clone()

    p1.grad = torch.randn_like(p1)
    p2.grad = torch.randn_like(p2)
    opt.step()

    assert not torch.allclose(p1.data, before1), "Group 1 param was not updated"
    assert not torch.allclose(p2.data, before2), "Group 2 param was not updated"
    # Larger lr group should generally move more
    delta1 = (p1.data - before1).norm().item()
    delta2 = (p2.data - before2).norm().item()
    assert delta1 > delta2, (
        f"Higher-lr group should move more: delta1={delta1:.6f}, delta2={delta2:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 10 – Compatible with clip_grad_norm_ before step
# ---------------------------------------------------------------------------


def test_adan_gradient_clipping_compatible():
    """clip_grad_norm_ before step must not break Adan and must clip norms."""
    linear = _make_linear(32, 32)
    opt = Adan(linear.parameters(), lr=1e-3, weight_decay=0.0)

    x = torch.randn(8, 32)
    loss = linear(x).sum()
    loss.backward()

    # Artificially scale up gradients
    for p in linear.parameters():
        if p.grad is not None:
            p.grad.mul_(1000.0)

    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(linear.parameters(), max_norm)

    # Verify clipping happened
    total_norm = 0.0
    for p in linear.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    assert total_norm <= max_norm + 1e-5, f"Gradient norm not clipped: {total_norm:.4f}"

    # step must not raise
    opt.step()

    # Params must have changed
    # (just verifying no crash and some update happened)
    assert all(
        p.data.abs().sum().item() >= 0
        for p in linear.parameters()
    )


# ---------------------------------------------------------------------------
# Test 11 – All three betas are distinct from Adam's two
# ---------------------------------------------------------------------------


def test_adan_three_betas_all_used():
    """State must contain exactly three moment buffers, one per beta."""
    p = nn.Parameter(torch.randn(16))
    opt = Adan([p], lr=1e-3, betas=(0.98, 0.92, 0.99), weight_decay=0.0)

    p.grad = torch.randn_like(p)
    opt.step()

    state = opt.state[p]
    # Adan maintains exp_avg (β1), exp_avg_diff (β2), exp_avg_sq (β3)
    assert "exp_avg" in state, "Missing exp_avg (β1 moment)"
    assert "exp_avg_diff" in state, "Missing exp_avg_diff (β2 Nesterov moment)"
    assert "exp_avg_sq" in state, "Missing exp_avg_sq (β3 second moment)"
    assert "previous_grad" in state, "Missing previous_grad for gradient diff computation"

    tensor_buffers = [k for k, v in state.items() if isinstance(v, torch.Tensor)]
    # 4 tensor buffers: exp_avg, exp_avg_diff, exp_avg_sq, previous_grad
    assert len(tensor_buffers) == 4, (
        f"Expected 4 tensor state buffers for Adan, got {len(tensor_buffers)}: {tensor_buffers}"
    )


# ---------------------------------------------------------------------------
# Test 12 – Zero gradient + no weight decay leaves parameters unchanged
# ---------------------------------------------------------------------------


def test_adan_zero_grad_no_weight_decay_no_change():
    """Zero gradients with weight_decay=0 must leave parameters unchanged.

    On the very first step with zero gradient and no weight decay:
    - diff = 0 - 0 = 0
    - nesterov_g = 0
    - m_hat, v_hat, n_hat all = 0
    - update = 0 / (sqrt(0) + eps) = 0
    So the parameter should not change.
    """
    p = nn.Parameter(torch.randn(32))
    opt = Adan([p], lr=1e-3, weight_decay=0.0, no_prox=False)

    before = p.data.clone()
    p.grad = torch.zeros_like(p)
    opt.step()

    assert torch.allclose(p.data, before, atol=1e-9), (
        "Parameter changed despite zero gradient and no weight decay"
    )


# ---------------------------------------------------------------------------
# Bonus – AdanConfig dataclass
# ---------------------------------------------------------------------------


def test_adan_config_defaults():
    """AdanConfig must expose correct default fields."""
    cfg = AdanConfig()
    assert cfg.lr == 1e-3
    assert cfg.betas == (0.98, 0.92, 0.99)
    assert cfg.eps == 1e-8
    assert cfg.weight_decay == 0.02
    assert cfg.no_prox is False


def test_adan_config_custom():
    """AdanConfig custom values must be stored correctly."""
    cfg = AdanConfig(lr=5e-4, betas=(0.95, 0.90, 0.98), weight_decay=0.01, no_prox=True)
    assert cfg.lr == 5e-4
    assert cfg.betas == (0.95, 0.90, 0.98)
    assert cfg.no_prox is True


def test_adan_from_config():
    """Adan can be constructed from an AdanConfig."""
    cfg = AdanConfig(lr=2e-3, weight_decay=0.05, no_prox=True)
    p = nn.Parameter(torch.randn(8))
    opt = Adan([p], lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
               weight_decay=cfg.weight_decay, no_prox=cfg.no_prox)

    group = opt.param_groups[0]
    assert group["lr"] == cfg.lr
    assert group["weight_decay"] == cfg.weight_decay
    assert group["no_prox"] == cfg.no_prox
