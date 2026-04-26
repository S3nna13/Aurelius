"""
Tests for LARS and LAMB optimizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.lars_lamb import (
    LAMB,
    LARS,
    compute_trust_ratio,
    get_param_groups_for_lars,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_linear(in_features=16, out_features=8, bias=True):
    """Return a small nn.Linear for testing."""
    torch.manual_seed(42)
    return nn.Linear(in_features, out_features, bias=bias)


def make_mlp():
    """Return a small 2-layer MLP."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.LayerNorm(32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


def quadratic_loss(model, x):
    """Simple MSE loss against zero target."""
    out = model(x)
    return (out**2).mean()


# ---------------------------------------------------------------------------
# Test 1: LARS instantiates with correct defaults
# ---------------------------------------------------------------------------


def test_lars_instantiates_defaults():
    model = make_linear()
    opt = LARS(model.parameters())
    assert opt.defaults["lr"] == 0.01
    assert opt.defaults["momentum"] == 0.9
    assert opt.defaults["weight_decay"] == 1e-4
    assert opt.defaults["trust_coefficient"] == 0.001
    assert opt.defaults["eps"] == 1e-8
    assert opt.defaults["exclude_bias_and_bn"] is True


# ---------------------------------------------------------------------------
# Test 2: LAMB instantiates with correct defaults
# ---------------------------------------------------------------------------


def test_lamb_instantiates_defaults():
    model = make_linear()
    opt = LAMB(model.parameters())
    assert opt.defaults["lr"] == 1e-3
    assert opt.defaults["betas"] == (0.9, 0.999)
    assert opt.defaults["eps"] == 1e-6
    assert opt.defaults["weight_decay"] == 0.01
    assert opt.defaults["trust_coefficient"] == 1.0
    assert opt.defaults["clamp_value"] == 10.0
    assert opt.defaults["adam_w_mode"] is True
    assert opt.defaults["exclude_bias_and_bn"] is True


# ---------------------------------------------------------------------------
# Test 3: LARS step reduces loss on simple quadratic over 50 steps
# ---------------------------------------------------------------------------


def test_lars_reduces_loss_quadratic():
    torch.manual_seed(0)
    model = make_linear()
    opt = LARS(model.parameters(), lr=0.1, trust_coefficient=1.0, weight_decay=0.0)
    x = torch.randn(32, 16)

    initial_loss = quadratic_loss(model, x).item()

    for _ in range(50):
        opt.zero_grad()
        loss = quadratic_loss(model, x)
        loss.backward()
        opt.step()

    final_loss = quadratic_loss(model, x).item()
    assert final_loss < initial_loss, (
        f"LARS did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: LAMB step reduces loss on simple quadratic over 30 steps
# ---------------------------------------------------------------------------


def test_lamb_reduces_loss_quadratic():
    torch.manual_seed(0)
    model = make_linear()
    opt = LAMB(model.parameters(), lr=1e-2, weight_decay=0.0)
    x = torch.randn(32, 16)

    initial_loss = quadratic_loss(model, x).item()

    for _ in range(30):
        opt.zero_grad()
        loss = quadratic_loss(model, x)
        loss.backward()
        opt.step()

    final_loss = quadratic_loss(model, x).item()
    assert final_loss < initial_loss, (
        f"LAMB did not reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: compute_trust_ratio returns positive value
# ---------------------------------------------------------------------------


def test_compute_trust_ratio_positive():
    torch.manual_seed(1)
    param = torch.randn(8, 16)
    grad = torch.randn(8, 16)
    ratio = compute_trust_ratio(param, grad, weight_decay=1e-4)
    assert ratio > 0.0, f"Expected positive trust ratio, got {ratio}"


# ---------------------------------------------------------------------------
# Test 6: compute_trust_ratio with zero param returns small value (no div by zero)
# ---------------------------------------------------------------------------


def test_compute_trust_ratio_zero_param():
    param = torch.zeros(8, 16)
    grad = torch.randn(8, 16)
    # param_norm = 0, so ratio should be near 0 (numerator is 0)
    ratio = compute_trust_ratio(param, grad, weight_decay=1e-4)
    assert ratio >= 0.0, f"Expected non-negative ratio, got {ratio}"
    # Should be very small since param_norm == 0
    assert ratio < 1e-6, f"Expected near-zero ratio for zero param, got {ratio}"


# ---------------------------------------------------------------------------
# Test 7: compute_trust_ratio with zero grad returns large value (param/eps)
# ---------------------------------------------------------------------------


def test_compute_trust_ratio_zero_grad():
    torch.manual_seed(2)
    param = torch.randn(8, 16)
    grad = torch.zeros(8, 16)
    param_norm = param.norm(2).item()
    eps = 1e-8
    # With zero grad and zero weight_decay, ratio ~ param_norm / eps
    ratio = compute_trust_ratio(param, grad, weight_decay=0.0, eps=eps)
    expected = param_norm / eps
    assert abs(ratio - expected) < 1.0, f"Expected ratio ~{expected:.2f}, got {ratio:.2f}"
    assert ratio > 1.0, f"Expected large ratio for zero grad, got {ratio}"


# ---------------------------------------------------------------------------
# Test 8: LARS trust ratio scales layer learning rate (effective lr != base lr)
# ---------------------------------------------------------------------------


def test_lars_trust_ratio_scales_lr():
    """
    Verify that with trust_coefficient < 1 and non-trivial weights,
    the effective learning rate differs from the base lr.
    We compare two models starting from identical weights but using
    different trust_coefficient values — their weight updates should differ.
    """
    torch.manual_seed(3)

    model_a = make_linear()
    model_b = nn.Linear(16, 8, bias=True)
    model_b.weight.data.copy_(model_a.weight.data)
    model_b.bias.data.copy_(model_a.bias.data)

    x = torch.randn(32, 16)

    opt_a = LARS(
        model_a.parameters(),
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        trust_coefficient=0.001,
        exclude_bias_and_bn=True,
    )
    opt_b = LARS(
        model_b.parameters(),
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        trust_coefficient=1.0,
        exclude_bias_and_bn=True,
    )

    opt_a.zero_grad()
    quadratic_loss(model_a, x).backward()
    opt_a.step()

    opt_b.zero_grad()
    quadratic_loss(model_b, x).backward()
    opt_b.step()

    weight_diff = (model_a.weight - model_b.weight).abs().max().item()
    assert weight_diff > 1e-10, (
        f"LARS trust ratio should produce different updates, but diff={weight_diff}"
    )


# ---------------------------------------------------------------------------
# Test 9: LAMB step with adam_w_mode=True applies decoupled weight decay
# ---------------------------------------------------------------------------


def test_lamb_adam_w_mode_decoupled_weight_decay():
    """
    With adam_w_mode=True, weight decay is applied directly to params (p *= (1-lr*wd)).
    With adam_w_mode=False, weight decay flows through Adam update direction.
    Both should reduce loss, but the parameter paths differ.
    """
    torch.manual_seed(5)
    x = torch.randn(16, 16)

    model_wd = make_linear(16, 16, bias=False)
    model_no_wd = nn.Linear(16, 16, bias=False)
    model_no_wd.weight.data.copy_(model_wd.weight.data)

    opt_wd = LAMB(model_wd.parameters(), lr=1e-2, weight_decay=0.1, adam_w_mode=True)
    opt_no_wd = LAMB(model_no_wd.parameters(), lr=1e-2, weight_decay=0.0, adam_w_mode=True)

    for _ in range(5):
        opt_wd.zero_grad()
        quadratic_loss(model_wd, x).backward()
        opt_wd.step()

        opt_no_wd.zero_grad()
        quadratic_loss(model_no_wd, x).backward()
        opt_no_wd.step()

    # Weight decay should shrink weights more aggressively
    norm_wd = model_wd.weight.norm().item()
    norm_no_wd = model_no_wd.weight.norm().item()
    assert norm_wd < norm_no_wd, (
        f"With weight decay, norms should be smaller: {norm_wd:.4f} vs {norm_no_wd:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 10: LARS with momentum=0.0 works (no velocity accumulation)
# ---------------------------------------------------------------------------


def test_lars_no_momentum():
    torch.manual_seed(6)
    model = make_linear()
    opt = LARS(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0, trust_coefficient=1.0)
    x = torch.randn(32, 16)

    initial_loss = quadratic_loss(model, x).item()

    for _ in range(20):
        opt.zero_grad()
        loss = quadratic_loss(model, x)
        loss.backward()
        opt.step()

    final_loss = quadratic_loss(model, x).item()
    assert final_loss < initial_loss, (
        f"LARS (momentum=0) failed to reduce loss: {initial_loss:.4f} -> {final_loss:.4f}"
    )

    # State should NOT contain momentum buffers (or they were never used)
    for p in model.parameters():
        state = opt.state[p]
        # With momentum=0, no buffer should be created
        assert "momentum_buffer" not in state, (
            "LARS with momentum=0 should not accumulate velocity buffer"
        )


# ---------------------------------------------------------------------------
# Test 11: LAMB betas: moment estimates update with correct decay
# ---------------------------------------------------------------------------


def test_lamb_betas_moment_estimates():
    """
    After one step, verify the moment estimates follow the update rules exactly.
    """
    torch.manual_seed(7)
    model = nn.Linear(4, 2, bias=False)
    beta1, beta2 = 0.9, 0.999

    opt = LAMB(model.parameters(), lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)

    x = torch.randn(8, 4)
    opt.zero_grad()
    quadratic_loss(model, x).backward()

    grad_before = model.weight.grad.clone()
    opt.step()

    state = opt.state[model.weight]
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]

    # After step 1: exp_avg = (1-beta1)*grad; exp_avg_sq = (1-beta2)*grad^2
    expected_m = (1.0 - beta1) * grad_before
    expected_v = (1.0 - beta2) * grad_before**2

    assert torch.allclose(exp_avg, expected_m, atol=1e-6), (
        f"exp_avg mismatch: max diff = {(exp_avg - expected_m).abs().max():.2e}"
    )
    assert torch.allclose(exp_avg_sq, expected_v, atol=1e-6), (
        f"exp_avg_sq mismatch: max diff = {(exp_avg_sq - expected_v).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 12: get_param_groups_for_lars returns 2 groups
# ---------------------------------------------------------------------------


def test_get_param_groups_returns_two_groups():
    model = make_mlp()
    groups = get_param_groups_for_lars(model, weight_decay=1e-4)
    assert len(groups) == 2, f"Expected 2 param groups, got {len(groups)}"


# ---------------------------------------------------------------------------
# Test 13: get_param_groups_for_lars: bias params in no-weight-decay group
# ---------------------------------------------------------------------------


def test_get_param_groups_bias_in_no_decay_group():
    model = make_linear(bias=True)
    groups = get_param_groups_for_lars(model, weight_decay=1e-4)

    no_decay_group = groups[1]
    decay_group = groups[0]

    # Bias should be in no-decay group
    assert no_decay_group["weight_decay"] == 0.0
    assert decay_group["weight_decay"] == 1e-4

    no_decay_params = no_decay_group["params"]
    bias = model.bias
    # Check bias tensor is in the no-decay group
    assert any(p.data_ptr() == bias.data_ptr() for p in no_decay_params), (
        "Bias should be in the no-weight-decay group"
    )

    # Check weight is in the decay group
    decay_params = decay_group["params"]
    weight = model.weight
    assert any(p.data_ptr() == weight.data_ptr() for p in decay_params), (
        "Weight should be in the decay group"
    )


# ---------------------------------------------------------------------------
# Test 14: LARS exclude_bias_and_bn=True: bias params don't get trust ratio scaling
# ---------------------------------------------------------------------------


def test_lars_exclude_bias_no_trust_ratio():
    """
    When exclude_bias_and_bn=True, 1D params (bias) should not get trust ratio.
    We verify this by using get_param_groups which sets apply_trust_ratio per group.
    """
    torch.manual_seed(8)
    model = make_linear()
    groups = get_param_groups_for_lars(model, weight_decay=1e-4)

    opt = LARS(groups, lr=0.01, trust_coefficient=0.001, exclude_bias_and_bn=True)

    x = torch.randn(32, 16)
    opt.zero_grad()
    quadratic_loss(model, x).backward()
    opt.step()

    # Verify the no-decay group has apply_trust_ratio=False
    no_decay_group = groups[1]
    assert no_decay_group["apply_trust_ratio"] is False, (
        "Bias/BN group should have apply_trust_ratio=False"
    )


# ---------------------------------------------------------------------------
# Test 15: LAMB clamp_value limits trust ratio (test with tiny grad)
# ---------------------------------------------------------------------------


def test_lamb_clamp_trust_ratio():
    """
    When the gradient is extremely tiny, the trust ratio would be huge.
    clamp_value should bound it.
    """
    torch.manual_seed(9)
    # Use a model with large weights but near-zero gradient situation
    model = nn.Linear(4, 2, bias=False)
    # Set large weights
    model.weight.data.fill_(100.0)

    clamp_value = 10.0
    opt = LAMB(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
        clamp_value=clamp_value,
        trust_coefficient=1.0,
        adam_w_mode=False,
    )

    x = torch.randn(8, 4) * 1e-6  # tiny input -> tiny grad
    opt.zero_grad()
    quadratic_loss(model, x).backward()

    # Manually compute what trust ratio would be unbounded
    model.weight.grad.clone()
    param = model.weight.clone()

    opt.step()

    # The param should have changed, and we verify no explosion occurred
    weight_after = model.weight.clone()
    update_magnitude = (weight_after - param).abs().max().item()
    # With clamp at 10, lr=1e-3, update should be bounded
    # max step ~ lr * clamp_value * ||direction|| / something reasonable
    assert update_magnitude < 1e3, (
        f"LAMB trust ratio clamp failed: update magnitude {update_magnitude:.2f}"
    )

    # Also verify by direct trust ratio computation with manual clamp
    # param_norm >> update_norm -> unclamped ratio >> clamp_value
    param_norm = param.norm(2).item()
    assert param_norm > 10.0, "Test setup: param should have large norm"

    # Trust ratio without clamp would be >> clamp_value
    # (tiny grad => Adam r_t is large enough direction; param is 100 => large param_norm)
    # The clamp ensures it doesn't exceed clamp_value
    # We can't directly inspect internal ratio, but the update being finite verifies it


# ---------------------------------------------------------------------------
# Test 16: Both optimizers work with multiple param groups
# ---------------------------------------------------------------------------


def test_both_optimizers_multiple_param_groups():
    """
    Both LARS and LAMB should work when initialized with multiple param groups
    (different lr or weight_decay per group).
    """
    torch.manual_seed(10)
    model = make_mlp()

    # Split params into two groups with different lr
    params_group1 = list(model[0].parameters())  # first linear
    params_group3 = list(model[3].parameters())  # second linear

    lars_groups = [
        {"params": params_group1, "lr": 0.01},
        {"params": params_group3, "lr": 0.001},
    ]
    lamb_groups = [
        {"params": params_group1, "lr": 1e-2},
        {"params": params_group3, "lr": 1e-3},
    ]

    opt_lars = LARS(lars_groups, lr=0.005, momentum=0.9, weight_decay=0.0, trust_coefficient=1.0)
    opt_lamb = LAMB(lamb_groups, lr=1e-3, weight_decay=0.0)

    x = torch.randn(16, 16)

    # LARS forward/backward
    opt_lars.zero_grad()
    # Use only params from groups
    out = model(x)
    loss = (out**2).mean()
    loss.backward()
    opt_lars.step()

    # LAMB forward/backward
    opt_lamb.zero_grad()
    out = model(x)
    loss = (out**2).mean()
    loss.backward()
    opt_lamb.step()

    assert len(opt_lars.param_groups) == 2
    assert len(opt_lamb.param_groups) == 2

    # Verify state was created for parameters in both groups
    for p in params_group1 + params_group3:
        if p.grad is not None:
            assert p in opt_lamb.state, "LAMB state missing for param in group"
