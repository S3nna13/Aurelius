"""Tests for Gradient Routing implementation.

Covers GradientRoutingConfig, RoutingRule, ModuleGradientMask, GradientRouter,
and the compute_gradient_conflict utility.
"""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.gradient_routing import (
    GradientRouter,
    GradientRoutingConfig,
    ModuleGradientMask,
    RoutingRule,
    compute_gradient_conflict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def tiny_linear():
    """Two-layer linear model for simple gradient tests."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 4),  # weight: fc0.weight / bias: fc0.bias  (named 0.weight etc.)
        nn.Linear(4, 2),
    )
    return model


# ---------------------------------------------------------------------------
# Test 1: GradientRoutingConfig defaults
# ---------------------------------------------------------------------------


def test_gradient_routing_config_defaults():
    cfg = GradientRoutingConfig()
    assert cfg.rules is None
    assert cfg.default_scale == 1.0
    assert cfg.log_gradient_norms is False


# ---------------------------------------------------------------------------
# Test 2: RoutingRule has all required fields
# ---------------------------------------------------------------------------


def test_routing_rule_fields():
    rule = RoutingRule()
    assert hasattr(rule, "data_tags")
    assert hasattr(rule, "allowed_modules")
    assert hasattr(rule, "blocked_modules")
    assert hasattr(rule, "gradient_scale")
    assert isinstance(rule.data_tags, list)
    assert isinstance(rule.allowed_modules, list)
    assert isinstance(rule.blocked_modules, list)
    assert rule.gradient_scale == 1.0


# ---------------------------------------------------------------------------
# Test 3: ModuleGradientMask.apply zeroes out blocked params' gradients
# ---------------------------------------------------------------------------


def test_module_gradient_mask_zeroes_blocked(tiny_linear):
    model = tiny_linear
    # Zero-out first layer's weight gradient
    mask = {"0.weight": 0.0, "0.bias": 0.0}
    mgm = ModuleGradientMask(model, mask)

    x = torch.randn(2, 8)
    loss = model(x).sum()
    mgm.apply(loss)

    assert model[0].weight.grad is not None
    assert model[0].weight.grad.abs().max().item() == 0.0, (
        "Blocked parameter should have zero gradient"
    )
    assert model[0].bias.grad.abs().max().item() == 0.0, "Blocked bias should have zero gradient"
    # Second layer should still have gradients
    assert model[1].weight.grad.abs().max().item() > 0.0, "Unmasked layer should retain gradients"


# ---------------------------------------------------------------------------
# Test 4: ModuleGradientMask.apply scales allowed params correctly
# ---------------------------------------------------------------------------


def test_module_gradient_mask_scales_allowed(tiny_linear):
    model = tiny_linear
    scale = 0.5

    # First pass: get reference gradients without masking
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(7))
    model.zero_grad()
    loss_ref = model(x).sum()
    loss_ref.backward()
    ref_grad = model[1].weight.grad.clone()

    # Second pass: apply scale mask to second layer
    model.zero_grad()
    mask = {"1.weight": scale}
    mgm = ModuleGradientMask(model, mask)
    loss2 = model(x).sum()
    mgm.apply(loss2)

    scaled_grad = model[1].weight.grad
    assert torch.allclose(scaled_grad, ref_grad * scale, atol=1e-6), (
        "Scaled gradient should be exactly scale * original gradient"
    )


# ---------------------------------------------------------------------------
# Test 5: get_masked_params returns correct list
# ---------------------------------------------------------------------------


def test_get_masked_params(tiny_linear):
    mask = {"0.weight": 0.0, "0.bias": 0.5, "1.weight": 1.0, "1.bias": 0.0}
    mgm = ModuleGradientMask(tiny_linear, mask)
    masked = mgm.get_masked_params()
    assert set(masked) == {"0.weight", "0.bias", "1.bias"}, (
        "Only params with scale < 1.0 should be returned"
    )
    assert "1.weight" not in masked


# ---------------------------------------------------------------------------
# Test 6: GradientRouter instantiates with rules
# ---------------------------------------------------------------------------


def test_gradient_router_instantiation(tiny_linear):
    rules = [
        RoutingRule(data_tags=["harmful"], blocked_modules=["0.*"]),
        RoutingRule(data_tags=["safe"], gradient_scale=1.0),
    ]
    router = GradientRouter(tiny_linear, rules)
    assert router.model is tiny_linear
    assert len(router.routing_rules) == 2


# ---------------------------------------------------------------------------
# Test 7: get_routing_summary returns dict with n_rules key
# ---------------------------------------------------------------------------


def test_get_routing_summary(tiny_linear):
    rules = [
        RoutingRule(data_tags=["harmful"], blocked_modules=["0.*"]),
        RoutingRule(data_tags=["safe"]),
    ]
    router = GradientRouter(tiny_linear, rules)
    summary = router.get_routing_summary()
    assert isinstance(summary, dict)
    assert "n_rules" in summary
    assert summary["n_rules"] == 2
    assert "blocked_params_per_tag" in summary
    assert "active_tags" in summary


# ---------------------------------------------------------------------------
# Test 8: apply_routing with unmatched tag doesn't block gradients
# ---------------------------------------------------------------------------


def test_apply_routing_unmatched_tag(tiny_linear):
    model = tiny_linear
    # Rule only applies to 'harmful' tag
    rules = [RoutingRule(data_tags=["harmful"], blocked_modules=["0.*"])]
    router = GradientRouter(model, rules)

    model.zero_grad()
    x = torch.randn(2, 8)
    loss = model(x).sum()
    router.apply_routing(loss, data_tag="benign")  # does not match any rule

    # All gradients should be non-zero (or at least present and unmodified)
    assert model[0].weight.grad is not None
    assert model[0].weight.grad.abs().max().item() > 0.0, (
        "Unmatched tag should not block any gradients"
    )


# ---------------------------------------------------------------------------
# Test 9: apply_routing with matching tag applies the rule
# ---------------------------------------------------------------------------


def test_apply_routing_matching_tag(tiny_linear):
    model = tiny_linear
    # Block first layer for 'harmful' data
    rules = [RoutingRule(data_tags=["harmful"], blocked_modules=["0.*"])]
    router = GradientRouter(model, rules)

    model.zero_grad()
    x = torch.randn(2, 8)
    loss = model(x).sum()
    router.apply_routing(loss, data_tag="harmful")

    # First layer gradients should be zeroed
    assert model[0].weight.grad.abs().max().item() == 0.0, (
        "Blocked layer gradient should be zero for matching tag"
    )
    assert model[0].bias.grad.abs().max().item() == 0.0, (
        "Blocked layer bias gradient should be zero for matching tag"
    )
    # Second layer should be unaffected
    assert model[1].weight.grad.abs().max().item() > 0.0, "Unblocked layer should retain gradients"


# ---------------------------------------------------------------------------
# Test 10: remove_hooks clears all hooks
# ---------------------------------------------------------------------------


def test_remove_hooks(tiny_linear):
    rules = [RoutingRule(data_tags=["harmful"], blocked_modules=["0.*"])]
    router = GradientRouter(tiny_linear, rules)
    router.register_hooks()
    assert len(router._hooks) > 0, "Hooks should be registered"

    router.remove_hooks()
    assert len(router._hooks) == 0, "All hooks should be removed after remove_hooks()"


# ---------------------------------------------------------------------------
# Test 11: compute_gradient_conflict returns value in [-1, 1]
# ---------------------------------------------------------------------------


def test_compute_gradient_conflict_range():
    torch.manual_seed(99)
    grads1 = [torch.randn(4, 4), torch.randn(4)]
    grads2 = [torch.randn(4, 4), torch.randn(4)]
    result = compute_gradient_conflict(grads1, grads2)
    assert -1.0 <= result <= 1.0, f"Expected result in [-1, 1], got {result}"


# ---------------------------------------------------------------------------
# Test 12: Identical gradient lists have cosine similarity = 1 (conflict ~ -1)
# ---------------------------------------------------------------------------


def test_compute_gradient_conflict_identical():
    """Identical gradients => cosine similarity = 1.0 (no conflict)."""
    torch.manual_seed(0)
    grads = [torch.randn(3, 3), torch.randn(3)]
    # Identical gradient lists should have cosine similarity = 1.0
    result = compute_gradient_conflict(grads, grads)
    assert abs(result - 1.0) < 1e-5, (
        f"Identical gradients should have cosine similarity ~1.0, got {result}"
    )
