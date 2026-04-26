"""Tests for src/model/soft_moe.py — Soft Mixture-of-Experts."""

from __future__ import annotations

import pytest
import torch

from src.model.soft_moe import (
    ExpertFFN,
    SoftMoEConfig,
    SoftMoELayer,
    SoftRouter,
    compute_load_stats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, D = 2, 10, 64
N_EXPERTS = 4
N_SLOTS = 1
D_FF = 128


@pytest.fixture()
def config() -> SoftMoEConfig:
    return SoftMoEConfig(
        n_experts=N_EXPERTS,
        n_slots=N_SLOTS,
        d_model=D,
        d_ff=D_FF,
        dropout=0.1,
    )


@pytest.fixture()
def x() -> torch.Tensor:
    return torch.randn(B, T, D)


@pytest.fixture()
def expert() -> ExpertFFN:
    return ExpertFFN(d_model=D, d_ff=D_FF, dropout=0.1)


@pytest.fixture()
def router() -> SoftRouter:
    return SoftRouter(d_model=D, n_experts=N_EXPERTS, n_slots=N_SLOTS)


@pytest.fixture()
def layer(config) -> SoftMoELayer:
    return SoftMoELayer(config)


# ---------------------------------------------------------------------------
# 1. SoftMoEConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SoftMoEConfig()
    assert cfg.n_experts == 8
    assert cfg.n_slots == 1
    assert cfg.d_model == 64
    assert cfg.d_ff == 256
    assert cfg.dropout == 0.1


# ---------------------------------------------------------------------------
# 2. ExpertFFN output shape
# ---------------------------------------------------------------------------


def test_expert_ffn_output_shape(expert, x):
    out = expert(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. SoftRouter dispatch_weights shape
# ---------------------------------------------------------------------------


def test_router_dispatch_shape(router, x):
    dispatch, _ = router(x)
    expected = (B, T, N_EXPERTS * N_SLOTS)
    assert dispatch.shape == expected, f"Expected {expected}, got {dispatch.shape}"


# ---------------------------------------------------------------------------
# 4. SoftRouter combine_weights shape
# ---------------------------------------------------------------------------


def test_router_combine_shape(router, x):
    _, combine = router(x)
    expected = (B, T, N_EXPERTS * N_SLOTS)
    assert combine.shape == expected, f"Expected {expected}, got {combine.shape}"


# ---------------------------------------------------------------------------
# 5. SoftRouter dispatch softmax sums to 1 along dim=1 (over tokens)
# ---------------------------------------------------------------------------


def test_router_dispatch_sums_to_1_over_tokens(router, x):
    dispatch, _ = router(x)
    # Sum over token dim → (B, E*S), should be all ~1
    sums = dispatch.sum(dim=1)  # (B, E*S)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        "dispatch_weights must sum to 1 over the token dimension (dim=1)"
    )


# ---------------------------------------------------------------------------
# 6. SoftRouter combine softmax sums to 1 along dim=2 (over slots)
# ---------------------------------------------------------------------------


def test_router_combine_sums_to_1_over_slots(router, x):
    _, combine = router(x)
    # Sum over slot dim → (B, T), should be all ~1
    sums = combine.sum(dim=2)  # (B, T)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        "combine_weights must sum to 1 over the slot dimension (dim=2)"
    )


# ---------------------------------------------------------------------------
# 7. SoftMoELayer output shape
# ---------------------------------------------------------------------------


def test_soft_moe_layer_output_shape(layer, x):
    out = layer(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 8. SoftMoELayer output is differentiable (backward works)
# ---------------------------------------------------------------------------


def test_soft_moe_layer_backward(layer, x):
    x_req = x.requires_grad_(True)
    out = layer(x_req)
    loss = out.sum()
    loss.backward()
    assert x_req.grad is not None, "Gradient w.r.t. input should not be None"
    assert x_req.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 9. SoftMoELayer output differs from input (transformation is applied)
# ---------------------------------------------------------------------------


def test_soft_moe_layer_transforms_input(layer, x):
    with torch.no_grad():
        out = layer(x)
    assert not torch.allclose(out, x, atol=1e-6), "SoftMoELayer output should differ from the input"


# ---------------------------------------------------------------------------
# 10. compute_load_stats returns required keys
# ---------------------------------------------------------------------------


def test_compute_load_stats_keys(router, x):
    dispatch, _ = router(x)
    stats = compute_load_stats(dispatch)
    for key in ("expert_load", "load_balance_loss", "max_load", "min_load"):
        assert key in stats, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 11. compute_load_stats expert_load sums to B*T
# ---------------------------------------------------------------------------


def test_compute_load_stats_expert_load_sum(router, x):
    dispatch, _ = router(x)
    stats = compute_load_stats(dispatch)
    total = stats["expert_load"].sum().item()
    # dispatch_weights are softmax over *tokens* (dim=1), so each slot column
    # sums to 1 per batch item.  Summing over batch and tokens gives:
    #   sum_{b,t,s} dispatch[b,t,s] = B * (E*S)   (one unit per batch-slot pair)
    expected = B * N_EXPERTS * N_SLOTS
    assert abs(total - expected) < 1e-4, (
        f"expert_load should sum to B*(E*S)={expected}, got {total}"
    )


# ---------------------------------------------------------------------------
# 12. compute_load_stats load_balance_loss >= 0
# ---------------------------------------------------------------------------


def test_compute_load_stats_loss_nonneg(router, x):
    dispatch, _ = router(x)
    stats = compute_load_stats(dispatch)
    assert stats["load_balance_loss"].item() >= 0.0, (
        "load_balance_loss (variance) must be non-negative"
    )


# ---------------------------------------------------------------------------
# 13. SoftMoELayer works with n_slots=2
# ---------------------------------------------------------------------------


def test_soft_moe_layer_multi_slot():
    cfg = SoftMoEConfig(n_experts=N_EXPERTS, n_slots=2, d_model=D, d_ff=D_FF, dropout=0.0)
    layer = SoftMoELayer(cfg)
    x = torch.randn(B, T, D)
    out = layer(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 14. ExpertFFN dropout affects output in train vs eval mode
# ---------------------------------------------------------------------------


def test_expert_ffn_dropout_train_vs_eval():
    torch.manual_seed(0)
    expert_module = ExpertFFN(d_model=D, d_ff=D_FF, dropout=0.5)

    x = torch.randn(4, 16, D)

    # Collect outputs in train mode (stochastic due to dropout)
    expert_module.train()
    outputs_train = set()
    for _ in range(10):
        with torch.no_grad():
            out = expert_module(x)
        outputs_train.add(out.sum().item())

    # Collect outputs in eval mode (deterministic, dropout disabled)
    expert_module.eval()
    outputs_eval = set()
    for _ in range(5):
        with torch.no_grad():
            out = expert_module(x)
        outputs_eval.add(out.sum().item())

    # In eval mode all runs produce identical output
    assert len(outputs_eval) == 1, "ExpertFFN should be deterministic in eval mode"
    # In train mode with high dropout (0.5) we expect variation across runs
    assert len(outputs_train) > 1, (
        "ExpertFFN with dropout=0.5 should produce varied outputs in train mode"
    )
