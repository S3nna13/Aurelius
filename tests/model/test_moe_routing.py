"""
Tests for src/model/moe_routing.py

Tiny config: N_EXPERTS=4, TOP_K=2, D=8, B=2, T=4  (B*T=8 tokens)
"""

import math
import pytest
import torch

from src.model.moe_routing import (
    RoutingConfig,
    topk_routing,
    expert_choice_routing,
    compute_router_z_loss,
    compute_load_balance_loss,
    Router,
    SwitchRouter,
)

# ---------------------------------------------------------------------------
# Tiny test constants
# ---------------------------------------------------------------------------
N_EXPERTS = 4
TOP_K = 2
D = 8
B = 2
T = 4
S = B * T  # 8 tokens

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_logits(s=S, n=N_EXPERTS) -> torch.Tensor:
    return torch.randn(s, n)


def make_input(b=B, t=T, d=D) -> torch.Tensor:
    return torch.randn(b, t, d)


def make_config(**kwargs) -> RoutingConfig:
    return RoutingConfig(n_experts=N_EXPERTS, top_k=TOP_K, **kwargs)


# ---------------------------------------------------------------------------
# 1. RoutingConfig defaults
# ---------------------------------------------------------------------------

def test_routing_config_defaults():
    cfg = RoutingConfig()
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.router_type == "topk"
    assert cfg.noise_std == 0.0
    assert cfg.capacity_factor == 1.25
    assert cfg.aux_loss_coef == 0.01


# ---------------------------------------------------------------------------
# 2. topk_routing — expert_indices shape
# ---------------------------------------------------------------------------

def test_topk_routing_expert_indices_shape():
    logits = make_logits()
    indices, _ = topk_routing(logits, k=TOP_K)
    assert indices.shape == (S, TOP_K), f"Expected ({S}, {TOP_K}), got {indices.shape}"


# ---------------------------------------------------------------------------
# 3. topk_routing — gates shape
# ---------------------------------------------------------------------------

def test_topk_routing_gates_shape():
    logits = make_logits()
    _, gates = topk_routing(logits, k=TOP_K)
    assert gates.shape == (S, TOP_K), f"Expected ({S}, {TOP_K}), got {gates.shape}"


# ---------------------------------------------------------------------------
# 4. topk_routing — gates sum to ~1 per token (softmax over top-k)
# ---------------------------------------------------------------------------

def test_topk_routing_gates_sum_to_one():
    logits = make_logits()
    _, gates = topk_routing(logits, k=TOP_K)
    sums = gates.sum(dim=-1)  # (S,)
    assert torch.allclose(sums, torch.ones(S), atol=1e-5), \
        f"Gates don't sum to 1; got min={sums.min():.6f}, max={sums.max():.6f}"


# ---------------------------------------------------------------------------
# 5. topk_routing — indices are valid expert indices
# ---------------------------------------------------------------------------

def test_topk_routing_indices_valid_range():
    logits = make_logits()
    indices, _ = topk_routing(logits, k=TOP_K)
    assert indices.min() >= 0
    assert indices.max() < N_EXPERTS


# ---------------------------------------------------------------------------
# 6. topk_routing with noise produces different indices from without noise
# ---------------------------------------------------------------------------

def test_topk_routing_noise_differs():
    torch.manual_seed(0)
    logits = make_logits()
    # Run without noise
    idx_no_noise, _ = topk_routing(logits, k=TOP_K, noise_std=0.0)
    # Run with large noise — very likely to produce different selections
    torch.manual_seed(1)
    idx_noise, _ = topk_routing(logits, k=TOP_K, noise_std=10.0)
    # With noise_std=10 on logits of order ~1, at least some assignments differ
    assert not torch.equal(idx_no_noise, idx_noise), \
        "Expected noise to change at least some expert assignments"


# ---------------------------------------------------------------------------
# 7. expert_choice_routing — token_indices shape (n_experts, capacity)
# ---------------------------------------------------------------------------

def test_expert_choice_routing_token_indices_shape():
    logits = make_logits()
    capacity = 3
    token_indices, _ = expert_choice_routing(logits, capacity=capacity)
    assert token_indices.shape == (N_EXPERTS, capacity), \
        f"Expected ({N_EXPERTS}, {capacity}), got {token_indices.shape}"


# ---------------------------------------------------------------------------
# 8. expert_choice_routing — gates shape (n_experts, capacity)
# ---------------------------------------------------------------------------

def test_expert_choice_routing_gates_shape():
    logits = make_logits()
    capacity = 3
    _, gates = expert_choice_routing(logits, capacity=capacity)
    assert gates.shape == (N_EXPERTS, capacity), \
        f"Expected ({N_EXPERTS}, {capacity}), got {gates.shape}"


# ---------------------------------------------------------------------------
# 9. expert_choice_routing — gates sum to ~1 per expert (softmax per expert)
# ---------------------------------------------------------------------------

def test_expert_choice_routing_gates_sum_to_one():
    logits = make_logits()
    capacity = 3
    _, gates = expert_choice_routing(logits, capacity=capacity)
    sums = gates.sum(dim=-1)  # (n_experts,)
    assert torch.allclose(sums, torch.ones(N_EXPERTS), atol=1e-5), \
        f"Expert gates don't sum to 1 per expert; sums={sums}"


# ---------------------------------------------------------------------------
# 10. compute_router_z_loss — scalar and positive
# ---------------------------------------------------------------------------

def test_router_z_loss_scalar_and_positive():
    logits = make_logits()
    loss = compute_router_z_loss(logits, coef=1e-3)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, f"Z-loss should be non-negative, got {loss.item()}"


# ---------------------------------------------------------------------------
# 11. compute_load_balance_loss — scalar and non-negative
# ---------------------------------------------------------------------------

def test_load_balance_loss_scalar_and_nonneg():
    probs = torch.softmax(make_logits(), dim=-1)  # (S, n_experts)
    loss = compute_load_balance_loss(probs, n_experts=N_EXPERTS, coef=0.01)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, f"Load balance loss should be non-negative, got {loss.item()}"


# ---------------------------------------------------------------------------
# 12. Router — output expert_indices shape
# ---------------------------------------------------------------------------

def test_router_expert_indices_shape():
    cfg = make_config()
    router = Router(d_model=D, config=cfg)
    x = make_input()
    expert_indices, gates, aux_loss = router(x)
    assert expert_indices.shape == (S, TOP_K), \
        f"Expected ({S}, {TOP_K}), got {expert_indices.shape}"


# ---------------------------------------------------------------------------
# 13. Router — output gates shape
# ---------------------------------------------------------------------------

def test_router_gates_shape():
    cfg = make_config()
    router = Router(d_model=D, config=cfg)
    x = make_input()
    expert_indices, gates, aux_loss = router(x)
    assert gates.shape == (S, TOP_K), \
        f"Expected ({S}, {TOP_K}), got {gates.shape}"


# ---------------------------------------------------------------------------
# 14. Router — aux_loss is finite scalar
# ---------------------------------------------------------------------------

def test_router_aux_loss_finite_scalar():
    cfg = make_config()
    router = Router(d_model=D, config=cfg)
    x = make_input()
    _, _, aux_loss = router(x)
    assert aux_loss.shape == (), f"Expected scalar aux_loss, got shape {aux_loss.shape}"
    assert torch.isfinite(aux_loss), f"aux_loss is not finite: {aux_loss.item()}"


# ---------------------------------------------------------------------------
# 15. SwitchRouter — expert_idx shape (B*T,)
# ---------------------------------------------------------------------------

def test_switch_router_expert_idx_shape():
    cfg = make_config()
    router = SwitchRouter(d_model=D, config=cfg)
    x = make_input()
    expert_idx, gates, aux_loss = router(x)
    assert expert_idx.shape == (S,), \
        f"Expected ({S},), got {expert_idx.shape}"


# ---------------------------------------------------------------------------
# 16. SwitchRouter — gates finite and shape (B*T,)
# ---------------------------------------------------------------------------

def test_switch_router_gates_finite():
    cfg = make_config()
    router = SwitchRouter(d_model=D, config=cfg)
    x = make_input()
    expert_idx, gates, aux_loss = router(x)
    assert gates.shape == (S,), f"Expected ({S},), got {gates.shape}"
    assert torch.all(torch.isfinite(gates)), "SwitchRouter gates contain non-finite values"


# ---------------------------------------------------------------------------
# 17. SwitchRouter — aux_loss finite scalar
# ---------------------------------------------------------------------------

def test_switch_router_aux_loss_finite():
    cfg = make_config()
    router = SwitchRouter(d_model=D, config=cfg)
    x = make_input()
    _, _, aux_loss = router(x)
    assert aux_loss.shape == (), f"Expected scalar, got {aux_loss.shape}"
    assert torch.isfinite(aux_loss), f"SwitchRouter aux_loss not finite: {aux_loss.item()}"


# ---------------------------------------------------------------------------
# 18. SwitchRouter — capacity dropping: no more than capacity tokens per expert
# ---------------------------------------------------------------------------

def test_switch_router_capacity_respected():
    cfg = make_config(capacity_factor=1.0)
    router = SwitchRouter(d_model=D, config=cfg)
    x = make_input()
    expert_idx, gates, _ = router(x)
    capacity = math.ceil(S / N_EXPERTS * cfg.capacity_factor)
    for e in range(N_EXPERTS):
        # Tokens assigned to expert e with non-zero gates
        active = ((expert_idx == e) & (gates > 0)).sum().item()
        assert active <= capacity, \
            f"Expert {e} has {active} active tokens but capacity is {capacity}"


# ---------------------------------------------------------------------------
# 19. topk_routing — gates are non-negative
# ---------------------------------------------------------------------------

def test_topk_routing_gates_nonneg():
    logits = make_logits()
    _, gates = topk_routing(logits, k=TOP_K)
    assert (gates >= 0).all(), "All gates should be non-negative (softmax output)"


# ---------------------------------------------------------------------------
# 20. Router — gates sum to ~1 per token
# ---------------------------------------------------------------------------

def test_router_gates_sum_to_one():
    cfg = make_config()
    router = Router(d_model=D, config=cfg)
    x = make_input()
    _, gates, _ = router(x)
    sums = gates.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(S), atol=1e-5), \
        f"Router gates don't sum to 1; min={sums.min():.6f}, max={sums.max():.6f}"
