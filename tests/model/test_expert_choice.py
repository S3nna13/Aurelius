"""Tests for Expert Choice MoE routing (Zhou et al., 2022)."""
import math

import pytest
import torch

from src.model.expert_choice import (
    BalancedMoEFFN,
    ExpertChoiceConfig,
    ExpertChoiceFFN,
    ExpertChoiceRouter,
    expert_choice_aux_loss,
)

# ---------------------------------------------------------------------------
# Common test parameters
# ---------------------------------------------------------------------------
B = 2
T = 16
D = 64
D_FF = 128
N_EXPERTS = 4
CAPACITY_FACTOR = 1.5


def make_config(**kwargs) -> ExpertChoiceConfig:
    defaults = dict(
        n_experts=N_EXPERTS,
        capacity_factor=CAPACITY_FACTOR,
        d_model=D,
        d_ff=D_FF,
        use_bias=False,
    )
    defaults.update(kwargs)
    return ExpertChoiceConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expert_choice_config_defaults():
    """ExpertChoiceConfig should have the correct default values."""
    cfg = ExpertChoiceConfig()
    assert cfg.n_experts == 8
    assert cfg.capacity_factor == 1.25
    assert cfg.d_model == 64
    assert cfg.d_ff == 128
    assert cfg.use_bias is False


def test_router_indices_shape():
    """Router indices should have shape (E, capacity)."""
    torch.manual_seed(0)
    cfg = make_config()
    router = ExpertChoiceRouter(D, N_EXPERTS, CAPACITY_FACTOR)
    hidden = torch.randn(B, T, D)
    indices, weights, router_probs = router(hidden)
    N = B * T
    expected_capacity = math.ceil(CAPACITY_FACTOR * N / N_EXPERTS)
    assert indices.shape == (N_EXPERTS, expected_capacity), (
        f"Expected indices shape ({N_EXPERTS}, {expected_capacity}), got {indices.shape}"
    )


def test_router_weights_shape():
    """Router weights should have shape (E, capacity)."""
    torch.manual_seed(0)
    router = ExpertChoiceRouter(D, N_EXPERTS, CAPACITY_FACTOR)
    hidden = torch.randn(B, T, D)
    indices, weights, router_probs = router(hidden)
    N = B * T
    expected_capacity = math.ceil(CAPACITY_FACTOR * N / N_EXPERTS)
    assert weights.shape == (N_EXPERTS, expected_capacity), (
        f"Expected weights shape ({N_EXPERTS}, {expected_capacity}), got {weights.shape}"
    )


def test_router_capacity_formula():
    """Capacity should equal ceil(capacity_factor * N / n_experts)."""
    torch.manual_seed(0)
    router = ExpertChoiceRouter(D, N_EXPERTS, CAPACITY_FACTOR)
    hidden = torch.randn(B, T, D)
    indices, weights, router_probs = router(hidden)
    N = B * T
    expected_capacity = math.ceil(CAPACITY_FACTOR * N / N_EXPERTS)
    actual_capacity = indices.shape[1]
    assert actual_capacity == expected_capacity, (
        f"Expected capacity {expected_capacity}, got {actual_capacity}"
    )


def test_expert_choice_ffn_output_shape():
    """ExpertChoiceFFN output should have shape (B, T, D)."""
    torch.manual_seed(0)
    cfg = make_config()
    model = ExpertChoiceFFN(cfg)
    hidden = torch.randn(B, T, D)
    output, aux_loss = model(hidden)
    assert output.shape == (B, T, D), (
        f"Expected output shape ({B}, {T}, {D}), got {output.shape}"
    )


def test_expert_choice_ffn_aux_loss_scalar():
    """ExpertChoiceFFN aux_loss should be a scalar (0-d tensor)."""
    torch.manual_seed(0)
    cfg = make_config()
    model = ExpertChoiceFFN(cfg)
    hidden = torch.randn(B, T, D)
    output, aux_loss = model(hidden)
    assert aux_loss.ndim == 0, f"aux_loss should be a scalar, got shape {aux_loss.shape}"


def test_expert_choice_ffn_no_nan():
    """ExpertChoiceFFN output should not contain NaN values."""
    torch.manual_seed(0)
    cfg = make_config()
    model = ExpertChoiceFFN(cfg)
    hidden = torch.randn(B, T, D)
    output, aux_loss = model(hidden)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isnan(aux_loss), "aux_loss is NaN"


def test_expert_choice_aux_loss_negative():
    """aux_loss should be <= 0 (it's negative entropy)."""
    torch.manual_seed(0)
    N = B * T
    E = N_EXPERTS
    router_probs = torch.softmax(torch.randn(N, E), dim=-1)
    loss = expert_choice_aux_loss(router_probs)
    assert loss.item() <= 0.0, (
        f"aux_loss should be <= 0 (negative entropy), got {loss.item()}"
    )


def test_balanced_moe_dense_fallback():
    """BalancedMoEFFN should use dense fallback when B*T < min_tokens_for_moe."""
    torch.manual_seed(0)
    cfg = make_config()
    model = BalancedMoEFFN(cfg)
    # Use fewer tokens than n_experts to trigger dense fallback
    small_T = 1  # B*T = 2 < n_experts = 4
    hidden = torch.randn(B, small_T, D)
    output, aux_loss = model(hidden)
    assert output.shape == (B, small_T, D), (
        f"Dense fallback output shape wrong: {output.shape}"
    )
    # Dense fallback aux_loss should be zeros
    assert (aux_loss == 0.0).all(), "Dense fallback should return zero aux_loss"


def test_balanced_moe_expert_choice():
    """BalancedMoEFFN should use Expert Choice when B*T >= min_tokens_for_moe."""
    torch.manual_seed(0)
    cfg = make_config()
    model = BalancedMoEFFN(cfg)
    # B*T = 32 >= n_experts = 4
    hidden = torch.randn(B, T, D)
    output, aux_loss = model(hidden)
    assert output.shape == (B, T, D), (
        f"Expert choice output shape wrong: {output.shape}"
    )
    # aux_loss should be negative entropy (not 0)
    assert aux_loss.ndim == 0, "aux_loss should be scalar"


def test_token_utilization_sums_to_one():
    """Per-expert token fractions should sum to approximately capacity*E/N (each token can go to multiple experts)."""
    torch.manual_seed(0)
    cfg = make_config()
    model = ExpertChoiceFFN(cfg)
    hidden = torch.randn(B, T, D)
    with torch.no_grad():
        output, _ = model(hidden)
    utilization = model.token_utilization()
    assert len(utilization) == N_EXPERTS, (
        f"Expected {N_EXPERTS} entries in utilization, got {len(utilization)}"
    )
    # Each expert processes capacity tokens; total = E * capacity
    # As fractions of N, they sum to E * capacity / N ≈ capacity_factor
    total = sum(utilization.values())
    N = B * T
    expected_capacity = math.ceil(CAPACITY_FACTOR * N / N_EXPERTS)
    expected_total = N_EXPERTS * expected_capacity / N
    assert abs(total - expected_total) < 1e-5, (
        f"Utilization fractions sum to {total}, expected ~{expected_total}"
    )


def test_expert_choice_gradient_flow():
    """loss.backward() should complete without error (gradient flows through routing)."""
    torch.manual_seed(0)
    cfg = make_config()
    model = ExpertChoiceFFN(cfg)
    hidden = torch.randn(B, T, D, requires_grad=True)
    output, aux_loss = model(hidden)
    loss = output.sum() + aux_loss
    loss.backward()  # should not raise
    assert hidden.grad is not None, "No gradient flowed to input hidden"
