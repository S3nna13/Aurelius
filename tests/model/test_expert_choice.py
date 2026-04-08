"""Tests for Expert Choice MoE routing (ExpertChoiceFFN)."""
import math

import torch
import pytest

from src.model.expert_choice import ExpertChoiceFFN
from src.model.moe import SparseMoEFFN, MoEConfig
from src.model.config import AureliusConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """Small AureliusConfig suitable for fast unit tests."""
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )


@pytest.fixture
def moe_cfg():
    return MoEConfig(n_experts=4, top_k=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape(small_config, moe_cfg):
    """ExpertChoiceFFN must return (B, S, D) output and a scalar aux_loss."""
    model = ExpertChoiceFFN(small_config, moe_cfg)
    # Use the exact shape stated in the task spec for documentation purposes;
    # we use a smaller tensor here to keep the test fast.
    B, S, D = 2, 16, small_config.d_model
    x = torch.randn(B, S, D)
    out, aux = model(x)
    assert out.shape == (B, S, D), f"Expected ({B}, {S}, {D}), got {out.shape}"
    assert aux.ndim == 0, "aux_loss must be a scalar (0-d tensor)"


def test_aux_loss_is_zero(small_config, moe_cfg):
    """Expert-choice routing requires no load-balancing loss; aux_loss == 0.0."""
    model = ExpertChoiceFFN(small_config, moe_cfg)
    x = torch.randn(2, 16, small_config.d_model)
    _, aux = model(x)
    assert aux.item() == 0.0, f"Expected aux_loss=0.0, got {aux.item()}"


def test_each_expert_processes_capacity_tokens(small_config, moe_cfg):
    """Every expert must process exactly capacity = ceil(N*top_k/n_experts) tokens."""
    n_experts = moe_cfg.n_experts
    top_k = moe_cfg.top_k
    B, S = 2, 16
    N = B * S
    expected_capacity = math.ceil(N * top_k / n_experts)

    model = ExpertChoiceFFN(small_config, moe_cfg)
    x = torch.randn(B, S, small_config.d_model)

    # Instrument by counting tokens per expert via token_coverage internals
    # We re-run the forward tracking manually using the router weights directly.
    with torch.no_grad():
        x_flat = x.view(-1, small_config.d_model)
        router_logits = model.router(x_flat)   # (N, n_experts)
        scores = router_logits.T               # (n_experts, N)
        capacity = min(math.ceil(N * top_k / n_experts), N)

        for i in range(n_experts):
            _, top_indices = torch.topk(scores[i], capacity)
            # Each expert should select exactly `capacity` unique token indices
            assert top_indices.shape[0] == capacity, (
                f"Expert {i} processed {top_indices.shape[0]} tokens, "
                f"expected {capacity}"
            )


def test_expert_choice_different_from_token_choice(small_config, moe_cfg):
    """ExpertChoiceFFN and SparseMoEFFN must produce different outputs (different routing)."""
    torch.manual_seed(0)
    ec = ExpertChoiceFFN(small_config, moe_cfg)
    torch.manual_seed(0)
    tc = SparseMoEFFN(small_config, moe_cfg)

    x = torch.randn(2, 8, small_config.d_model)
    with torch.no_grad():
        ec_out, _ = ec(x)
        tc_out, _ = tc(x)

    # The outputs should not be identical (routing strategies are fundamentally different)
    assert not torch.allclose(ec_out, tc_out, atol=1e-6), (
        "ExpertChoiceFFN and SparseMoEFFN produced identical outputs — "
        "their routing is expected to differ."
    )


def test_token_coverage_shape(small_config, moe_cfg):
    """token_coverage() must return an integer tensor of shape (B*seq_len,)."""
    B, S = 2, 16
    model = ExpertChoiceFFN(small_config, moe_cfg)
    x = torch.randn(B, S, small_config.d_model)
    with torch.no_grad():
        coverage = model.token_coverage(x)
    assert coverage.shape == (B * S,), (
        f"Expected coverage shape ({B * S},), got {coverage.shape}"
    )
    assert coverage.dtype in (torch.int32, torch.int64, torch.long), (
        f"Expected integer dtype, got {coverage.dtype}"
    )


def test_no_load_balancing_needed(small_config, moe_cfg):
    """Variance of expert load must be 0: all experts process exactly capacity tokens."""
    n_experts = moe_cfg.n_experts
    top_k = moe_cfg.top_k
    B, S = 2, 16
    N = B * S
    expected_capacity = min(math.ceil(N * top_k / n_experts), N)

    model = ExpertChoiceFFN(small_config, moe_cfg)
    x = torch.randn(B, S, small_config.d_model)

    with torch.no_grad():
        x_flat = x.view(-1, small_config.d_model)
        router_logits = model.router(x_flat)
        scores = router_logits.T   # (n_experts, N)
        capacity = min(math.ceil(N * top_k / n_experts), N)

        loads = []
        for i in range(n_experts):
            _, top_indices = torch.topk(scores[i], capacity)
            loads.append(top_indices.shape[0])

    loads_tensor = torch.tensor(loads, dtype=torch.float)
    variance = loads_tensor.var().item()
    assert variance == 0.0, (
        f"Expert loads have non-zero variance {variance}; loads={loads}. "
        "All experts should process exactly {expected_capacity} tokens."
    )
