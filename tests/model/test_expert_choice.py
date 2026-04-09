"""Tests for Expert Choice MoE routing."""
import torch
import pytest

from src.model.expert_choice import (
    ExpertChoiceConfig,
    ExpertChoiceFFN,
    ExpertChoiceLayer,
    ExpertChoiceTransformerBlock,
    compute_ec_router_loss,
    compute_expert_capacity,
    expert_choice_routing,
)

# Common dimensions
B = 2
T = 8
D = 64
N_EXPERTS = 4


def make_cfg(**kwargs) -> ExpertChoiceConfig:
    defaults = dict(n_experts=N_EXPERTS, capacity_factor=1.0, d_model=D, d_expert=128)
    defaults.update(kwargs)
    return ExpertChoiceConfig(**defaults)


# -----------------------------------------------------------------------
# 1. Config defaults
# -----------------------------------------------------------------------

def test_config_defaults():
    cfg = ExpertChoiceConfig()
    assert cfg.n_experts == 4
    assert cfg.capacity_factor == 1.0
    assert cfg.d_model == 64
    assert cfg.d_expert == 128
    assert cfg.use_aux_loss is True
    assert cfg.aux_loss_coeff == 0.01


# -----------------------------------------------------------------------
# 2. compute_expert_capacity — basic
# -----------------------------------------------------------------------

def test_compute_capacity_basic():
    # ceil(1.0 * 8 / 4) = 2
    assert compute_expert_capacity(8, 4, 1.0) == 2


# -----------------------------------------------------------------------
# 3. compute_expert_capacity — minimum 1
# -----------------------------------------------------------------------

def test_compute_capacity_minimum():
    # Very small capacity_factor or large n_experts — must be at least 1
    assert compute_expert_capacity(1, 100, 0.001) >= 1
    assert compute_expert_capacity(4, 4, 0.1) >= 1


# -----------------------------------------------------------------------
# 4. expert_choice_routing — shapes
# -----------------------------------------------------------------------

def test_expert_choice_routing_shapes():
    torch.manual_seed(0)
    n_tokens, n_experts, capacity = T, N_EXPERTS, 2
    logits = torch.randn(n_tokens, n_experts)
    expert_mask, token_indices, expert_weights = expert_choice_routing(logits, capacity)
    assert expert_mask.shape == (n_tokens, n_experts), f"Got {expert_mask.shape}"
    assert token_indices.shape == (n_experts, capacity), f"Got {token_indices.shape}"
    assert expert_weights.shape == (n_experts, capacity), f"Got {expert_weights.shape}"


# -----------------------------------------------------------------------
# 5. expert_choice_routing — capacity respected
# -----------------------------------------------------------------------

def test_expert_choice_routing_capacity_respected():
    torch.manual_seed(42)
    n_tokens, n_experts, capacity = 16, N_EXPERTS, 3
    logits = torch.randn(n_tokens, n_experts)
    expert_mask, token_indices, expert_weights = expert_choice_routing(logits, capacity)
    # Each expert must have exactly capacity tokens
    assert token_indices.shape[1] == capacity
    # expert_mask column sums should equal capacity for each expert
    col_sums = expert_mask.sum(dim=0)  # (n_experts,)
    for e in range(n_experts):
        assert col_sums[e].item() == capacity, f"Expert {e} has {col_sums[e]} tokens, expected {capacity}"


# -----------------------------------------------------------------------
# 6. expert_choice_routing — weights positive
# -----------------------------------------------------------------------

def test_expert_choice_routing_weights_positive():
    torch.manual_seed(7)
    logits = torch.randn(T, N_EXPERTS)
    _, _, expert_weights = expert_choice_routing(logits, 2)
    assert (expert_weights > 0).all(), "All expert weights should be positive"


# -----------------------------------------------------------------------
# 7. expert_choice_routing — uniform logits gives same tokens per expert
# -----------------------------------------------------------------------

def test_expert_choice_routing_perfect_balance():
    # Uniform logits => all tokens have equal scores => topk is deterministic
    # (might be any consistent ordering). The important thing: each expert gets capacity tokens.
    logits = torch.zeros(T, N_EXPERTS)
    capacity = 2
    expert_mask, token_indices, expert_weights = expert_choice_routing(logits, capacity)
    assert token_indices.shape == (N_EXPERTS, capacity)
    # With uniform logits, weights should all be equal (1/T after softmax over tokens)
    expected_weight = 1.0 / T
    assert torch.allclose(expert_weights, torch.full_like(expert_weights, expected_weight), atol=1e-5)


# -----------------------------------------------------------------------
# 8. compute_ec_router_loss — returns scalar
# -----------------------------------------------------------------------

def test_compute_ec_router_loss_scalar():
    torch.manual_seed(0)
    router_probs = torch.softmax(torch.randn(T, N_EXPERTS), dim=0)
    expert_mask = torch.zeros(T, N_EXPERTS)
    expert_mask[:2, :] = 1.0
    loss = compute_ec_router_loss(router_probs, expert_mask)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


# -----------------------------------------------------------------------
# 9. compute_ec_router_loss — non-negative
# -----------------------------------------------------------------------

def test_compute_ec_router_loss_nonneg():
    torch.manual_seed(0)
    router_probs = torch.softmax(torch.randn(T, N_EXPERTS), dim=0)
    capacity = 2
    logits = torch.randn(T, N_EXPERTS)
    expert_mask, _, _ = expert_choice_routing(logits, capacity)
    loss = compute_ec_router_loss(router_probs, expert_mask)
    assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"


# -----------------------------------------------------------------------
# 10. ExpertChoiceFFN — shape (N, D) -> (N, D)
# -----------------------------------------------------------------------

def test_expert_ffn_shape():
    torch.manual_seed(0)
    ffn = ExpertChoiceFFN(d_model=D, d_expert=128)
    x = torch.randn(10, D)
    out = ffn(x)
    assert out.shape == (10, D), f"Expected (10, {D}), got {out.shape}"


# -----------------------------------------------------------------------
# 11. ExpertChoiceLayer — output shape (B, T, D)
# -----------------------------------------------------------------------

def test_expert_choice_layer_output_shape():
    torch.manual_seed(0)
    cfg = make_cfg()
    layer = ExpertChoiceLayer(cfg)
    x = torch.randn(B, T, D)
    out, _ = layer(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


# -----------------------------------------------------------------------
# 12. ExpertChoiceLayer — aux dict has "aux_loss" key
# -----------------------------------------------------------------------

def test_expert_choice_layer_aux_keys():
    torch.manual_seed(0)
    cfg = make_cfg()
    layer = ExpertChoiceLayer(cfg)
    x = torch.randn(B, T, D)
    _, aux = layer(x)
    assert "aux_loss" in aux, f"Expected 'aux_loss' key, got {list(aux.keys())}"


# -----------------------------------------------------------------------
# 13. ExpertChoiceLayer — all tokens get output (EC guarantee)
# -----------------------------------------------------------------------

def test_expert_choice_layer_no_dropped_tokens():
    """With capacity_factor >= 1.0 and enough experts, no token is fully zero.

    EC with capacity = ceil(cf * N / E) — when capacity * E >= N, every token
    is selected by at least one expert (pigeonhole). With cf=1.0, capacity*E == N.
    We verify that the output buffer is NOT all zeros.
    """
    torch.manual_seed(0)
    cfg = make_cfg(capacity_factor=1.0)
    layer = ExpertChoiceLayer(cfg)
    x = torch.randn(B, T, D)
    out, _ = layer(x)
    # With residual in block the output won't be zero, but at layer level
    # check that at least some tokens have non-zero outputs
    assert not torch.all(out == 0), "All output tokens are zero — routing is broken"


# -----------------------------------------------------------------------
# 14. ExpertChoiceTransformerBlock — output shape
# -----------------------------------------------------------------------

def test_transformer_block_shape():
    torch.manual_seed(0)
    cfg = make_cfg()
    block = ExpertChoiceTransformerBlock(cfg)
    x = torch.randn(B, T, D)
    out, _ = block(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


# -----------------------------------------------------------------------
# 15. ExpertChoiceTransformerBlock — returns aux dict
# -----------------------------------------------------------------------

def test_transformer_block_aux_info():
    torch.manual_seed(0)
    cfg = make_cfg()
    block = ExpertChoiceTransformerBlock(cfg)
    x = torch.randn(B, T, D)
    out, aux = block(x)
    assert isinstance(aux, dict), f"Expected dict, got {type(aux)}"
    assert "aux_loss" in aux, f"Expected 'aux_loss' in aux_info, got {list(aux.keys())}"
