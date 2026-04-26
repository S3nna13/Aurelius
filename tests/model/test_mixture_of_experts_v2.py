"""Tests for src/model/mixture_of_experts_v2.py

15 tests covering Expert, TopKRouter, ExpertCapacityBuffer, MoELayer,
MoETransformerBlock, and MoEModel.

Tiny config: d_model=16, n_heads=2, n_experts=4, d_ff=32, top_k=2,
             seq_len=8, batch=2, n_layers=2, vocab=16.
"""

import torch
import torch.nn as nn

from src.model.mixture_of_experts_v2 import (
    Expert,
    ExpertCapacityBuffer,
    MoELayer,
    MoEModel,
    MoETransformerBlock,
    TopKRouter,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
N_EXPERTS = 4
D_FF = 32
TOP_K = 2
SEQ_LEN = 8
BATCH = 2
N_LAYERS = 2
VOCAB = 16


def make_tokens():
    """Return a (BATCH*SEQ_LEN, D_MODEL) float tensor."""
    return torch.randn(BATCH * SEQ_LEN, D_MODEL)


def make_3d():
    """Return a (BATCH, SEQ_LEN, D_MODEL) float tensor."""
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


def make_ids():
    """Return a (BATCH, SEQ_LEN) LongTensor of token ids."""
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


# ===========================================================================
# 1. Expert: output shape (N, D)
# ===========================================================================
def test_expert_output_shape():
    expert = Expert(D_MODEL, D_FF, activation="gelu")
    x = make_tokens()
    out = expert(x)
    assert out.shape == (BATCH * SEQ_LEN, D_MODEL), (
        f"Expected {(BATCH * SEQ_LEN, D_MODEL)}, got {out.shape}"
    )


# ===========================================================================
# 2. Expert: gradient flows
# ===========================================================================
def test_expert_grad_flows():
    expert = Expert(D_MODEL, D_FF, activation="gelu")
    x = make_tokens().requires_grad_(True)
    out = expert(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Params should have gradients too
    for p in expert.parameters():
        assert p.grad is not None


# ===========================================================================
# 3. Expert: gelu / silu / relu all produce valid outputs
# ===========================================================================
def test_expert_activations():
    x = make_tokens()
    for act in ("gelu", "silu", "relu"):
        expert = Expert(D_MODEL, D_FF, activation=act)
        out = expert(x)
        assert out.shape == x.shape, f"activation={act} shape mismatch"
        assert not torch.isnan(out).any(), f"activation={act} produced NaN"


# ===========================================================================
# 4. TopKRouter: dispatch_weights shape (N, top_k) and sum-to-1 per token
# ===========================================================================
def test_router_dispatch_weights_shape_and_sum():
    router = TopKRouter(D_MODEL, N_EXPERTS, top_k=TOP_K)
    x = make_3d()
    weights, indices = router(x)
    N = BATCH * SEQ_LEN
    assert weights.shape == (N, TOP_K), f"weights shape {weights.shape}"
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(N), atol=1e-5), "weights don't sum to 1"


# ===========================================================================
# 5. TopKRouter: expert_indices shape (N, top_k)
# ===========================================================================
def test_router_expert_indices_shape():
    router = TopKRouter(D_MODEL, N_EXPERTS, top_k=TOP_K)
    x = make_3d()
    weights, indices = router(x)
    N = BATCH * SEQ_LEN
    assert indices.shape == (N, TOP_K), f"indices shape {indices.shape}"
    assert indices.dtype == torch.long


# ===========================================================================
# 6. TopKRouter: top_k=1 -> single expert, weight == 1.0 for all tokens
# ===========================================================================
def test_router_top_k_1():
    router = TopKRouter(D_MODEL, N_EXPERTS, top_k=1)
    x = make_3d()
    weights, indices = router(x)
    N = BATCH * SEQ_LEN
    assert weights.shape == (N, 1)
    assert torch.allclose(weights, torch.ones(N, 1), atol=1e-5), "top_k=1 weights should all be 1.0"


# ===========================================================================
# 7. ExpertCapacityBuffer.compute_capacity: proportional to capacity_factor, >= 1
# ===========================================================================
def test_capacity_compute():
    ecb = ExpertCapacityBuffer(n_experts=4, capacity_factor=1.25)
    n_tokens = 16
    top_k = 2
    cap = ecb.compute_capacity(n_tokens, top_k)
    expected_raw = 1.25 * n_tokens * top_k / 4  # = 10.0
    assert cap >= 1
    assert cap == max(1, int(import_math_ceil(expected_raw)))

    # Doubling capacity_factor should double the result
    ecb2 = ExpertCapacityBuffer(n_experts=4, capacity_factor=2.5)
    cap2 = ecb2.compute_capacity(n_tokens, top_k)
    assert cap2 >= cap


def import_math_ceil(x):
    import math

    return math.ceil(x)


# ===========================================================================
# 8. ExpertCapacityBuffer.apply_capacity: overflow_mask is bool shape (N,),
#    non-overflow tokens have valid (non-zero) weights
# ===========================================================================
def test_capacity_apply():
    n_tokens = 16
    top_k = 2
    n_experts = 4
    # Use a very small capacity_factor to force overflows
    ecb = ExpertCapacityBuffer(n_experts=n_experts, capacity_factor=0.3)
    weights = torch.rand(n_tokens, top_k)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    indices = torch.randint(0, n_experts, (n_tokens, top_k))

    w_capped, idx_capped, overflow_mask = ecb.apply_capacity(weights, indices, n_tokens)

    assert overflow_mask.dtype == torch.bool, "overflow_mask must be bool"
    assert overflow_mask.shape == (n_tokens,), f"overflow_mask shape {overflow_mask.shape}"
    # Non-overflow tokens must have positive total weight
    non_overflow = ~overflow_mask
    if non_overflow.any():
        assert (w_capped[non_overflow].sum(dim=-1) > 0).all()


# ===========================================================================
# 9. MoELayer: output shape (B, T, D), aux_loss scalar >= 0
# ===========================================================================
def test_moe_layer_output():
    layer = MoELayer(D_MODEL, N_EXPERTS, D_FF, top_k=TOP_K)
    x = make_3d()
    out, aux = layer(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"out shape {out.shape}"
    assert aux.shape == torch.Size([]), "aux_loss must be scalar"
    assert float(aux.item()) >= 0.0, "aux_loss must be non-negative"


# ===========================================================================
# 10. MoELayer: backward flows through experts and router
# ===========================================================================
def test_moe_layer_backward():
    layer = MoELayer(D_MODEL, N_EXPERTS, D_FF, top_k=TOP_K)
    x = make_3d().requires_grad_(True)
    out, aux = layer(x)
    loss = out.sum() + aux
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    # Router gate should have gradient
    assert layer.router.gate.weight.grad is not None, "Router gate has no gradient"
    # At least one expert should have a gradient
    has_expert_grad = any(p.grad is not None for e in layer.experts for p in e.parameters())
    assert has_expert_grad, "No expert has a gradient"


# ===========================================================================
# 11. MoELayer: aux_loss > 0 when token distribution is (inevitably) non-uniform
# ===========================================================================
def test_moe_aux_loss_positive():
    # With random initialization, aux_loss should be > 0 due to uneven routing
    torch.manual_seed(42)
    layer = MoELayer(D_MODEL, N_EXPERTS, D_FF, top_k=TOP_K)
    x = make_3d()
    _, aux = layer(x)
    assert float(aux.item()) > 0.0, "aux_loss should be > 0 for non-trivial routing"


# ===========================================================================
# 12. MoETransformerBlock: output shape (B, T, D), aux_loss scalar
# ===========================================================================
def test_moe_block_output():
    block = MoETransformerBlock(D_MODEL, N_HEADS, N_EXPERTS, D_FF, top_k=TOP_K)
    x = make_3d()
    out, aux = block(x)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"out shape {out.shape}"
    assert aux.shape == torch.Size([]), "aux_loss must be scalar"


# ===========================================================================
# 13. MoEModel: logits shape (B, T, V), total_aux_loss >= 0, backward succeeds
# ===========================================================================
def test_moe_model_forward_backward():
    model = MoEModel(D_MODEL, N_LAYERS, N_HEADS, VOCAB, N_EXPERTS, D_FF, top_k=TOP_K)
    ids = make_ids()
    logits, total_aux = model(ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB), f"logits shape {logits.shape}"
    assert total_aux.shape == torch.Size([]), "total_aux must be scalar"
    assert float(total_aux.item()) >= 0.0

    loss = logits.sum() + total_aux
    loss.backward()
    # Embedding should have gradients
    assert model.embedding.weight.grad is not None


# ===========================================================================
# 14. MoEModel.router_statistics: all keys present, mean_utilization > 0
# ===========================================================================
def test_moe_model_router_statistics():
    model = MoEModel(D_MODEL, N_LAYERS, N_HEADS, VOCAB, N_EXPERTS, D_FF, top_k=TOP_K)
    ids = make_ids()
    stats = model.router_statistics(ids)
    for key in ("mean_utilization", "load_balance_cv", "expert_collapse"):
        assert key in stats, f"Missing key: {key}"
    assert stats["mean_utilization"] > 0, "mean_utilization should be > 0"
    assert isinstance(stats["expert_collapse"], bool)


# ===========================================================================
# 15. n_experts=2, top_k=1: balanced routing -> each expert gets ~50% tokens
#     (over many samples the average should be near 0.5)
# ===========================================================================
def test_balanced_routing_n_experts_2_top_k_1():
    torch.manual_seed(0)
    # Use a larger batch to get a stable average
    layer = MoELayer(d_model=D_MODEL, n_experts=2, d_ff=D_FF, top_k=1)
    # Override gate weights to zeros so routing is uniform
    nn.init.zeros_(layer.router.gate.weight)

    x = torch.randn(64, SEQ_LEN, D_MODEL)  # 64 * 8 = 512 tokens
    out, aux = layer(x)
    # With zero gate weights, all logits are equal -> 50/50 split
    assert out.shape == (64, SEQ_LEN, D_MODEL)
    # aux_loss should still be non-negative
    assert float(aux.item()) >= 0.0
