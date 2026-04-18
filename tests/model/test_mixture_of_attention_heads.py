"""
Tests for Mixture of Attention Heads (MoAH).

Fixture dimensions:  d_model=16, vocab_size=16, n_layers=2,
                     d_head=8, B=2, T=8, window_size=3.
"""

import math
import pytest
import torch

from src.model.mixture_of_attention_heads import (
    LocalAttentionHead,
    GlobalAttentionHead,
    RelativePositionHead,
    HeadRouter,
    MixtureOfAttentionHeads,
    MoAHTransformerBlock,
    MoAHLanguageModel,
    MoAHConfig,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
D_MODEL = 16
D_HEAD = 8
VOCAB_SIZE = 16
N_LAYERS = 2
B = 2
T = 8
WINDOW_SIZE = 3
TOP_K = 2
N_HEADS_PER_TYPE = 2

torch.manual_seed(42)


def _rand_input() -> torch.Tensor:
    return torch.randn(B, T, D_MODEL)


def _rand_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ===========================================================================
# LocalAttentionHead
# ===========================================================================

def test_local_attention_head_output_shape():
    head = LocalAttentionHead(D_MODEL, D_HEAD, window_size=WINDOW_SIZE)
    x = _rand_input()
    out = head(x)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B},{T},{D_HEAD}), got {out.shape}"


def test_local_attention_head_no_nan():
    head = LocalAttentionHead(D_MODEL, D_HEAD, window_size=WINDOW_SIZE)
    x = _rand_input()
    out = head(x)
    assert not torch.isnan(out).any(), "LocalAttentionHead output contains NaN"


def test_local_attention_head_window_mask_applied():
    """
    Verify that the sliding-window mask is actually applied.

    Compare a head with window_size=0 against one with window_size=T.
    With window_size=0 a token can only attend to itself; with window_size=T
    it attends everywhere, so outputs must differ on a random input.
    """
    torch.manual_seed(0)
    head_narrow = LocalAttentionHead(D_MODEL, D_HEAD, window_size=0)
    head_wide   = LocalAttentionHead(D_MODEL, D_HEAD, window_size=T)

    # Share weights so the only difference is the mask
    with torch.no_grad():
        head_wide.W_q.weight.copy_(head_narrow.W_q.weight)
        head_wide.W_k.weight.copy_(head_narrow.W_k.weight)
        head_wide.W_v.weight.copy_(head_narrow.W_v.weight)

    x = torch.randn(1, 6, D_MODEL)
    out_narrow = head_narrow(x)
    out_wide   = head_wide(x)

    # For T>1 the outputs should differ somewhere (window genuinely restricts)
    assert not torch.allclose(out_narrow, out_wide, atol=1e-5), (
        "Window-0 and window-T heads produced identical outputs — "
        "mask is not being applied."
    )


# ===========================================================================
# GlobalAttentionHead
# ===========================================================================

def test_global_attention_head_output_shape():
    head = GlobalAttentionHead(D_MODEL, D_HEAD)
    x = _rand_input()
    out = head(x)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B},{T},{D_HEAD}), got {out.shape}"


def test_global_attention_head_no_nan():
    head = GlobalAttentionHead(D_MODEL, D_HEAD)
    x = _rand_input()
    out = head(x)
    assert not torch.isnan(out).any(), "GlobalAttentionHead output contains NaN"


def test_global_attention_head_causal():
    """
    Changing a future token must not change past token outputs (causal mask).
    """
    torch.manual_seed(1)
    head = GlobalAttentionHead(D_MODEL, D_HEAD)
    head.train(False)  # inference mode

    x = _rand_input()
    x2 = x.clone()
    x2[:, -1, :] = x2[:, -1, :] + 10.0  # perturb last position

    with torch.no_grad():
        out1 = head(x)
        out2 = head(x2)

    # Position 0 output must be identical (it only attends to itself)
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), (
        "Causal mask violated: position 0 changed after perturbing the last position."
    )


# ===========================================================================
# RelativePositionHead
# ===========================================================================

def test_relative_position_head_output_shape():
    head = RelativePositionHead(D_MODEL, D_HEAD, max_relative_pos=4)
    x = _rand_input()
    out = head(x)
    assert out.shape == (B, T, D_HEAD), f"Expected ({B},{T},{D_HEAD}), got {out.shape}"


def test_relative_position_head_no_nan():
    head = RelativePositionHead(D_MODEL, D_HEAD, max_relative_pos=4)
    x = _rand_input()
    out = head(x)
    assert not torch.isnan(out).any(), "RelativePositionHead output contains NaN"


# ===========================================================================
# HeadRouter
# ===========================================================================

def test_head_router_gates_shape():
    n_head_types = 3
    router = HeadRouter(D_MODEL, n_head_types, top_k=TOP_K)
    x = _rand_input()
    gates, indices = router(x)
    assert gates.shape == (B, T, TOP_K), f"Expected ({B},{T},{TOP_K}), got {gates.shape}"


def test_head_router_indices_shape():
    n_head_types = 3
    router = HeadRouter(D_MODEL, n_head_types, top_k=TOP_K)
    x = _rand_input()
    _, indices = router(x)
    assert indices.shape == (B, T, TOP_K)


def test_head_router_gates_sum_to_one():
    n_head_types = 3
    router = HeadRouter(D_MODEL, n_head_types, top_k=TOP_K)
    x = _rand_input()
    gates, _ = router(x)
    sums = gates.sum(dim=-1)  # [B, T]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Gates do not sum to 1.0 per token. Max deviation: {(sums - 1).abs().max().item()}"
    )


def test_head_router_indices_in_range():
    n_head_types = 3
    router = HeadRouter(D_MODEL, n_head_types, top_k=TOP_K)
    x = _rand_input()
    _, indices = router(x)
    assert indices.min().item() >= 0
    assert indices.max().item() < n_head_types


# ===========================================================================
# MixtureOfAttentionHeads
# ===========================================================================

def test_moah_output_shape():
    moah = MixtureOfAttentionHeads(
        d_model=D_MODEL, d_head=D_HEAD,
        n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    x = _rand_input()
    out = moah(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out.shape}"


def test_moah_routing_stats_keys():
    moah = MixtureOfAttentionHeads(
        d_model=D_MODEL, d_head=D_HEAD,
        n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    moah(_rand_input())
    stats = moah.routing_stats()
    expected_keys = {"local", "global", "relative_position"}
    assert set(stats.keys()) == expected_keys, (
        f"routing_stats keys mismatch: got {set(stats.keys())}"
    )


def test_moah_routing_stats_fractions_sum_to_one():
    moah = MixtureOfAttentionHeads(
        d_model=D_MODEL, d_head=D_HEAD,
        n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    moah(_rand_input())
    stats = moah.routing_stats()
    total = sum(stats.values())
    assert abs(total - 1.0) < 1e-5, f"Routing fractions sum to {total}, expected ~1.0"


def test_moah_gradient_flows():
    moah = MixtureOfAttentionHeads(
        d_model=D_MODEL, d_head=D_HEAD,
        n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    x = _rand_input().requires_grad_(True)
    out = moah(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed back to input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"


# ===========================================================================
# MoAHTransformerBlock
# ===========================================================================

def test_moah_transformer_block_output_shape():
    block = MoAHTransformerBlock(
        d_model=D_MODEL, d_head=D_HEAD,
        n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    x = _rand_input()
    out = block(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B},{T},{D_MODEL}), got {out.shape}"


def test_moah_transformer_block_residual_not_zero():
    """Block output must differ from input (residual + attention are non-trivial)."""
    block = MoAHTransformerBlock(
        d_model=D_MODEL, d_head=D_HEAD,
        n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    x = _rand_input()
    out = block(x)
    assert not torch.allclose(out, x, atol=1e-5), "Block output identical to input"


# ===========================================================================
# MoAHLanguageModel
# ===========================================================================

def test_language_model_output_shape():
    model = MoAHLanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        d_head=D_HEAD, n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    ids = _rand_ids()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE), (
        f"Expected ({B},{T},{VOCAB_SIZE}), got {logits.shape}"
    )


def test_language_model_compute_loss_finite_positive():
    model = MoAHLanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        d_head=D_HEAD, n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    ids = _rand_ids()
    loss = model.compute_loss(ids)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"


def test_language_model_compute_loss_backward():
    model = MoAHLanguageModel(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS,
        d_head=D_HEAD, n_heads_per_type=N_HEADS_PER_TYPE, top_k=TOP_K,
        window_size=WINDOW_SIZE,
    )
    ids = _rand_ids()
    loss = model.compute_loss(ids)
    loss.backward()

    # At least one parameter must have a gradient
    has_grad = any(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
    )
    assert has_grad, "No valid gradients found after backward pass"


# ===========================================================================
# MoAHConfig
# ===========================================================================

def test_moah_config_defaults():
    cfg = MoAHConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.d_head == 8
    assert cfg.n_heads_per_type == 2
    assert cfg.top_k == 2
    assert cfg.window_size == 4


def test_moah_config_custom():
    cfg = MoAHConfig(d_model=64, vocab_size=128, n_layers=4)
    assert cfg.d_model == 64
    assert cfg.vocab_size == 128
    assert cfg.n_layers == 4
    # Remaining defaults unchanged
    assert cfg.d_head == 8
