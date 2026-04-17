"""Tests for src/model/jamba.py — Jamba hybrid Transformer-Mamba model.

Covers all 15 required test cases from the spec.
Pure PyTorch only — no HuggingFace, einops, mamba_ssm, etc.
"""

from __future__ import annotations

import torch
import pytest

from src.model.jamba import (
    JambaConfig,
    JambaModel,
    JambaAttentionBlock,
    JambaMambaBlock,
    jamba_tiny_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(cfg=None):
    """Create a tiny JambaModel; use default tiny config if none provided."""
    if cfg is None:
        cfg = jamba_tiny_config()
    return JambaModel(cfg).eval()


def rand_input(B=2, T=8, d_model=64):
    return torch.randn(B, T, d_model)


# ---------------------------------------------------------------------------
# Test 1: Output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_output_shape():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    x = rand_input(B=2, T=8, d_model=cfg.d_model)
    out, _ = model(x)
    assert out.shape == (2, 8, cfg.d_model), f"Expected (2, 8, {cfg.d_model}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: hidden_states shape: list of Mamba-layer states, each (B, d_model, d_state)
# ---------------------------------------------------------------------------

def test_hidden_states_shape():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    B, T = 2, 8
    x = rand_input(B=B, T=T, d_model=cfg.d_model)
    _, hs = model(x)
    n_mamba = sum(1 for l in model.layers if isinstance(l, JambaMambaBlock))
    assert len(hs) == n_mamba, f"Expected {n_mamba} hidden states, got {len(hs)}"
    for i, h in enumerate(hs):
        assert h.shape == (B, cfg.d_model, cfg.d_state), (
            f"hidden_states[{i}] shape {h.shape} != ({B}, {cfg.d_model}, {cfg.d_state})"
        )


# ---------------------------------------------------------------------------
# Test 3: Gradient flow — backward gives finite grads on all params
# ---------------------------------------------------------------------------

def test_gradient_flow():
    cfg = jamba_tiny_config()
    model = JambaModel(cfg).train()
    x = rand_input(B=2, T=8, d_model=cfg.d_model)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Param {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Param {name} has non-finite gradient"


# ---------------------------------------------------------------------------
# Test 4: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism():
    cfg = jamba_tiny_config()

    torch.manual_seed(42)
    model1 = make_model(cfg)
    x = rand_input(B=2, T=8, d_model=cfg.d_model)
    out1, _ = model1(x)

    torch.manual_seed(42)
    model2 = make_model(cfg)
    out2, _ = model2(x)

    assert torch.allclose(out1, out2), "Outputs differ despite same seed"


# ---------------------------------------------------------------------------
# Test 5: batch=1, seq_len=1
# ---------------------------------------------------------------------------

def test_single_token_single_batch():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    x = rand_input(B=1, T=1, d_model=cfg.d_model)
    out, hs = model(x)
    assert out.shape == (1, 1, cfg.d_model)
    assert all(h.shape == (1, cfg.d_model, cfg.d_state) for h in hs)


# ---------------------------------------------------------------------------
# Test 6: hidden_states=None -> zero init, no crash
# ---------------------------------------------------------------------------

def test_hidden_states_none_no_crash():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    x = rand_input(B=2, T=4, d_model=cfg.d_model)
    out, hs = model(x, hidden_states=None)
    assert out.shape == (2, 4, cfg.d_model)
    assert len(hs) > 0


# ---------------------------------------------------------------------------
# Test 7: hidden_states passed -> used and updated
# ---------------------------------------------------------------------------

def test_hidden_states_passed_and_updated():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    B, T = 2, 4
    x = rand_input(B=B, T=T, d_model=cfg.d_model)

    # First forward: get hidden states
    _, hs1 = model(x)

    # Second forward: pass the hidden states back in
    out2, hs2 = model(x, hidden_states=hs1)

    assert out2.shape == (B, T, cfg.d_model)
    # Hidden states should be updated (different from hs1 in general)
    any_changed = any(not torch.allclose(h1, h2) for h1, h2 in zip(hs1, hs2))
    assert any_changed, "hidden_states were not updated after second forward pass"


# ---------------------------------------------------------------------------
# Test 8: No NaN/Inf on zeros input
# ---------------------------------------------------------------------------

def test_no_nan_inf_zeros():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    x = torch.zeros(2, 8, cfg.d_model)
    out, hs = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf in output for zeros input"
    for h in hs:
        assert torch.isfinite(h).all(), "NaN/Inf in hidden state for zeros input"


# ---------------------------------------------------------------------------
# Test 9: No NaN/Inf on large inputs
# ---------------------------------------------------------------------------

def test_no_nan_inf_large_input():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    x = torch.randn(2, 8, cfg.d_model) * 100.0
    out, hs = model(x)
    assert torch.isfinite(out).all(), "NaN/Inf in output for large input"
    for h in hs:
        assert torch.isfinite(h).all(), "NaN/Inf in hidden state for large input"


# ---------------------------------------------------------------------------
# Test 10: Attention layers at correct positions (layer_type attribute)
# ---------------------------------------------------------------------------

def test_attention_layers_correct_positions():
    cfg = jamba_tiny_config()  # attn_layer_offset=0, attn_every_k=4
    model = make_model(cfg)
    for i, layer in enumerate(model.layers):
        expected_attn = (i - cfg.attn_layer_offset) % cfg.attn_every_k == 0
        is_attn = isinstance(layer, JambaAttentionBlock)
        assert is_attn == expected_attn, (
            f"Layer {i}: expected {'attention' if expected_attn else 'mamba'}, "
            f"got {'attention' if is_attn else 'mamba'}"
        )
        # layer_type attribute
        if is_attn:
            assert layer.layer_type == "attention", f"Layer {i} layer_type wrong"
        else:
            assert layer.layer_type == "mamba", f"Layer {i} layer_type wrong"


# ---------------------------------------------------------------------------
# Test 11: Number of attention blocks == expected count
# ---------------------------------------------------------------------------

def test_number_of_attention_blocks():
    cfg = jamba_tiny_config()  # n_layers=8, attn_every_k=4, offset=0
    model = make_model(cfg)
    n_attn = sum(1 for l in model.layers if isinstance(l, JambaAttentionBlock))
    expected = sum(
        1 for i in range(cfg.n_layers)
        if (i - cfg.attn_layer_offset) % cfg.attn_every_k == 0
    )
    assert n_attn == expected, f"Expected {expected} attention blocks, got {n_attn}"


# ---------------------------------------------------------------------------
# Test 12: State carries context
# ---------------------------------------------------------------------------

def test_state_carries_context():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    torch.manual_seed(0)
    x_part1 = rand_input(B=1, T=4, d_model=cfg.d_model)
    x_part2 = rand_input(B=1, T=8, d_model=cfg.d_model)

    # Run first chunk to get non-trivial hidden states
    _, hs_after_part1 = model(x_part1)

    # Forward with history
    out_with_history, _ = model(x_part2, hidden_states=hs_after_part1)

    # Forward without history (zeros)
    out_without_history, _ = model(x_part2, hidden_states=None)

    assert not torch.allclose(out_with_history, out_without_history), (
        "Outputs are identical with and without history — state is not being used"
    )


# ---------------------------------------------------------------------------
# Test 13: Different hidden_states -> different outputs
# ---------------------------------------------------------------------------

def test_different_hidden_states_give_different_outputs():
    cfg = jamba_tiny_config()
    model = make_model(cfg)
    x = rand_input(B=2, T=4, d_model=cfg.d_model)

    # Two different sets of hidden states
    n_mamba = sum(1 for l in model.layers if isinstance(l, JambaMambaBlock))
    hs_a = [torch.zeros(2, cfg.d_model, cfg.d_state) for _ in range(n_mamba)]
    hs_b = [torch.randn(2, cfg.d_model, cfg.d_state) for _ in range(n_mamba)]

    out_a, _ = model(x, hidden_states=hs_a)
    out_b, _ = model(x, hidden_states=hs_b)

    assert not torch.allclose(out_a, out_b), (
        "Different hidden_states produced identical outputs"
    )


# ---------------------------------------------------------------------------
# Test 14: attn_every_k=1 -> all attention layers (pure transformer mode)
# ---------------------------------------------------------------------------

def test_all_attention_when_attn_every_k_1():
    cfg = JambaConfig(
        d_model=64,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        d_state=16,
        d_conv=4,
        attn_layer_offset=0,
        attn_every_k=1,
    )
    model = make_model(cfg)
    n_attn = sum(1 for l in model.layers if isinstance(l, JambaAttentionBlock))
    n_mamba = sum(1 for l in model.layers if isinstance(l, JambaMambaBlock))
    assert n_attn == cfg.n_layers, f"Expected all {cfg.n_layers} attn, got {n_attn}"
    assert n_mamba == 0, f"Expected 0 Mamba layers, got {n_mamba}"

    # Forward should work with no Mamba layers -> empty hidden_states
    x = rand_input(B=2, T=4, d_model=cfg.d_model)
    out, hs = model(x)
    assert out.shape == (2, 4, cfg.d_model)
    assert hs == [], f"Expected empty hidden_states list, got {hs}"


# ---------------------------------------------------------------------------
# Test 15: attn_every_k=n_layers -> single attention at offset, rest Mamba
# ---------------------------------------------------------------------------

def test_single_attention_rest_mamba():
    n_layers = 8
    cfg = JambaConfig(
        d_model=64,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=2,
        d_state=16,
        d_conv=4,
        attn_layer_offset=0,
        attn_every_k=n_layers,
    )
    model = make_model(cfg)
    n_attn = sum(1 for l in model.layers if isinstance(l, JambaAttentionBlock))
    n_mamba = sum(1 for l in model.layers if isinstance(l, JambaMambaBlock))

    # With attn_every_k=n_layers=8 and offset=0:
    # only layer 0 (8 is out of range) -> 1 attention block
    assert n_attn == 1, f"Expected 1 attention block, got {n_attn}"
    assert n_mamba == n_layers - 1, f"Expected {n_layers - 1} Mamba blocks, got {n_mamba}"

    # Forward should work
    x = rand_input(B=2, T=4, d_model=cfg.d_model)
    out, hs = model(x)
    assert out.shape == (2, 4, cfg.d_model)
    assert len(hs) == n_mamba
