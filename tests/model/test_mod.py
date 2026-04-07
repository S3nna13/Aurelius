"""Tests for Mixture of Depths (MoD) implementation."""

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer, MoDBlock, TransformerBlock


# ---------------------------------------------------------------------------
# Shared small config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=4,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=128,
    )


@pytest.fixture
def small_cfg_mod(small_cfg):
    small_cfg.use_mod = True
    return small_cfg


# ---------------------------------------------------------------------------
# Test 1: MoDBlock output shape matches input shape
# ---------------------------------------------------------------------------

def test_mod_block_output_shape(small_cfg):
    block = TransformerBlock(small_cfg, layer_idx=0)
    mod = MoDBlock(block, capacity_factor=0.5)

    B, S, D = 2, 16, small_cfg.d_model
    x = torch.randn(B, S, D)

    from src.model.attention import precompute_rope_frequencies
    freqs_cis = precompute_rope_frequencies(small_cfg.head_dim, small_cfg.max_seq_len, small_cfg.rope_theta)
    freqs_cis = freqs_cis[:S]

    with torch.no_grad():
        out = mod(x, freqs_cis, mask=None)

    assert out.shape == (B, S, D), f"Expected shape {(B, S, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: With capacity_factor=0.5, exactly half the tokens are routed
# ---------------------------------------------------------------------------

def test_mod_block_selects_top_k(small_cfg):
    block = TransformerBlock(small_cfg, layer_idx=0)
    capacity_factor = 0.5
    mod = MoDBlock(block, capacity_factor=capacity_factor)

    B, S, D = 1, 20, small_cfg.d_model
    x = torch.randn(B, S, D)

    from src.model.attention import precompute_rope_frequencies
    freqs_cis = precompute_rope_frequencies(small_cfg.head_dim, small_cfg.max_seq_len, small_cfg.rope_theta)
    freqs_cis = freqs_cis[:S]

    # Verify that k == int(capacity_factor * S) tokens are selected
    expected_k = max(1, int(capacity_factor * S))

    # Inspect the router's top-k selection directly
    with torch.no_grad():
        routing_weights = mod.router(x).squeeze(-1)  # (B, S)
        k = max(1, int(capacity_factor * S))
        _, top_indices = torch.topk(routing_weights, k, dim=-1)

    assert top_indices.shape == (B, expected_k), (
        f"Expected top_indices shape {(B, expected_k)}, got {top_indices.shape}"
    )
    assert k == expected_k, f"Expected k={expected_k}, got k={k}"


# ---------------------------------------------------------------------------
# Test 3: AureliusConfig has correct default fields for MoD
# ---------------------------------------------------------------------------

def test_mod_config_fields():
    cfg = AureliusConfig()
    assert hasattr(cfg, "use_mod"), "AureliusConfig missing 'use_mod' field"
    assert hasattr(cfg, "mod_capacity_factor"), "AureliusConfig missing 'mod_capacity_factor' field"
    assert cfg.use_mod is False, f"Expected use_mod=False, got {cfg.use_mod}"
    assert cfg.mod_capacity_factor == 0.5, (
        f"Expected mod_capacity_factor=0.5, got {cfg.mod_capacity_factor}"
    )


# ---------------------------------------------------------------------------
# Test 4: use_mod=False (default) — transformer behaves normally, no MoDBlock
# ---------------------------------------------------------------------------

def test_mod_disabled_passes_through(small_cfg):
    # Default config has use_mod=False
    assert small_cfg.use_mod is False

    model = AureliusTransformer(small_cfg)

    # No layer should be wrapped in MoDBlock
    for layer in model.layers:
        assert not isinstance(layer, MoDBlock), (
            "With use_mod=False, no layer should be wrapped in MoDBlock"
        )

    # Forward pass should work normally
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 32, small_cfg.vocab_size), (
        f"Expected logits shape (2, 32, {small_cfg.vocab_size}), got {logits.shape}"
    )
