"""Tests for src/model/local_global_attention.py.

Covers:
- is_global_layer helper
- LocalAttentionLayer and GlobalAttentionLayer forward shapes
- Sliding-window mask correctness
- InterleavedAttentionLayer layer selection
- RoPE theta differentiation
"""

import math
import pytest
import torch

from src.model.config import AureliusConfig
from src.model.local_global_attention import (
    LOCAL_ROPE_THETA,
    GLOBAL_ROPE_THETA,
    LOCAL_WINDOW_SIZE,
    is_global_layer,
    LocalAttentionLayer,
    GlobalAttentionLayer,
    InterleavedAttentionLayer,
    _build_local_mask,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> AureliusConfig:
    return AureliusConfig()


def inference_mode(layer):
    """Put a module in inference mode (train=False). Avoids dropout."""
    return layer.train(False)


# ---------------------------------------------------------------------------
# 1. is_global_layer
# ---------------------------------------------------------------------------

def test_is_global_layer():
    """Layers 0-4 are local; layer 5 is global; pattern repeats."""
    # Local layers in the first group
    for idx in (0, 1, 2, 3, 4):
        assert not is_global_layer(idx), f"Layer {idx} should be local"

    # Global layers
    assert is_global_layer(5), "Layer 5 should be global"
    assert is_global_layer(11), "Layer 11 should be global"

    # Second group of local layers
    for idx in (6, 7, 8, 9, 10):
        assert not is_global_layer(idx), f"Layer {idx} should be local"


# ---------------------------------------------------------------------------
# 2. LocalAttentionLayer — forward shape
# ---------------------------------------------------------------------------

def test_local_attention_forward_shape(config):
    """(2, 64, 2048) input produces (2, 64, 2048) output."""
    layer = inference_mode(LocalAttentionLayer(config))

    x = torch.randn(2, 64, config.d_model)
    with torch.no_grad():
        out = layer(x)

    assert out.shape == (2, 64, config.d_model), (
        f"Expected (2, 64, {config.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 3. GlobalAttentionLayer — forward shape
# ---------------------------------------------------------------------------

def test_global_attention_forward_shape(config):
    """(2, 64, 2048) input produces (2, 64, 2048) output."""
    layer = inference_mode(GlobalAttentionLayer(config))

    x = torch.randn(2, 64, config.d_model)
    with torch.no_grad():
        out = layer(x)

    assert out.shape == (2, 64, config.d_model), (
        f"Expected (2, 64, {config.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 4. Local mask restricts attention range
# ---------------------------------------------------------------------------

def test_local_attention_mask_restricts_range():
    """Token at position 100 should NOT attend to token at position 0
    when window_size=32 and seq_len=128.
    """
    window_size = 32
    seq_len = 128
    mask = _build_local_mask(seq_len, window_size, device=torch.device("cpu"))

    # mask[i, j] == -inf  means token i cannot attend to token j
    assert mask[100, 0] == float("-inf"), (
        "Token 100 must not attend to token 0 with window_size=32"
    )

    # Sanity-check: token 100 CAN attend to token 100 (itself)
    assert mask[100, 100] == 0.0, "Token 100 must be able to attend to itself"

    # Token 100 can attend to token 100 - window_size = 68 (boundary)
    assert mask[100, 68] == 0.0, (
        "Token 100 must be able to attend to token 68 (boundary of window)"
    )

    # Token 100 cannot attend to token 67 (just outside window)
    assert mask[100, 67] == float("-inf"), (
        "Token 100 must not attend to token 67 (outside window)"
    )

    # Causal: token 50 cannot attend to token 51 (future)
    assert mask[50, 51] == float("-inf"), (
        "Token 50 must not attend to future token 51"
    )


# ---------------------------------------------------------------------------
# 5. InterleavedAttentionLayer selects correctly
# ---------------------------------------------------------------------------

def test_interleaved_layer_selects_correctly(config):
    """layer_idx=5 uses global; layer_idx=0 uses local. Both produce correct shapes."""
    x = torch.randn(2, 64, config.d_model)

    global_layer = inference_mode(InterleavedAttentionLayer(config, layer_idx=5))
    local_layer = inference_mode(InterleavedAttentionLayer(config, layer_idx=0))

    # Verify internal type selection
    assert isinstance(global_layer.attn, GlobalAttentionLayer), (
        "layer_idx=5 must use GlobalAttentionLayer"
    )
    assert isinstance(local_layer.attn, LocalAttentionLayer), (
        "layer_idx=0 must use LocalAttentionLayer"
    )

    with torch.no_grad():
        out_global = global_layer(x)
        out_local = local_layer(x)

    assert out_global.shape == (2, 64, config.d_model)
    assert out_local.shape == (2, 64, config.d_model)


# ---------------------------------------------------------------------------
# 6. Different RoPE theta per layer type
# ---------------------------------------------------------------------------

def test_different_rope_theta(config):
    """Local layers use theta=10 000; global layers use theta=1 000 000."""
    local_layer = LocalAttentionLayer(config)
    global_layer = GlobalAttentionLayer(config)

    assert local_layer.rope_theta == LOCAL_ROPE_THETA, (
        f"Local layer theta should be {LOCAL_ROPE_THETA}, got {local_layer.rope_theta}"
    )
    assert global_layer.rope_theta == GLOBAL_ROPE_THETA, (
        f"Global layer theta should be {GLOBAL_ROPE_THETA}, got {global_layer.rope_theta}"
    )

    # Also verify via InterleavedAttentionLayer proxy property
    interleaved_local = InterleavedAttentionLayer(config, layer_idx=0)
    interleaved_global = InterleavedAttentionLayer(config, layer_idx=5)

    assert interleaved_local.rope_theta == LOCAL_ROPE_THETA
    assert interleaved_global.rope_theta == GLOBAL_ROPE_THETA
