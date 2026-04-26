"""Tests for Mamba selective state-space model (SSM) layer.

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
"""

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.mamba import MambaBlock, MambaConfig, MambaLayer, SelectiveSSM

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    """Minimal AureliusConfig matching the project test config."""
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


# ---------------------------------------------------------------------------
# 1. test_mamba_block_output_shape
# ---------------------------------------------------------------------------


def test_mamba_block_output_shape(small_cfg):
    """MambaBlock(input (2,16,64)) must return tensor of shape (2,16,64)."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    out = block(x)
    assert out.shape == (2, 16, small_cfg.d_model), (
        f"Expected (2, 16, {small_cfg.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. test_mamba_layer_output_shape
# ---------------------------------------------------------------------------


def test_mamba_layer_output_shape(small_cfg):
    """MambaLayer(input (2,16,64)) must return tensor of same shape."""
    layer = MambaLayer(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    out = layer(x)
    assert out.shape == (2, 16, small_cfg.d_model), (
        f"Expected (2, 16, {small_cfg.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 3. test_mamba_residual_connection
# ---------------------------------------------------------------------------


def test_mamba_residual_connection(small_cfg):
    """MambaLayer output should not be equal to its input (not identity)."""
    torch.manual_seed(0)
    layer = MambaLayer(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    out = layer(x)
    assert not torch.allclose(out, x, atol=1e-5), (
        "MambaLayer output is identical to input -- residual or SSM is broken."
    )


# ---------------------------------------------------------------------------
# 4. test_selective_ssm_output_shape
# ---------------------------------------------------------------------------


def test_selective_ssm_output_shape():
    """SelectiveSSM(B, L, d_inner) must return tensor of same shape (B, L, d_inner)."""
    d_model = 64
    d_inner = 128  # expand=2
    mamba_cfg = MambaConfig(d_state=16, d_conv=4, expand=2)
    ssm = SelectiveSSM(d_model=d_model, d_inner=d_inner, mamba_cfg=mamba_cfg)
    x = torch.randn(2, 16, d_inner)
    out = ssm(x)
    assert out.shape == (2, 16, d_inner), f"Expected (2, 16, {d_inner}), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. test_mamba_block_causal_depthwise_conv
# ---------------------------------------------------------------------------


def test_mamba_block_causal_depthwise_conv(small_cfg):
    """Causal conv: output at t=0 must not change when inputs at t>0 change."""
    torch.manual_seed(42)
    block = MambaBlock(small_cfg)
    block.eval()

    B, L, D = 1, 8, small_cfg.d_model
    x1 = torch.randn(B, L, D)

    # Modify only positions t >= 1 (future from perspective of t=0)
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(B, L - 1, D)

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), (
        "Output at t=0 changed when only t>0 inputs changed -- conv is not causal!"
    )


# ---------------------------------------------------------------------------
# 6. test_mamba_config_dt_rank_auto
# ---------------------------------------------------------------------------


def test_mamba_config_dt_rank_auto():
    """MambaConfig with d_model=64 should resolve dt_rank to ceil(64/16)=4."""
    mamba_cfg = MambaConfig(dt_rank="auto")
    # Resolution happens inside SelectiveSSM where d_model is known
    d_model = 64
    d_inner = 128  # expand=2
    ssm = SelectiveSSM(d_model=d_model, d_inner=d_inner, mamba_cfg=mamba_cfg)
    expected = math.ceil(64 / 16)  # = 4
    assert ssm.dt_rank == expected, f"Expected dt_rank={expected} for d_model=64, got {ssm.dt_rank}"


# ---------------------------------------------------------------------------
# 7. test_mamba_block_different_configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expand", [2, 4])
def test_mamba_block_different_configs(small_cfg, expand):
    """MambaBlock should work correctly with expand=2 and expand=4."""
    mamba_cfg = MambaConfig(expand=expand)
    block = MambaBlock(small_cfg, mamba_cfg=mamba_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    out = block(x)
    assert out.shape == (2, 16, small_cfg.d_model), (
        f"expand={expand}: Expected (2, 16, {small_cfg.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 8. test_mamba_kwargs_passthrough
# ---------------------------------------------------------------------------


def test_mamba_kwargs_passthrough(small_cfg):
    """MambaLayer.forward(x, past_key_values=None) must not raise any error."""
    layer = MambaLayer(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    # Should not raise -- **kwargs must be accepted and silently ignored
    out = layer(x, past_key_values=None)
    assert out.shape == (2, 16, small_cfg.d_model)
