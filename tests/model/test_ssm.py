"""Tests for Mamba-style Selective State Space Model (S6).

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
"""

import math
import pytest
import torch
from src.model.config import AureliusConfig
from src.model.ssm import SSMConfig, SelectiveSSM, MambaBlock, selective_scan_naive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    """Small AureliusConfig for fast tests."""
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=1000,
        max_seq_len=128,
    )


@pytest.fixture
def ssm_cfg():
    """Small SSMConfig for unit tests."""
    return SSMConfig(d_model=64, d_state=8, d_conv=4, expand=2)


@pytest.fixture
def ssm(ssm_cfg):
    return SelectiveSSM(ssm_cfg)


# ---------------------------------------------------------------------------
# 1. SSMConfig dt_rank auto
# ---------------------------------------------------------------------------

def test_ssm_config_dt_rank_auto():
    """'auto' dt_rank should equal ceil(d_model / 16)."""
    cfg = SSMConfig(d_model=64, dt_rank="auto")
    expected = math.ceil(64 / 16)  # = 4
    assert cfg.dt_rank == expected or cfg.dt_rank == "auto", (
        "SSMConfig with dt_rank='auto' should store 'auto' or the resolved int"
    )
    # Resolve via SelectiveSSM constructor
    ssm = SelectiveSSM(cfg)
    assert ssm.dt_rank == expected, f"Expected dt_rank={expected}, got {ssm.dt_rank}"


# ---------------------------------------------------------------------------
# 2. selective_scan_naive output shape
# ---------------------------------------------------------------------------

def test_selective_scan_output_shape():
    """selective_scan_naive must return (B, L, d_inner)."""
    B, L, d_inner, d_state, dt_rank = 2, 16, 32, 8, 4
    u = torch.randn(B, L, d_inner)
    delta = torch.randn(B, L, dt_rank)
    A = torch.randn(d_inner, d_state)
    B_mat = torch.randn(B, L, d_state)
    C = torch.randn(B, L, d_state)
    D = torch.randn(d_inner)
    out = selective_scan_naive(u, delta, A, B_mat, C, D)
    assert out.shape == (B, L, d_inner), f"Expected ({B}, {L}, {d_inner}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. selective_scan causal property
# ---------------------------------------------------------------------------

def test_selective_scan_causal():
    """Output at position t must not depend on inputs after t."""
    B, L, d_inner, d_state, dt_rank = 1, 8, 16, 4, 2
    torch.manual_seed(42)
    u = torch.randn(B, L, d_inner)
    delta = torch.randn(B, L, dt_rank)
    A = torch.randn(d_inner, d_state)
    B_mat = torch.randn(B, L, d_state)
    C = torch.randn(B, L, d_state)
    D = torch.randn(d_inner)

    out_full = selective_scan_naive(u, delta, A, B_mat, C, D)

    # Perturb inputs at positions >= 4 and verify first 4 outputs are unchanged
    u2 = u.clone()
    u2[:, 4:, :] = torch.randn_like(u2[:, 4:, :])
    out_perturbed = selective_scan_naive(u2, delta, A, B_mat, C, D)

    assert torch.allclose(out_full[:, :4, :], out_perturbed[:, :4, :], atol=1e-5), (
        "Output before perturbation point changed — scan is not causal!"
    )


# ---------------------------------------------------------------------------
# 4. MambaBlock output shape
# ---------------------------------------------------------------------------

def test_mamba_block_output_shape(small_cfg):
    """MambaBlock must return (B, L, d_model)."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    out = block(x)
    assert out.shape == (2, 16, small_cfg.d_model), f"Got shape {out.shape}"


# ---------------------------------------------------------------------------
# 5. MambaBlock forward returns Tensor, not tuple
# ---------------------------------------------------------------------------

def test_mamba_block_no_aux_loss(small_cfg):
    """MambaBlock.forward() must return a plain Tensor (no aux_loss tuple)."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 8, small_cfg.d_model)
    out = block(x)
    assert isinstance(out, torch.Tensor), (
        f"Expected Tensor, got {type(out)}. MambaBlock must not return a tuple."
    )


# ---------------------------------------------------------------------------
# 6. MambaBlock gradients flow
# ---------------------------------------------------------------------------

def test_mamba_block_gradients_flow(small_cfg):
    """backward() must run without error and produce non-None gradients."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 8, small_cfg.d_model, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed back to input"
    # At least one parameter should have a gradient
    param_grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed"


# ---------------------------------------------------------------------------
# 7. SSM state dimension
# ---------------------------------------------------------------------------

def test_ssm_state_dimension(ssm_cfg):
    """SelectiveSSM internal state should have d_state dimension."""
    ssm = SelectiveSSM(ssm_cfg)
    # A_log parameter should be (d_inner, d_state)
    d_inner = ssm_cfg.d_model * ssm_cfg.expand
    assert ssm.A_log.shape == (d_inner, ssm_cfg.d_state), (
        f"A_log shape {ssm.A_log.shape} != ({d_inner}, {ssm_cfg.d_state})"
    )


# ---------------------------------------------------------------------------
# 8. MambaBlock works for L=1, 16, 128
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [1, 16, 128])
def test_mamba_block_different_seq_lens(small_cfg, seq_len):
    """MambaBlock must handle varied sequence lengths without error."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, seq_len, small_cfg.d_model)
    out = block(x)
    assert out.shape == (2, seq_len, small_cfg.d_model)


# ---------------------------------------------------------------------------
# 9. selective_scan_naive shape
# ---------------------------------------------------------------------------

def test_selective_scan_naive_shape():
    """Standalone test that selective_scan_naive output shape matches spec."""
    B, L, d_inner, d_state, dt_rank = 3, 12, 24, 6, 3
    u = torch.zeros(B, L, d_inner)
    delta = torch.zeros(B, L, dt_rank)
    A = torch.zeros(d_inner, d_state)
    B_mat = torch.zeros(B, L, d_state)
    C = torch.zeros(B, L, d_state)
    D = torch.ones(d_inner)
    out = selective_scan_naive(u, delta, A, B_mat, C, D)
    assert out.shape == (B, L, d_inner)


# ---------------------------------------------------------------------------
# 10. MambaBlock as drop-in FFN replacement in TransformerBlock
# ---------------------------------------------------------------------------

def test_mamba_block_as_ffn_replacement(small_cfg):
    """MambaBlock can replace SwiGLUFFN in TransformerBlock.ffn."""
    from src.model.transformer import TransformerBlock
    from src.model.attention import precompute_rope_frequencies

    block = TransformerBlock(small_cfg)
    # Replace FFN with MambaBlock
    block.ffn = MambaBlock(small_cfg)

    B, L = 2, 8
    x = torch.randn(B, L, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, small_cfg.max_seq_len, small_cfg.rope_theta)
    freqs_cis = freqs[:L]

    out, kv = block(x, freqs_cis)
    assert out.shape == (B, L, small_cfg.d_model), f"Got {out.shape}"
    assert isinstance(kv, tuple) and len(kv) == 2
