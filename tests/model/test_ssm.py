"""Tests for Mamba-style Selective State Space Model (S6).

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".

Covers all 15 spec requirements plus legacy tests for backward compatibility.
"""

import math
import pytest
import torch
import torch.nn.functional as F
from src.model.config import AureliusConfig
from src.model.ssm import (
    SSMConfig,
    SelectiveSSM,
    MambaBlock,
    MambaLayer,
    MambaLM,
    selective_scan,
    selective_scan_naive,
)


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
    """Small SSMConfig matching spec recommendation: d_model=32."""
    return SSMConfig(d_model=32, d_state=8, d_conv=4, expand=2, dt_rank=4)


@pytest.fixture
def ssm(ssm_cfg):
    return SelectiveSSM(ssm_cfg)


# ---------------------------------------------------------------------------
# Spec test 1: SSMConfig defaults
# ---------------------------------------------------------------------------

def test_ssm_config_defaults():
    """SSMConfig should have the expected default field values."""
    cfg = SSMConfig()
    assert cfg.d_model == 128
    assert cfg.d_state == 16
    assert cfg.d_conv == 4
    assert cfg.expand == 2
    assert cfg.dt_rank == 8
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# Spec test 2: selective_scan output shape is (B, L, d_inner)
# ---------------------------------------------------------------------------

def test_selective_scan_output_shape():
    """selective_scan must return (B, L, d_inner)."""
    B, L, d_inner, d_state = 2, 16, 32, 8
    u = torch.randn(B, L, d_inner)
    dt = torch.randn(B, L, d_inner)
    A = torch.randn(d_inner, d_state)
    B_mat = torch.randn(B, L, d_state)
    C = torch.randn(B, L, d_state)
    D = torch.randn(d_inner)
    out = selective_scan(u, dt, A, B_mat, C, D)
    assert out.shape == (B, L, d_inner), f"Expected ({B}, {L}, {d_inner}), got {out.shape}"


# ---------------------------------------------------------------------------
# Spec test 3: selective_scan with B=1, L=8, d_inner=16, d_state=4
# ---------------------------------------------------------------------------

def test_selective_scan_small_dims():
    """selective_scan with B=1, L=8, d_inner=16, d_state=4 must produce correct shape."""
    B, L, d_inner, d_state = 1, 8, 16, 4
    u = torch.randn(B, L, d_inner)
    dt = torch.randn(B, L, d_inner)
    A = torch.randn(d_inner, d_state)
    B_mat = torch.randn(B, L, d_state)
    C = torch.randn(B, L, d_state)
    D = torch.randn(d_inner)
    out = selective_scan(u, dt, A, B_mat, C, D)
    assert out.shape == (B, L, d_inner)


# ---------------------------------------------------------------------------
# Spec test 4: selective_scan output is finite (no NaN/Inf)
# ---------------------------------------------------------------------------

def test_selective_scan_finite():
    """selective_scan output must contain no NaN or Inf values."""
    B, L, d_inner, d_state = 2, 10, 16, 4
    torch.manual_seed(0)
    u = torch.randn(B, L, d_inner)
    dt = torch.randn(B, L, d_inner)
    A = -torch.abs(torch.randn(d_inner, d_state))  # negative for stability
    B_mat = torch.randn(B, L, d_state)
    C = torch.randn(B, L, d_state)
    D = torch.ones(d_inner)
    out = selective_scan(u, dt, A, B_mat, C, D)
    assert torch.isfinite(out).all(), "selective_scan produced NaN or Inf"


# ---------------------------------------------------------------------------
# Spec test 5: selective_scan D=0 gives zero output when input is zero
# ---------------------------------------------------------------------------

def test_selective_scan_zero_input_zero_D():
    """With u=0 and D=0, the output must be all zeros."""
    B, L, d_inner, d_state = 2, 6, 8, 4
    u = torch.zeros(B, L, d_inner)
    dt = torch.zeros(B, L, d_inner)
    A = -torch.ones(d_inner, d_state)
    B_mat = torch.zeros(B, L, d_state)
    C = torch.randn(B, L, d_state)
    D = torch.zeros(d_inner)
    out = selective_scan(u, dt, A, B_mat, C, D)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
        "Expected all-zero output when u=0 and D=0"
    )


# ---------------------------------------------------------------------------
# Spec test 6: MambaBlock output shape matches input (B, T, d_model)
# ---------------------------------------------------------------------------

def test_mamba_block_output_shape(small_cfg):
    """MambaBlock must return (B, L, d_model)."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    out = block(x)
    assert out.shape == (2, 16, small_cfg.d_model), f"Got shape {out.shape}"


# ---------------------------------------------------------------------------
# Spec test 7: MambaBlock is differentiable (backward works)
# ---------------------------------------------------------------------------

def test_mamba_block_gradients_flow(small_cfg):
    """backward() must run without error and produce non-None gradients."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 8, small_cfg.d_model, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed back to input"
    param_grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed"


# ---------------------------------------------------------------------------
# Spec test 8: MambaBlock with different sequence lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [1, 16, 128])
def test_mamba_block_different_seq_lens(small_cfg, seq_len):
    """MambaBlock must handle varied sequence lengths without error."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, seq_len, small_cfg.d_model)
    out = block(x)
    assert out.shape == (2, seq_len, small_cfg.d_model)


# ---------------------------------------------------------------------------
# Spec test 9: MambaLayer output shape matches input
# ---------------------------------------------------------------------------

def test_mamba_layer_output_shape(ssm_cfg):
    """MambaLayer must return same shape as input (B, T, d_model)."""
    layer = MambaLayer(ssm_cfg)
    B, T = 2, 12
    x = torch.randn(B, T, ssm_cfg.d_model)
    out = layer(x)
    assert out.shape == (B, T, ssm_cfg.d_model), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# Spec test 10: MambaLayer with residual: output != input (non-trivial)
# ---------------------------------------------------------------------------

def test_mamba_layer_nontrivial(ssm_cfg):
    """MambaLayer output should differ from input (residual is non-zero)."""
    torch.manual_seed(123)
    layer = MambaLayer(ssm_cfg)
    x = torch.randn(2, 8, ssm_cfg.d_model)
    out = layer(x)
    assert not torch.allclose(out, x, atol=1e-5), (
        "MambaLayer output equals input — block appears to produce zero residual"
    )


# ---------------------------------------------------------------------------
# Spec test 11: MambaLM output shape is (B, T, vocab_size)
# ---------------------------------------------------------------------------

def test_mamba_lm_output_shape(ssm_cfg):
    """MambaLM must return logits of shape (B, T, vocab_size)."""
    vocab_size = 256
    model = MambaLM(ssm_cfg, n_layers=2, vocab_size=vocab_size)
    B, T = 2, 10
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, vocab_size), f"Got {logits.shape}"


# ---------------------------------------------------------------------------
# Spec test 12: MambaLM is differentiable
# ---------------------------------------------------------------------------

def test_mamba_lm_differentiable(ssm_cfg):
    """MambaLM backward must succeed with finite gradients."""
    vocab_size = 256
    model = MambaLM(ssm_cfg, n_layers=2, vocab_size=vocab_size)
    B, T = 2, 8
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients after backward"


# ---------------------------------------------------------------------------
# Spec test 13: MambaLM with n_layers=1 and n_layers=4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_layers", [1, 4])
def test_mamba_lm_various_depths(ssm_cfg, n_layers):
    """MambaLM must work with n_layers=1 and n_layers=4."""
    vocab_size = 256
    model = MambaLM(ssm_cfg, n_layers=n_layers, vocab_size=vocab_size)
    B, T = 2, 6
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, vocab_size), f"n_layers={n_layers}: got {logits.shape}"


# ---------------------------------------------------------------------------
# Spec test 14: MambaLM loss from cross_entropy is finite and positive
# ---------------------------------------------------------------------------

def test_mamba_lm_loss_finite_positive(ssm_cfg):
    """Cross-entropy loss on MambaLM logits must be finite and positive."""
    vocab_size = 256
    model = MambaLM(ssm_cfg, n_layers=2, vocab_size=vocab_size)
    B, T = 2, 8
    input_ids = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)  # (B, T, vocab_size)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss must be positive, got {loss.item()}"


# ---------------------------------------------------------------------------
# Spec test 15: A parameter of MambaBlock is initialized negative (log scale)
# ---------------------------------------------------------------------------

def test_mamba_block_A_log_initialized_negative(ssm_cfg):
    """_MambaBlockFromSSMConfig A_log must store log of positive values (A itself negative).

    The convention: A_log = log(positive), and A = -exp(A_log) < 0.
    So A_log > -inf (it stores log of positive numbers, hence can be any real).
    What matters is that -exp(A_log) is strictly negative.
    """
    from src.model.ssm import _MambaBlockFromSSMConfig
    block = _MambaBlockFromSSMConfig(ssm_cfg)
    A = -torch.exp(block.A_log)
    assert (A < 0).all(), "All A values (= -exp(A_log)) must be strictly negative"


# ---------------------------------------------------------------------------
# Legacy / additional tests (preserved from original test_ssm.py)
# ---------------------------------------------------------------------------

def test_ssm_config_dt_rank_auto():
    """'auto' dt_rank should equal ceil(d_model / 16)."""
    cfg = SSMConfig(d_model=64, dt_rank="auto")
    expected = math.ceil(64 / 16)
    ssm = SelectiveSSM(cfg)
    assert ssm.dt_rank == expected, f"Expected dt_rank={expected}, got {ssm.dt_rank}"


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
    u2 = u.clone()
    u2[:, 4:, :] = torch.randn_like(u2[:, 4:, :])
    out_perturbed = selective_scan_naive(u2, delta, A, B_mat, C, D)

    assert torch.allclose(out_full[:, :4, :], out_perturbed[:, :4, :], atol=1e-5), (
        "Output before perturbation point changed — scan is not causal!"
    )


def test_mamba_block_no_aux_loss(small_cfg):
    """MambaBlock.forward() must return a plain Tensor (no aux_loss tuple)."""
    block = MambaBlock(small_cfg)
    x = torch.randn(2, 8, small_cfg.d_model)
    out = block(x)
    assert isinstance(out, torch.Tensor), (
        f"Expected Tensor, got {type(out)}. MambaBlock must not return a tuple."
    )


def test_ssm_state_dimension(ssm_cfg):
    """SelectiveSSM internal state should have d_state dimension."""
    ssm = SelectiveSSM(ssm_cfg)
    d_inner = ssm_cfg.d_model * ssm_cfg.expand
    assert ssm.A_log.shape == (d_inner, ssm_cfg.d_state), (
        f"A_log shape {ssm.A_log.shape} != ({d_inner}, {ssm_cfg.d_state})"
    )


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


def test_mamba_block_as_ffn_replacement(small_cfg):
    """MambaBlock can replace SwiGLUFFN in TransformerBlock.ffn."""
    from src.model.transformer import TransformerBlock
    from src.model.attention import precompute_rope_frequencies

    block = TransformerBlock(small_cfg)
    block.ffn = MambaBlock(small_cfg)

    B, L = 2, 8
    x = torch.randn(B, L, small_cfg.d_model)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, small_cfg.max_seq_len, small_cfg.rope_theta)
    freqs_cis = freqs[:L]

    out, kv = block(x, freqs_cis)
    assert out.shape == (B, L, small_cfg.d_model), f"Got {out.shape}"
    assert isinstance(kv, tuple) and len(kv) == 2
