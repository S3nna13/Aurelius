"""
Tests for src/model/state_space_model.py

All tests use small dimensions:
  d_model=16, vocab_size=16, n_layers=2, d_state=4, d_conv=4, expand=2
  B=2 (batch), T=8 (sequence length)
"""

import math
import torch
import torch.nn as nn
import pytest

from src.model.state_space_model import (
    RMSNorm,
    SSMKernel,
    S6Block,
    MambaBlock,
    MambaLanguageModel,
    MambaConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
D_STATE = 4
D_CONV = 4
EXPAND = 2
B = 2
T = 8
DT_RANK = max((D_MODEL * EXPAND) // 16, 1)


def _make_input(d_model: int = D_MODEL) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, d_model)


def _make_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ===========================================================================
# RMSNorm tests
# ===========================================================================

def test_rmsnorm_output_shape():
    """RMSNorm should not change the tensor shape."""
    norm = RMSNorm(D_MODEL)
    x = _make_input()
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_unit_rms():
    """After RMSNorm (with weight=1), the RMS along the last dim should be ~1."""
    norm = RMSNorm(D_MODEL)
    nn.init.ones_(norm.weight)  # ensure weight is exactly 1
    x = _make_input()
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_learnable_weight():
    """RMSNorm weight parameter should be part of the module's parameters."""
    norm = RMSNorm(D_MODEL)
    param_names = [n for n, _ in norm.named_parameters()]
    assert "weight" in param_names


# ===========================================================================
# SSMKernel tests
# ===========================================================================

def _make_ssm_kernel(d_model: int = D_MODEL) -> SSMKernel:
    torch.manual_seed(2)
    d_inner = d_model * EXPAND
    return SSMKernel(d_inner, D_STATE, dt_rank=DT_RANK)


def test_ssm_discretize_A_bar_shape():
    """SSMKernel.discretize should return A_bar of shape [B, d_model, d_state]."""
    d_inner = D_MODEL * EXPAND
    kernel = _make_ssm_kernel()
    A = -torch.exp(kernel.A_log)                     # [d_inner, d_state]
    B_mat = torch.randn(B, d_inner, D_STATE)
    dt = torch.rand(B, d_inner).add(0.01)
    A_bar, B_bar = kernel.discretize(A, B_mat, dt)
    assert A_bar.shape == (B, d_inner, D_STATE)


def test_ssm_discretize_B_bar_shape():
    """SSMKernel.discretize should return B_bar of shape [B, d_model, d_state]."""
    d_inner = D_MODEL * EXPAND
    kernel = _make_ssm_kernel()
    A = -torch.exp(kernel.A_log)
    B_mat = torch.randn(B, d_inner, D_STATE)
    dt = torch.rand(B, d_inner).add(0.01)
    _, B_bar = kernel.discretize(A, B_mat, dt)
    assert B_bar.shape == (B, d_inner, D_STATE)


def test_ssm_selective_scan_output_shape():
    """SSMKernel.selective_scan output should be [B, T, d_model]."""
    d_inner = D_MODEL * EXPAND
    kernel = _make_ssm_kernel()
    u = torch.randn(B, T, d_inner)
    y = kernel.selective_scan(u)
    assert y.shape == (B, T, d_inner)


def test_ssm_selective_scan_gradient_flows():
    """Gradients must flow back to SSMKernel parameters."""
    d_inner = D_MODEL * EXPAND
    kernel = _make_ssm_kernel()
    u = torch.randn(B, T, d_inner, requires_grad=True)
    y = kernel.selective_scan(u)
    loss = y.sum()
    loss.backward()
    assert u.grad is not None
    assert u.grad.shape == u.shape
    # At least one parameter gradient should be non-None and non-zero
    param_grads = [p.grad for p in kernel.parameters() if p.grad is not None]
    assert len(param_grads) > 0


def test_ssm_D_skip_contributes_to_output():
    """Setting D to zero should change the output, proving D contributes."""
    d_inner = D_MODEL * EXPAND
    kernel = _make_ssm_kernel()
    torch.manual_seed(3)
    u = torch.randn(B, T, d_inner)

    with torch.no_grad():
        y_with_D = kernel.selective_scan(u).clone()
        kernel.D.fill_(0.0)
        y_no_D = kernel.selective_scan(u).clone()

    assert not torch.allclose(y_with_D, y_no_D)


# ===========================================================================
# S6Block tests
# ===========================================================================

def _make_s6block() -> S6Block:
    torch.manual_seed(4)
    return S6Block(D_MODEL, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND)


def test_s6block_output_shape():
    """S6Block output should be [B, T, d_model]."""
    block = _make_s6block()
    x = _make_input()
    y = block(x)
    assert y.shape == (B, T, D_MODEL)


def test_s6block_conv1d_trimmed_to_T():
    """The conv1d output must be trimmed so that the time dimension equals T."""
    block = _make_s6block()
    x = _make_input()
    # Access internal conv directly to verify trim
    xz = block.in_proj(x)
    x_proj, _ = xz.chunk(2, dim=-1)
    raw_conv = block.conv1d(x_proj.transpose(1, 2))  # [B, d_inner, T+pad]
    trimmed = raw_conv[..., :T]
    assert trimmed.shape[-1] == T
    assert raw_conv.shape[-1] > T or raw_conv.shape[-1] == T  # pad >= 0


def test_s6block_gradient_flows():
    """Gradients must flow end-to-end through S6Block."""
    block = _make_s6block()
    x = _make_input().requires_grad_(True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_s6block_different_inputs_different_outputs():
    """S6Block should produce different outputs for different inputs."""
    block = _make_s6block()
    x1 = _make_input()
    x2 = x1 + 1.0
    with torch.no_grad():
        y1 = block(x1)
        y2 = block(x2)
    assert not torch.allclose(y1, y2)


# ===========================================================================
# MambaBlock tests
# ===========================================================================

def _make_mamba_block() -> MambaBlock:
    torch.manual_seed(5)
    return MambaBlock(D_MODEL, d_state=D_STATE)


def test_mambablock_output_shape():
    """MambaBlock output should be [B, T, d_model]."""
    block = _make_mamba_block()
    x = _make_input()
    y = block(x)
    assert y.shape == (B, T, D_MODEL)


def test_mambablock_residual_differs_from_input():
    """With a residual connection, MambaBlock output should differ from input."""
    block = _make_mamba_block()
    x = _make_input()
    with torch.no_grad():
        y = block(x)
    assert not torch.allclose(y, x)


def test_mambablock_prenorm_uses_rmsnorm():
    """MambaBlock.norm should be an RMSNorm instance."""
    block = _make_mamba_block()
    assert isinstance(block.norm, RMSNorm)


def test_mambablock_prenorm_applied_before_s6():
    """Pre-norm: passing normalised x directly to s6 should match block internals."""
    block = _make_mamba_block()
    x = _make_input()
    with torch.no_grad():
        # Manually compute: x + s6(norm(x))
        expected = x + block.s6(block.norm(x))
        actual = block(x)
    assert torch.allclose(actual, expected, atol=1e-6)


# ===========================================================================
# MambaLanguageModel tests
# ===========================================================================

def _make_lm() -> MambaLanguageModel:
    torch.manual_seed(6)
    return MambaLanguageModel(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        d_state=D_STATE,
    )


def test_mamba_lm_forward_output_shape():
    """MambaLanguageModel forward should return [B, T, vocab_size]."""
    model = _make_lm()
    ids = _make_ids()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE)


def test_mamba_lm_compute_loss_finite_positive():
    """compute_loss should return a finite positive scalar."""
    model = _make_lm()
    ids = _make_ids()
    loss = model.compute_loss(ids)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
    assert math.isfinite(loss.item())


def test_mamba_lm_compute_loss_backward():
    """Backward pass through compute_loss must succeed and produce gradients."""
    model = _make_lm()
    ids = _make_ids()
    loss = model.compute_loss(ids)
    loss.backward()
    # Check that at least one parameter received a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_mamba_lm_embedding_used():
    """The embedding layer should be an nn.Embedding with correct vocab/d_model."""
    model = _make_lm()
    assert isinstance(model.embedding, nn.Embedding)
    assert model.embedding.num_embeddings == VOCAB_SIZE
    assert model.embedding.embedding_dim == D_MODEL


# ===========================================================================
# MambaConfig tests
# ===========================================================================

def test_mamba_config_defaults():
    """MambaConfig should have the specified default values."""
    cfg = MambaConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 4
    assert cfg.d_state == 8
    assert cfg.d_conv == 4
    assert cfg.expand == 2


def test_mamba_config_custom_values():
    """MambaConfig should accept and store custom values."""
    cfg = MambaConfig(d_model=64, vocab_size=256, n_layers=8, d_state=16)
    assert cfg.d_model == 64
    assert cfg.vocab_size == 256
    assert cfg.n_layers == 8
    assert cfg.d_state == 16
