"""Integration tests for ZambaBlock.

Build ZambaBlock(d_model=64, n_heads=4, head_dim=16, d_state=16,
                 attn_every_n=3, n_layers=6).

Input: [2, 8, 64].
Verify: output shape, n_attention_layers=2, n_ssm_layers=4, backward works,
        shared_attn is a single module, registry wired.
"""

from __future__ import annotations

import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.zamba_block import ZambaBlock, ZambaConfig, ZambaSharedAttention

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
HEAD_DIM = 16
D_STATE = 16
ATTN_EVERY_N = 3
N_LAYERS = 6


def make_block() -> ZambaBlock:
    torch.manual_seed(99)
    cfg = ZambaConfig(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=4,
        expand=2,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        attn_every_n=ATTN_EVERY_N,
        n_layers=N_LAYERS,
    )
    return ZambaBlock(cfg)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_integration_output_shape():
    """Full forward pass produces the correct output shape."""
    blk = make_block()
    x = torch.randn(2, 8, D_MODEL)
    out = blk(x)
    assert out.shape == (2, 8, D_MODEL), f"Expected output shape (2, 8, {D_MODEL}), got {out.shape}"


def test_integration_n_attention_layers():
    """With attn_every_n=3, n_layers=6: positions 2 and 5 are attention → 2."""
    blk = make_block()
    assert blk.n_attention_layers() == 2, (
        f"Expected 2 attention layers, got {blk.n_attention_layers()}"
    )


def test_integration_n_ssm_layers():
    """With 6 total layers and 2 attention positions → 4 SSM layers."""
    blk = make_block()
    assert blk.n_ssm_layers() == 4, f"Expected 4 SSM layers, got {blk.n_ssm_layers()}"
    assert blk.n_ssm_layers() + blk.n_attention_layers() == N_LAYERS


def test_integration_backward():
    """Loss.backward() succeeds; input and param gradients are non-None."""
    blk = make_block()
    x = torch.randn(2, 8, D_MODEL, requires_grad=True)
    out = blk(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient on input tensor"
    assert x.grad.shape == x.shape

    params_with_grad = [p for p in blk.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0, "No parameter gradients after backward"


def test_integration_shared_attn_is_single_module():
    """shared_attn must be a ZambaSharedAttention instance, not a list."""
    blk = make_block()
    assert isinstance(blk.shared_attn, ZambaSharedAttention), (
        f"shared_attn type: {type(blk.shared_attn)}"
    )
    assert not isinstance(blk.shared_attn, torch.nn.ModuleList)


def test_integration_registry_wired():
    """MODEL_COMPONENT_REGISTRY['zamba_block'] points to ZambaBlock."""
    assert "zamba_block" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["zamba_block"] is ZambaBlock


def test_integration_parameter_sharing_ratio():
    """Sharing ratio is in (0, 1) — one attn covers multiple positions."""
    blk = make_block()
    ratio = blk.parameter_sharing_ratio()
    assert 0.0 < ratio < 1.0, f"Sharing ratio out of range: {ratio}"


def test_integration_eval_determinism():
    """Same input in eval mode always produces same output."""
    blk = make_block()
    blk.train(False)
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out_a = blk(x)
        out_b = blk(x)
    assert torch.allclose(out_a, out_b, atol=1e-6), "Non-deterministic output in eval mode"
