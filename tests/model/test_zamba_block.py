"""Unit tests for ZambaBlock — Mamba SSM layers with periodic shared attention.

Tiny config: d_model=64, n_heads=4, head_dim=16, d_state=16.
"""

from __future__ import annotations

import pytest
import torch

from src.model.zamba_block import (
    ZambaBlock,
    ZambaConfig,
    ZambaSharedAttention,
    ZambaSSMLayer,
)
from src.model import MODEL_COMPONENT_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_D_MODEL = 64
TINY_N_HEADS = 4
TINY_HEAD_DIM = 16
TINY_D_STATE = 16


@pytest.fixture
def tiny_cfg() -> ZambaConfig:
    return ZambaConfig(
        d_model=TINY_D_MODEL,
        d_state=TINY_D_STATE,
        d_conv=4,
        expand=2,
        n_heads=TINY_N_HEADS,
        head_dim=TINY_HEAD_DIM,
        attn_every_n=6,
        n_layers=26,
    )


@pytest.fixture
def block(tiny_cfg: ZambaConfig) -> ZambaBlock:
    torch.manual_seed(42)
    return ZambaBlock(tiny_cfg)


@pytest.fixture
def small_block() -> ZambaBlock:
    """Smaller block for faster gradient tests."""
    cfg = ZambaConfig(
        d_model=TINY_D_MODEL,
        d_state=TINY_D_STATE,
        d_conv=4,
        expand=2,
        n_heads=TINY_N_HEADS,
        head_dim=TINY_HEAD_DIM,
        attn_every_n=4,
        n_layers=8,
    )
    torch.manual_seed(0)
    return ZambaBlock(cfg)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ZambaConfig()
    assert cfg.attn_every_n == 6
    assert cfg.n_layers == 26


# ---------------------------------------------------------------------------
# 2. test_output_shape
# ---------------------------------------------------------------------------

def test_output_shape(block: ZambaBlock):
    x = torch.randn(2, 8, TINY_D_MODEL)
    out = block(x)
    assert out.shape == (2, 8, TINY_D_MODEL), f"Expected (2, 8, {TINY_D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_n_attention_layers
# ---------------------------------------------------------------------------

def test_n_attention_layers():
    cfg = ZambaConfig(
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        head_dim=TINY_HEAD_DIM,
        d_state=TINY_D_STATE,
        attn_every_n=4,
        n_layers=12,
    )
    b = ZambaBlock(cfg)
    # Positions 3, 7, 11 → 3 attention layers
    assert b.n_attention_layers() == 3


# ---------------------------------------------------------------------------
# 4. test_n_ssm_layers
# ---------------------------------------------------------------------------

def test_n_ssm_layers():
    cfg = ZambaConfig(
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        head_dim=TINY_HEAD_DIM,
        d_state=TINY_D_STATE,
        attn_every_n=4,
        n_layers=12,
    )
    b = ZambaBlock(cfg)
    expected_ssm = 12 - 3  # total - attention
    assert b.n_ssm_layers() == expected_ssm
    assert b.n_ssm_layers() + b.n_attention_layers() == 12


# ---------------------------------------------------------------------------
# 5. test_shared_attention_same_params
# ---------------------------------------------------------------------------

def test_shared_attention_same_params(block: ZambaBlock):
    # shared_attn is a single module — its parameters are the same object at
    # every injection point (not separate copies).
    attn = block.shared_attn
    q_id = id(attn.q_proj.weight)
    k_id = id(attn.k_proj.weight)
    # The object identity must remain stable (same module reference).
    assert id(block.shared_attn.q_proj.weight) == q_id
    assert id(block.shared_attn.k_proj.weight) == k_id


# ---------------------------------------------------------------------------
# 6. test_ssm_layer_output_shape
# ---------------------------------------------------------------------------

def test_ssm_layer_output_shape():
    torch.manual_seed(1)
    ssm = ZambaSSMLayer(d_model=TINY_D_MODEL, d_state=TINY_D_STATE, d_conv=4, expand=2)
    x = torch.randn(2, 8, TINY_D_MODEL)
    out = ssm(x)
    assert out.shape == (2, 8, TINY_D_MODEL)


# ---------------------------------------------------------------------------
# 7. test_attention_layer_output_shape
# ---------------------------------------------------------------------------

def test_attention_layer_output_shape():
    torch.manual_seed(2)
    attn = ZambaSharedAttention(d_model=TINY_D_MODEL, n_heads=TINY_N_HEADS, head_dim=TINY_HEAD_DIM)
    x = torch.randn(2, 8, TINY_D_MODEL)
    out = attn(x)
    assert out.shape == (2, 8, TINY_D_MODEL)


# ---------------------------------------------------------------------------
# 8. test_gradients_flow
# ---------------------------------------------------------------------------

def test_gradients_flow(small_block: ZambaBlock):
    x = torch.randn(2, 4, TINY_D_MODEL, requires_grad=True)
    out = small_block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed back to input"
    assert x.grad.shape == x.shape
    # At least some parameters should have gradients
    params_with_grad = [p for p in small_block.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0, "No parameter gradients computed"


# ---------------------------------------------------------------------------
# 9. test_shared_attn_single_module
# ---------------------------------------------------------------------------

def test_shared_attn_single_module(block: ZambaBlock):
    assert isinstance(block.shared_attn, ZambaSharedAttention), (
        f"block.shared_attn should be ZambaSharedAttention, got {type(block.shared_attn)}"
    )
    # Must NOT be a ModuleList
    assert not isinstance(block.shared_attn, torch.nn.ModuleList)


# ---------------------------------------------------------------------------
# 10. test_parameter_sharing_ratio
# ---------------------------------------------------------------------------

def test_parameter_sharing_ratio(block: ZambaBlock):
    ratio = block.parameter_sharing_ratio()
    assert 0.0 < ratio < 1.0, f"Sharing ratio should be in (0,1), got {ratio}"


# ---------------------------------------------------------------------------
# 11. test_attn_every_1
# ---------------------------------------------------------------------------

def test_attn_every_1():
    """When attn_every_n=1, every layer is attention → 0 SSM layers."""
    cfg = ZambaConfig(
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        head_dim=TINY_HEAD_DIM,
        d_state=TINY_D_STATE,
        attn_every_n=1,
        n_layers=4,
    )
    b = ZambaBlock(cfg)
    assert b.n_ssm_layers() == 0
    assert b.n_attention_layers() == 4

    # Forward should still work (all attention)
    x = torch.randn(1, 4, TINY_D_MODEL)
    out = b(x)
    assert out.shape == (1, 4, TINY_D_MODEL)


# ---------------------------------------------------------------------------
# 12. test_batch_size_one
# ---------------------------------------------------------------------------

def test_batch_size_one(small_block: ZambaBlock):
    x = torch.randn(1, 8, TINY_D_MODEL)
    out = small_block(x)
    assert out.shape == (1, 8, TINY_D_MODEL)


# ---------------------------------------------------------------------------
# 13. test_seq_len_one
# ---------------------------------------------------------------------------

def test_seq_len_one(small_block: ZambaBlock):
    x = torch.randn(2, 1, TINY_D_MODEL)
    out = small_block(x)
    assert out.shape == (2, 1, TINY_D_MODEL)


# ---------------------------------------------------------------------------
# 14. test_determinism
# ---------------------------------------------------------------------------

def test_determinism(small_block: ZambaBlock):
    small_block.train(False)
    x = torch.randn(2, 6, TINY_D_MODEL)
    with torch.no_grad():
        out1 = small_block(x)
        out2 = small_block(x)
    assert torch.allclose(out1, out2, atol=1e-6), "Output not deterministic in eval mode"


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------

def test_registry():
    assert "zamba_block" in MODEL_COMPONENT_REGISTRY, (
        "zamba_block not found in MODEL_COMPONENT_REGISTRY"
    )
    assert MODEL_COMPONENT_REGISTRY["zamba_block"] is ZambaBlock
