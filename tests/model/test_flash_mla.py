"""Unit tests for FlashMLAAttention (15 tests).

Tiny config used throughout: d_model=64, n_heads=4, kv_lrank=16, head_dim=16.
Run with: .venv/bin/python3.14 -m pytest tests/model/test_flash_mla.py -v
"""
from __future__ import annotations

import torch
import pytest

from src.model.flash_mla import FlashMLAAttention, FlashMLAConfig
from src.model import MODEL_COMPONENT_REGISTRY


# ---------------------------------------------------------------------------
# Shared tiny config (kv_lrank=16, n_heads=4, head_dim=16 → ratio=0.25)
# ---------------------------------------------------------------------------

TINY_CFG = FlashMLAConfig(
    d_model=64,
    n_heads=4,
    head_dim=16,
    kv_lrank=16,
    q_lrank=32,
    rope_dim=8,
    dropout=0.0,
)


def make_model(cfg: FlashMLAConfig | None = None) -> FlashMLAAttention:
    if cfg is None:
        cfg = TINY_CFG
    return FlashMLAAttention(cfg).eval()


def tiny_input(B: int = 2, T: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# Test 1 - config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = FlashMLAConfig()
    assert cfg.kv_lrank == 512
    assert cfg.q_lrank == 1536
    assert cfg.d_model == 2048
    assert cfg.n_heads == 16
    assert cfg.head_dim == 128


# ---------------------------------------------------------------------------
# Test 2 - output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    model = make_model()
    x = tiny_input(B=2, T=8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8, TINY_CFG.d_model), f"Expected (2,8,64), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 3 - compress_kv shape
# ---------------------------------------------------------------------------

def test_compress_kv_shape():
    model = make_model()
    x = tiny_input(B=3, T=5)
    with torch.no_grad():
        c = model._compress_kv(x)
    assert c.shape == (3, 5, TINY_CFG.kv_lrank), f"Expected (3,5,16), got {c.shape}"


# ---------------------------------------------------------------------------
# Test 4 - standard path runs
# ---------------------------------------------------------------------------

def test_standard_path_runs():
    model = make_model()
    x = tiny_input()
    with torch.no_grad():
        out = model(x, use_absorbed=False)
    assert out.shape == (2, 8, TINY_CFG.d_model)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 5 - absorbed path runs
# ---------------------------------------------------------------------------

def test_absorbed_path_runs():
    model = make_model()
    model.absorb_projections()
    x = tiny_input()
    with torch.no_grad():
        out = model(x, use_absorbed=True)
    assert out.shape == (2, 8, TINY_CFG.d_model)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 6 - absorbed path matches standard path (rtol=1e-4)
# ---------------------------------------------------------------------------

def test_absorbed_matches_standard():
    torch.manual_seed(42)
    model = make_model()
    model.absorb_projections()
    x = tiny_input()
    with torch.no_grad():
        out_std = model(x, use_absorbed=False)
        out_abs = model(x, use_absorbed=True)
    assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5), (
        f"Max diff: {(out_std - out_abs).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7 - kv_cache_size_ratio
# ---------------------------------------------------------------------------

def test_kv_cache_ratio():
    # kv_lrank=16, n_heads=4, head_dim=16 -> 16 / (4*16) = 0.25
    model = make_model()
    ratio = model.kv_cache_size_ratio()
    assert abs(ratio - 0.25) < 1e-9, f"Expected 0.25, got {ratio}"


# ---------------------------------------------------------------------------
# Test 8 - gradients flow
# ---------------------------------------------------------------------------

def test_gradients_flow():
    model = FlashMLAAttention(TINY_CFG).train()
    x = tiny_input().requires_grad_(True)
    out = model(x, use_absorbed=False)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0


# ---------------------------------------------------------------------------
# Test 9 - absorb_projections is idempotent
# ---------------------------------------------------------------------------

def test_absorb_idempotent():
    model = make_model()
    model.absorb_projections()
    first = model.absorbed_qk.clone()
    model.absorb_projections()
    second = model.absorbed_qk.clone()
    assert torch.allclose(first, second, atol=1e-9), "Double absorb changed the matrix"
    x = tiny_input()
    with torch.no_grad():
        out = model(x, use_absorbed=True)
    assert out.shape == (2, 8, TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# Test 10 - batch size one
# ---------------------------------------------------------------------------

def test_batch_size_one():
    model = make_model()
    x = torch.randn(1, 8, TINY_CFG.d_model)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 8, TINY_CFG.d_model)


# ---------------------------------------------------------------------------
# Test 11 - sequence length one (single token)
# ---------------------------------------------------------------------------

def test_seq_len_one():
    model = make_model()
    x = torch.randn(2, 1, TINY_CFG.d_model)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, TINY_CFG.d_model)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 12 - determinism in eval mode
# ---------------------------------------------------------------------------

def test_determinism():
    model = make_model()
    torch.manual_seed(7)
    x = torch.randn(2, 8, TINY_CFG.d_model)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2), "Eval mode outputs differ on identical inputs"


# ---------------------------------------------------------------------------
# Test 13 - compression is real (kv_lrank < n_heads * head_dim)
# ---------------------------------------------------------------------------

def test_compression_smaller_than_full():
    model = make_model()
    full_kv_size = TINY_CFG.n_heads * TINY_CFG.head_dim  # 64
    assert TINY_CFG.kv_lrank < full_kv_size, (
        f"kv_lrank={TINY_CFG.kv_lrank} should be < n_heads*head_dim={full_kv_size}"
    )
    assert model.kv_cache_size_ratio() < 1.0


# ---------------------------------------------------------------------------
# Test 14 - out_proj maps n_heads*head_dim to d_model
# ---------------------------------------------------------------------------

def test_out_proj_shape():
    model = make_model()
    expected_in = TINY_CFG.n_heads * TINY_CFG.head_dim  # 64
    expected_out = TINY_CFG.d_model                      # 64
    assert model.out_proj.in_features == expected_in
    assert model.out_proj.out_features == expected_out


# ---------------------------------------------------------------------------
# Test 15 - registry entry
# ---------------------------------------------------------------------------

def test_registry():
    assert "flash_mla" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["flash_mla"] is FlashMLAAttention
