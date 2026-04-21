"""Unit tests for GQAAbsorbedAttention.

Tiny config: d_model=64, n_heads=4, n_kv_heads=2, head_dim=16.
"""

from __future__ import annotations

import pytest
import torch

from src.model.gqa_absorbed import GQAAbsorbedAttention, GQAAbsorbedConfig
from src.model import MODEL_COMPONENT_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16


@pytest.fixture
def tiny_cfg() -> GQAAbsorbedConfig:
    return GQAAbsorbedConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dropout=0.0,
        use_absorbed=False,
    )


@pytest.fixture
def standard_model(tiny_cfg: GQAAbsorbedConfig) -> GQAAbsorbedAttention:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.eval()
    return model


@pytest.fixture
def absorbed_model(tiny_cfg: GQAAbsorbedConfig) -> GQAAbsorbedAttention:
    cfg = GQAAbsorbedConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dropout=0.0,
        use_absorbed=True,
    )
    model = GQAAbsorbedAttention(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults() -> None:
    cfg = GQAAbsorbedConfig()
    assert cfg.n_kv_heads == 8
    assert cfg.use_absorbed is False
    assert cfg.n_heads == 16
    assert cfg.d_model == 2048
    assert cfg.head_dim == 128


# ---------------------------------------------------------------------------
# 2. test_output_shape
# ---------------------------------------------------------------------------

def test_output_shape(standard_model: GQAAbsorbedAttention) -> None:
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out = standard_model(x)
    assert out.shape == (2, 8, D_MODEL)


# ---------------------------------------------------------------------------
# 3. test_standard_gqa_runs
# ---------------------------------------------------------------------------

def test_standard_gqa_runs(standard_model: GQAAbsorbedAttention) -> None:
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out = standard_model._standard_gqa(x)
    assert out.shape == (2, 8, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4. test_absorbed_gqa_runs
# ---------------------------------------------------------------------------

def test_absorbed_gqa_runs(absorbed_model: GQAAbsorbedAttention) -> None:
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out = absorbed_model._absorbed_gqa(x)
    assert out.shape == (2, 8, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5. test_standard_vs_absorbed_agree
# ---------------------------------------------------------------------------

def test_standard_vs_absorbed_agree(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.eval()

    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out_std = model._standard_gqa(x)
        out_abs = model._absorbed_gqa(x)

    assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5), (
        f"Max diff: {(out_std - out_abs).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 6. test_kv_heads_ratio
# ---------------------------------------------------------------------------

def test_kv_heads_ratio(standard_model: GQAAbsorbedAttention) -> None:
    assert standard_model.kv_heads_ratio() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 7. test_gradients_standard
# ---------------------------------------------------------------------------

def test_gradients_standard(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.train()
    x = torch.randn(2, 8, D_MODEL, requires_grad=True)
    out = model._standard_gqa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    for p in model.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# 8. test_gradients_absorbed
# ---------------------------------------------------------------------------

def test_gradients_absorbed(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.train()
    x = torch.randn(2, 8, D_MODEL, requires_grad=True)
    out = model._absorbed_gqa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    for p in model.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# 9. test_n_kv_heads_equals_n_heads  (MHA case)
# ---------------------------------------------------------------------------

def test_n_kv_heads_equals_n_heads() -> None:
    cfg = GQAAbsorbedConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        head_dim=HEAD_DIM,
    )
    model = GQAAbsorbedAttention(cfg)
    model.eval()
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out_std = model._standard_gqa(x)
        out_abs = model._absorbed_gqa(x)
    assert out_std.shape == (2, 8, D_MODEL)
    assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 10. test_n_kv_heads_1  (MQA case)
# ---------------------------------------------------------------------------

def test_n_kv_heads_1() -> None:
    cfg = GQAAbsorbedConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=1,
        head_dim=HEAD_DIM,
    )
    model = GQAAbsorbedAttention(cfg)
    model.eval()
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out_std = model._standard_gqa(x)
        out_abs = model._absorbed_gqa(x)
    assert out_std.shape == (2, 8, D_MODEL)
    assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. test_absorb_called
# ---------------------------------------------------------------------------

def test_absorb_called(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.eval()

    assert model._absorbed_qk is None
    assert model._absorbed_qv is None

    model.absorb()

    G = N_HEADS // N_KV_HEADS
    assert model._absorbed_qk is not None
    assert model._absorbed_qv is not None
    assert model._absorbed_qk.shape == (N_KV_HEADS, G, HEAD_DIM, HEAD_DIM)
    assert model._absorbed_qv.shape == (N_KV_HEADS, G, HEAD_DIM, HEAD_DIM)


# ---------------------------------------------------------------------------
# 12. test_batch_size_one
# ---------------------------------------------------------------------------

def test_batch_size_one(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.eval()
    x = torch.randn(1, 8, D_MODEL)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 8, D_MODEL)


# ---------------------------------------------------------------------------
# 13. test_seq_len_one
# ---------------------------------------------------------------------------

def test_seq_len_one(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.eval()
    x = torch.randn(2, 1, D_MODEL)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, D_MODEL)


# ---------------------------------------------------------------------------
# 14. test_determinism
# ---------------------------------------------------------------------------

def test_determinism(tiny_cfg: GQAAbsorbedConfig) -> None:
    model = GQAAbsorbedAttention(tiny_cfg)
    model.eval()
    x = torch.randn(2, 8, D_MODEL)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------

def test_registry() -> None:
    assert "gqa_absorbed" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["gqa_absorbed"] is GQAAbsorbedAttention
