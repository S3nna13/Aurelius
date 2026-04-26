"""Unit tests for VisionProjector -- ViT->LLM linear projection with temporal pooling.

Tiny test config: d_model (llm_hidden)=64, vit_hidden=32 throughout.
Run with: .venv/bin/python3.14 -m pytest tests/model/test_vision_projector.py
"""

from __future__ import annotations

import math

import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.vision_projector import VisionProjector, VisionProjectorConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_VIT = 32
TINY_LLM = 64


def tiny_cfg(**kwargs) -> VisionProjectorConfig:
    """Return a tiny VisionProjectorConfig for tests."""
    defaults = dict(vit_hidden=TINY_VIT, llm_hidden=TINY_LLM)
    defaults.update(kwargs)
    return VisionProjectorConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Default VisionProjectorConfig should have canonical production values."""
    cfg = VisionProjectorConfig()
    assert cfg.vit_hidden == 1024
    assert cfg.llm_hidden == 2048
    assert cfg.temporal_pool is True
    assert cfg.pool_factor == 4


# ---------------------------------------------------------------------------
# 2. test_output_shape_no_pool
# ---------------------------------------------------------------------------


def test_output_shape_no_pool():
    """temporal_pool=False: output sequence length equals input sequence length."""
    cfg = tiny_cfg(temporal_pool=False)
    model = VisionProjector(cfg)
    x = torch.randn(2, 16, TINY_VIT)
    out = model(x)
    assert out.shape == (2, 16, TINY_LLM), f"Expected (2, 16, {TINY_LLM}), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_output_shape_with_pool
# ---------------------------------------------------------------------------


def test_output_shape_with_pool():
    """temporal_pool=True, pool_factor=4, N=16 -> N_out=4."""
    cfg = tiny_cfg(temporal_pool=True, pool_factor=4)
    model = VisionProjector(cfg)
    x = torch.randn(2, 16, TINY_VIT)
    out = model(x)
    assert out.shape == (2, 4, TINY_LLM), f"Expected (2, 4, {TINY_LLM}), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. test_output_shape_odd_N
# ---------------------------------------------------------------------------


def test_output_shape_odd_N():
    """N=17, pool_factor=4 -> ceil(17/4)=5 patches out."""
    N = 17
    pool_factor = 4
    expected_N_out = math.ceil(N / pool_factor)  # 5
    cfg = tiny_cfg(temporal_pool=True, pool_factor=pool_factor)
    model = VisionProjector(cfg)
    x = torch.randn(1, N, TINY_VIT)
    out = model(x)
    assert out.shape[1] == expected_N_out, f"Expected N_out={expected_N_out}, got {out.shape[1]}"
    assert out.shape == (1, expected_N_out, TINY_LLM)


# ---------------------------------------------------------------------------
# 5. test_projection_dim
# ---------------------------------------------------------------------------


def test_projection_dim():
    """Output last dimension always equals llm_hidden regardless of pooling setting."""
    for temporal_pool in (True, False):
        cfg = tiny_cfg(temporal_pool=temporal_pool, pool_factor=4)
        model = VisionProjector(cfg)
        x = torch.randn(1, 8, TINY_VIT)
        out = model(x)
        assert out.shape[-1] == TINY_LLM, (
            f"temporal_pool={temporal_pool}: last dim should be {TINY_LLM}, got {out.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# 6. test_no_pool_preserves_N
# ---------------------------------------------------------------------------


def test_no_pool_preserves_N():
    """When temporal_pool=False, N_out == N_in for any sequence length."""
    cfg = tiny_cfg(temporal_pool=False)
    model = VisionProjector(cfg)
    for N in (1, 7, 16, 100):
        x = torch.randn(1, N, TINY_VIT)
        out = model(x)
        assert out.shape[1] == N, f"N={N}: expected N_out={N}, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# 7. test_gradients_flow
# ---------------------------------------------------------------------------


def test_gradients_flow():
    """Backward pass should propagate gradients to proj.weight."""
    cfg = tiny_cfg(temporal_pool=True, pool_factor=4)
    model = VisionProjector(cfg)
    x = torch.randn(2, 8, TINY_VIT, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.proj.weight.grad is not None, "proj.weight.grad must not be None after backward"
    assert model.proj.weight.grad.shape == model.proj.weight.shape


# ---------------------------------------------------------------------------
# 8. test_pool_factor_1
# ---------------------------------------------------------------------------


def test_pool_factor_1():
    """pool_factor=1 is identity pooling: N_out == N_in."""
    cfg = tiny_cfg(temporal_pool=True, pool_factor=1)
    model = VisionProjector(cfg)
    x = torch.randn(2, 12, TINY_VIT)
    out = model(x)
    assert out.shape[1] == 12, f"pool_factor=1 should preserve N, got {out.shape[1]}"
    assert out.shape == (2, 12, TINY_LLM)


# ---------------------------------------------------------------------------
# 9. test_pool_factor_equals_N
# ---------------------------------------------------------------------------


def test_pool_factor_equals_N():
    """pool_factor == N pools the entire sequence down to 1 token."""
    N = 8
    cfg = tiny_cfg(temporal_pool=True, pool_factor=N)
    model = VisionProjector(cfg)
    x = torch.randn(2, N, TINY_VIT)
    out = model(x)
    assert out.shape[1] == 1, f"pool_factor==N should yield 1 token, got {out.shape[1]}"
    assert out.shape == (2, 1, TINY_LLM)


# ---------------------------------------------------------------------------
# 10. test_batch_size_independence
# ---------------------------------------------------------------------------


def test_batch_size_independence():
    """B=1 and B=3 produce consistent output shapes."""
    cfg = tiny_cfg(temporal_pool=True, pool_factor=4)
    model = VisionProjector(cfg)
    N = 16
    for B in (1, 3):
        x = torch.randn(B, N, TINY_VIT)
        out = model(x)
        expected = (B, N // 4, TINY_LLM)
        assert out.shape == expected, f"B={B}: expected {expected}, got {out.shape}"


# ---------------------------------------------------------------------------
# 11. test_weight_init
# ---------------------------------------------------------------------------


def test_weight_init():
    """proj.weight should be a float tensor with shape [llm_hidden, vit_hidden]."""
    cfg = tiny_cfg()
    model = VisionProjector(cfg)
    w = model.proj.weight
    assert w.dtype == torch.float32, f"Expected float32, got {w.dtype}"
    assert w.shape == (TINY_LLM, TINY_VIT), (
        f"Expected weight shape ({TINY_LLM}, {TINY_VIT}), got {w.shape}"
    )


# ---------------------------------------------------------------------------
# 12. test_determinism
# ---------------------------------------------------------------------------


def test_determinism():
    """Same input tensor produces identical output on repeated forward passes."""
    cfg = tiny_cfg(temporal_pool=True, pool_factor=4)
    model = VisionProjector(cfg)
    model.eval()
    x = torch.randn(2, 8, TINY_VIT)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2), "Two forward passes with same input must be identical"


# ---------------------------------------------------------------------------
# 13. test_zero_input
# ---------------------------------------------------------------------------


def test_zero_input():
    """Zero tensor input -> zero output (no bias in proj)."""
    cfg = tiny_cfg(temporal_pool=True, pool_factor=4)
    model = VisionProjector(cfg)
    x = torch.zeros(2, 8, TINY_VIT)
    with torch.no_grad():
        out = model(x)
    assert torch.all(out == 0), "Zero input with bias=False must produce zero output"


# ---------------------------------------------------------------------------
# 14. test_registry_entry
# ---------------------------------------------------------------------------


def test_registry_entry():
    """MODEL_COMPONENT_REGISTRY['vision_projector'] must be VisionProjector."""
    assert "vision_projector" in MODEL_COMPONENT_REGISTRY, (
        "MODEL_COMPONENT_REGISTRY must contain key 'vision_projector'"
    )
    assert MODEL_COMPONENT_REGISTRY["vision_projector"] is VisionProjector, (
        "Registry entry must be the VisionProjector class itself"
    )


# ---------------------------------------------------------------------------
# 15. test_tiny_config
# ---------------------------------------------------------------------------


def test_tiny_config():
    """vit_hidden=8, llm_hidden=16, pool_factor=2, input [1, 4, 8] -> [1, 2, 16]."""
    cfg = VisionProjectorConfig(vit_hidden=8, llm_hidden=16, temporal_pool=True, pool_factor=2)
    model = VisionProjector(cfg)
    x = torch.randn(1, 4, 8)
    out = model(x)
    assert out.shape == (1, 2, 16), f"Expected (1, 2, 16), got {out.shape}"
