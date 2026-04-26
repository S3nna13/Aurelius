"""Integration test: GQAAbsorbedAttention end-to-end.

Builds GQAAbsorbedAttention(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16).
Verifies both paths produce matching outputs, kv_heads_ratio=0.5, backward
works, and registry is wired correctly.
"""

from __future__ import annotations

import pytest
import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.gqa_absorbed import GQAAbsorbedAttention, GQAAbsorbedConfig

# ---------------------------------------------------------------------------
# Shared integration fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integration_model() -> GQAAbsorbedAttention:
    cfg = GQAAbsorbedConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        dropout=0.0,
        use_absorbed=False,
    )
    model = GQAAbsorbedAttention(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_integration_both_paths_match(integration_model: GQAAbsorbedAttention) -> None:
    """Standard and absorbed paths must agree to rtol=1e-4."""
    x = torch.randn(3, 12, 64)
    with torch.no_grad():
        out_std = integration_model._standard_gqa(x)
        out_abs = integration_model._absorbed_gqa(x)
    assert out_std.shape == (3, 12, 64)
    assert out_abs.shape == (3, 12, 64)
    assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5), (
        f"Max diff: {(out_std - out_abs).abs().max().item()}"
    )


def test_integration_kv_heads_ratio(integration_model: GQAAbsorbedAttention) -> None:
    assert integration_model.kv_heads_ratio() == pytest.approx(0.5)


def test_integration_forward_standard_mode(integration_model: GQAAbsorbedAttention) -> None:
    """forward() dispatches to standard path when use_absorbed=False."""
    assert integration_model.cfg.use_absorbed is False
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out = integration_model(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_integration_forward_absorbed_mode() -> None:
    """forward() dispatches to absorbed path when use_absorbed=True."""
    cfg = GQAAbsorbedConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, use_absorbed=True)
    model = GQAAbsorbedAttention(cfg)
    model.eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_integration_backward_standard() -> None:
    """Gradients flow through the standard path."""
    cfg = GQAAbsorbedConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16)
    model = GQAAbsorbedAttention(cfg)
    model.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = model._standard_gqa(x)
    out.sum().backward()
    assert x.grad is not None
    for p in model.parameters():
        assert p.grad is not None


def test_integration_backward_absorbed() -> None:
    """Gradients flow through the absorbed path."""
    cfg = GQAAbsorbedConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16)
    model = GQAAbsorbedAttention(cfg)
    model.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = model._absorbed_gqa(x)
    out.sum().backward()
    assert x.grad is not None
    for p in model.parameters():
        assert p.grad is not None


def test_integration_registry_wired() -> None:
    """MODEL_COMPONENT_REGISTRY['gqa_absorbed'] must point to GQAAbsorbedAttention."""
    assert "gqa_absorbed" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["gqa_absorbed"] is GQAAbsorbedAttention


def test_integration_registry_instantiable() -> None:
    """Registry entry can be used to instantiate the model."""
    cls = MODEL_COMPONENT_REGISTRY["gqa_absorbed"]
    cfg = GQAAbsorbedConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16)
    model = cls(cfg)
    model.eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 4, 64)


def test_integration_absorb_then_forward_match() -> None:
    """After absorb(), both paths still agree."""
    cfg = GQAAbsorbedConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16)
    model = GQAAbsorbedAttention(cfg)
    model.eval()
    model.absorb()

    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out_std = model._standard_gqa(x)
        out_abs = model._absorbed_gqa(x)
    assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5)


def test_integration_output_not_all_zeros(integration_model: GQAAbsorbedAttention) -> None:
    """Output should be non-trivial for random input."""
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        out = integration_model(x)
    assert out.abs().max().item() > 1e-6
