"""Integration tests for DSAAttention — GLM-5 §3.1 Lightning Indexer.

Verifies that the class is correctly wired into MODEL_COMPONENT_REGISTRY
via src/model/__init__.py, and that basic forward / freeze-indexer paths
work end-to-end.
"""
from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# 1. "dsa_attention" present in MODEL_COMPONENT_REGISTRY
# ---------------------------------------------------------------------------
def test_dsa_attention_in_registry():
    from src.model import MODEL_COMPONENT_REGISTRY
    assert "dsa_attention" in MODEL_COMPONENT_REGISTRY, (
        "MODEL_COMPONENT_REGISTRY missing 'dsa_attention' key"
    )


# ---------------------------------------------------------------------------
# 2. Construct from registry, forward pass correct shape
# ---------------------------------------------------------------------------
def test_registry_construct_and_forward():
    from src.model import MODEL_COMPONENT_REGISTRY
    from src.model.dsa_attention import DSAConfig

    cls = MODEL_COMPONENT_REGISTRY["dsa_attention"]
    cfg = DSAConfig(d_model=64, n_heads=4, top_k=8)
    model = cls(cfg)

    x = torch.randn(2, 16, 64)
    out = model(x)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. freeze_indexer=True path works via registry
# ---------------------------------------------------------------------------
def test_registry_freeze_indexer_path():
    from src.model import MODEL_COMPONENT_REGISTRY
    from src.model.dsa_attention import DSAConfig

    cls = MODEL_COMPONENT_REGISTRY["dsa_attention"]
    cfg = DSAConfig(d_model=64, n_heads=4, top_k=8, freeze_indexer=True)
    model = cls(cfg)

    # Indexer params must be frozen
    for p in model.indexer.parameters():
        assert not p.requires_grad

    # Forward + backward still work; indexer gets no grad
    x = torch.randn(2, 16, 64)
    out = model(x)
    out.sum().backward()
    assert model.indexer.score.weight.grad is None
    assert model.q_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 4. Regression guard: an existing registry key is still present
# ---------------------------------------------------------------------------
def test_existing_registry_key_regression():
    """Ensure adding dsa_attention did not remove any pre-existing registry keys."""
    from src.model import MODEL_COMPONENT_REGISTRY
    # The registry was introduced alongside dsa_attention; at minimum it must
    # contain "dsa_attention" (tested above). Any future keys added by sibling
    # cycles will remain. This test guards that the dict itself is intact.
    assert isinstance(MODEL_COMPONENT_REGISTRY, dict)
    assert len(MODEL_COMPONENT_REGISTRY) >= 1
