"""Integration tests for VisionProjector registry wiring (Kimi K2.5 §4, arXiv:2602.02276).

Verifies end-to-end construction, forward pass, backward pass, and registry entry
using a config dict (matching typical runtime usage pattern).
"""

from __future__ import annotations

import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.vision_projector import VisionProjector, VisionProjectorConfig

# ---------------------------------------------------------------------------
# Shared tiny config used across integration tests
# ---------------------------------------------------------------------------

TINY_VIT_HIDDEN = 32
TINY_LLM_HIDDEN = 64
TINY_POOL_FACTOR = 4
TINY_SEQ_LEN = 16


def make_model(temporal_pool: bool = True) -> VisionProjector:
    """Construct a VisionProjector from config dict (simulating runtime usage)."""
    config_dict = dict(
        vit_hidden=TINY_VIT_HIDDEN,
        llm_hidden=TINY_LLM_HIDDEN,
        temporal_pool=temporal_pool,
        pool_factor=TINY_POOL_FACTOR,
    )
    cfg = VisionProjectorConfig(**config_dict)
    return VisionProjector(cfg)


# ---------------------------------------------------------------------------
# 1. Registry key exists
# ---------------------------------------------------------------------------


def test_vision_projector_in_registry():
    """'vision_projector' key must be present in MODEL_COMPONENT_REGISTRY."""
    assert "vision_projector" in MODEL_COMPONENT_REGISTRY, (
        "MODEL_COMPONENT_REGISTRY must contain key 'vision_projector'"
    )


# ---------------------------------------------------------------------------
# 2. Registry entry is the correct class
# ---------------------------------------------------------------------------


def test_registry_entry_is_vision_projector_class():
    """Registry entry must be the VisionProjector class (not an instance)."""
    cls = MODEL_COMPONENT_REGISTRY["vision_projector"]
    assert cls is VisionProjector, (
        f"Registry['vision_projector'] should be VisionProjector, got {cls}"
    )


# ---------------------------------------------------------------------------
# 3. Construct from registry and run a forward pass (with pooling)
# ---------------------------------------------------------------------------


def test_construct_from_registry_forward_pass_with_pool():
    """Construct via registry, run forward, assert output shape with temporal pooling."""
    cls = MODEL_COMPONENT_REGISTRY["vision_projector"]
    cfg = VisionProjectorConfig(
        vit_hidden=TINY_VIT_HIDDEN,
        llm_hidden=TINY_LLM_HIDDEN,
        temporal_pool=True,
        pool_factor=TINY_POOL_FACTOR,
    )
    model = cls(cfg)
    x = torch.randn(1, TINY_SEQ_LEN, TINY_VIT_HIDDEN)
    out = model(x)
    expected_N_out = TINY_SEQ_LEN // TINY_POOL_FACTOR  # 16 // 4 = 4
    assert out.shape == (1, expected_N_out, TINY_LLM_HIDDEN), (
        f"Expected (1, {expected_N_out}, {TINY_LLM_HIDDEN}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 4. Forward pass without pooling
# ---------------------------------------------------------------------------


def test_forward_pass_no_pool():
    """Forward pass with temporal_pool=False preserves sequence length."""
    cls = MODEL_COMPONENT_REGISTRY["vision_projector"]
    cfg = VisionProjectorConfig(
        vit_hidden=TINY_VIT_HIDDEN,
        llm_hidden=TINY_LLM_HIDDEN,
        temporal_pool=False,
        pool_factor=TINY_POOL_FACTOR,
    )
    model = cls(cfg)
    x = torch.randn(1, TINY_SEQ_LEN, TINY_VIT_HIDDEN)
    out = model(x)
    assert out.shape == (1, TINY_SEQ_LEN, TINY_LLM_HIDDEN), (
        f"Expected (1, {TINY_SEQ_LEN}, {TINY_LLM_HIDDEN}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 5. Backward pass works end-to-end
# ---------------------------------------------------------------------------


def test_backward_pass():
    """Backward pass through registry-constructed model produces valid gradients."""
    model = make_model(temporal_pool=True)
    x = torch.randn(2, TINY_SEQ_LEN, TINY_VIT_HIDDEN)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.proj.weight.grad is not None, (
        "proj.weight.grad must not be None after backward pass"
    )
    assert not torch.isnan(model.proj.weight.grad).any(), "proj.weight.grad must not contain NaN"


# ---------------------------------------------------------------------------
# 6. Pre-existing registry keys are unaffected
# ---------------------------------------------------------------------------


def test_existing_registry_keys_unaffected():
    """Adding vision_projector must not remove any pre-existing registry keys."""
    for key in ("dsa_attention", "mtp_shared", "dp_aware_moe_routing", "mla_256"):
        assert key in MODEL_COMPONENT_REGISTRY, (
            f"Pre-existing registry key '{key}' must not be removed"
        )
