"""Integration tests for MLA-256 registry wiring (GLM-5 §3.1, arXiv:2602.15763)."""

from __future__ import annotations

import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.mla_256 import MLA256Attention, MLA256Config

# ---------------------------------------------------------------------------
# Shared tiny config used across integration tests
# ---------------------------------------------------------------------------
TINY = MLA256Config(d_model=64, n_heads=4, head_dim=16, kv_lrank=8)


# ---------------------------------------------------------------------------
# 1. Registry key exists
# ---------------------------------------------------------------------------
def test_mla_256_in_registry():
    assert "mla_256" in MODEL_COMPONENT_REGISTRY, (
        "MODEL_COMPONENT_REGISTRY must contain key 'mla_256'"
    )


# ---------------------------------------------------------------------------
# 2. Registry entry is the correct class
# ---------------------------------------------------------------------------
def test_registry_entry_is_mla256_attention():
    cls = MODEL_COMPONENT_REGISTRY["mla_256"]
    assert cls is MLA256Attention, f"Registry['mla_256'] should be MLA256Attention, got {cls}"


# ---------------------------------------------------------------------------
# 3. Construct from registry and run a forward pass
# ---------------------------------------------------------------------------
def test_construct_from_registry_forward_pass():
    cls = MODEL_COMPONENT_REGISTRY["mla_256"]
    model = cls(TINY)
    x = torch.randn(2, 8, TINY.d_model)
    out = model(x)
    assert out.shape == (2, 8, TINY.d_model), (
        f"Forward pass shape mismatch: expected (2, 8, {TINY.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 4. KV latent dimension is strictly smaller than d_model (compression verified)
# ---------------------------------------------------------------------------
def test_kv_lrank_smaller_than_d_model():
    assert TINY.kv_lrank < TINY.d_model, (
        f"kv_lrank ({TINY.kv_lrank}) must be < d_model ({TINY.d_model}) "
        "to confirm low-rank KV compression"
    )
    cls = MODEL_COMPONENT_REGISTRY["mla_256"]
    model = cls(TINY)
    # Confirm the actual Linear layer dimensions
    assert model.kv_down.in_features == TINY.d_model
    assert model.kv_down.out_features == TINY.kv_lrank
    assert model.kv_down.out_features < model.kv_down.in_features


# ---------------------------------------------------------------------------
# 5. Regression guard: pre-existing registry keys are unaffected
# ---------------------------------------------------------------------------
def test_existing_registry_keys_unaffected():
    assert "dsa_attention" in MODEL_COMPONENT_REGISTRY, (
        "Pre-existing registry key 'dsa_attention' must not be removed"
    )
    assert "mtp_shared" in MODEL_COMPONENT_REGISTRY, (
        "Pre-existing registry key 'mtp_shared' must not be removed"
    )
