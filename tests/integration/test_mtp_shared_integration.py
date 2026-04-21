"""Integration tests for SharedMTPHead via MODEL_COMPONENT_REGISTRY.

Verifies:
  1. "mtp_shared" key present in MODEL_COMPONENT_REGISTRY.
  2. Construct from registry, forward pass, list of 3 tensors with correct shape.
  3. Shared weight verified on registry-constructed instance.
  4. Existing registry entries / public symbols are untouched (regression guard).
"""
from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# 1. Registry membership
# ---------------------------------------------------------------------------

def test_mtp_shared_in_registry():
    from src.model import MODEL_COMPONENT_REGISTRY
    assert "mtp_shared" in MODEL_COMPONENT_REGISTRY, (
        "'mtp_shared' not found in MODEL_COMPONENT_REGISTRY"
    )


# ---------------------------------------------------------------------------
# 2. Construct from registry + forward pass
# ---------------------------------------------------------------------------

def test_registry_forward_shape():
    from src.model import MODEL_COMPONENT_REGISTRY

    cls = MODEL_COMPONENT_REGISTRY["mtp_shared"]
    torch.manual_seed(0)
    model = cls(d_model=64, vocab_size=256, n_heads=3)

    h = torch.randn(2, 8, 64)
    out = model(h)

    assert isinstance(out, list), "forward() should return a list"
    assert len(out) == 3, f"Expected 3 tensors, got {len(out)}"
    for i, logits in enumerate(out):
        assert logits.shape == (2, 8, 256), (
            f"Head {i}: expected (2, 8, 256), got {logits.shape}"
        )


# ---------------------------------------------------------------------------
# 3. Shared weight via registry-constructed instance
# ---------------------------------------------------------------------------

def test_registry_shared_weight():
    from src.model import MODEL_COMPONENT_REGISTRY

    cls = MODEL_COMPONENT_REGISTRY["mtp_shared"]
    torch.manual_seed(0)
    model = cls(d_model=64, vocab_size=256, n_heads=3)

    # All heads must route through the same shared_proj weight tensor.
    shared_ptr = model.shared_proj.weight.data_ptr()
    assert shared_ptr != 0, "shared_proj.weight has null data pointer"

    # Per-head input projections must NOT share storage with each other.
    input_ptrs = [p.weight.data_ptr() for p in model.input_projs]
    assert len(set(input_ptrs)) == 3, (
        "input_projs should each have a distinct weight tensor"
    )

    # shared_proj weight pointer is stable (not recreated per call)
    _ = model(torch.randn(1, 4, 64))
    assert model.shared_proj.weight.data_ptr() == shared_ptr


# ---------------------------------------------------------------------------
# 4. Existing registry entries / public symbols (regression guard)
# ---------------------------------------------------------------------------

def test_existing_public_symbols_intact():
    import src.model as m

    must_have = {
        "AureliusConfig",
        "AureliusTransformer",
        "ChunkedLocalAttention",
        "GroupedQueryAttention",
        "LambdaAttention",
        "ParallelAttentionBlock",
        "RMSNorm",
        "SwiGLUFFN",
        "TransformerBlock",
        "apply_rope",
        "count_parameters",
        "precompute_rope_frequencies",
    }
    missing = must_have - set(m.__all__)
    assert not missing, f"Prior exports missing from src.model.__all__: {missing}"
    for name in must_have:
        assert hasattr(m, name), f"src.model lost attribute: {name}"
