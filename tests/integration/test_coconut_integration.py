"""Integration test for CoCoNut — Chain of Continuous Thought.

Builds CoCoNut from a config dict (d_model=64, n_continuous_steps=4),
passes [2, 8, 64] hidden states through it, verifies output shape,
trace length, backward pass, and registry wiring.
"""
from __future__ import annotations

import pytest
import torch

from src.inference.coconut import CoCoNut, CoCoNutConfig
from src.inference import DECODER_REGISTRY


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_coconut_integration():
    # -----------------------------------------------------------------------
    # 1. Build model from config dict (tiny dims matching the project spec)
    # -----------------------------------------------------------------------
    config = CoCoNutConfig(
        d_model=64,
        n_continuous_steps=4,
        continuous_step_hidden=None,   # defaults to d_model
        dropout=0.0,
        use_layer_norm=True,
    )
    model = CoCoNut(config)
    model.eval()

    # -----------------------------------------------------------------------
    # 2. Forward pass — 3-D input [B=2, T=8, d_model=64]
    # -----------------------------------------------------------------------
    B, T, D = 2, 8, 64
    h = torch.randn(B, T, D)
    with torch.no_grad():
        out = model(h)

    # Output shape must match input shape
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"

    # -----------------------------------------------------------------------
    # 3. Trace — verify length and shapes
    # -----------------------------------------------------------------------
    with torch.no_grad():
        final_h, trace = model.reason_with_trace(h)

    assert len(trace) == config.n_continuous_steps, (
        f"Expected trace length {config.n_continuous_steps}, got {len(trace)}"
    )
    for i, t in enumerate(trace):
        assert t.shape == (B, T, D), (
            f"Trace[{i}] has shape {t.shape}, expected ({B}, {T}, {D})"
        )

    # Last trace tensor must equal final_h
    assert torch.allclose(trace[-1], final_h), (
        "Last trace entry does not match the returned final hidden state"
    )

    # -----------------------------------------------------------------------
    # 4. Backward pass — gradients must flow
    # -----------------------------------------------------------------------
    model.train()
    h_grad = torch.randn(B, T, D, requires_grad=True)
    out_train = model(h_grad)
    loss = out_train.sum()
    loss.backward()

    assert h_grad.grad is not None, "Input tensor did not receive a gradient"
    param_grads = [
        p.grad for step in model.steps for p in step.parameters()
        if p.grad is not None
    ]
    assert len(param_grads) > 0, "No step parameter received a gradient"

    # -----------------------------------------------------------------------
    # 5. Registry wiring
    # -----------------------------------------------------------------------
    assert "coconut" in DECODER_REGISTRY, "'coconut' missing from DECODER_REGISTRY"
    assert DECODER_REGISTRY["coconut"] is CoCoNut, (
        "DECODER_REGISTRY['coconut'] does not point to CoCoNut class"
    )
