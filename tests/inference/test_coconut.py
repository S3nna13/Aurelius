"""Unit tests for CoCoNut — Chain of Continuous Thought (Hao et al. 2024).

Tiny test config: d_model=64, n_heads=4, vocab_size=256, seq_len=16.
"""

from __future__ import annotations

import torch

from src.inference import DECODER_REGISTRY
from src.inference.coconut import (
    CoCoNut,
    CoCoNutConfig,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

D_MODEL = 64
BATCH = 2
SEQ_LEN = 16


def make_config(**kwargs) -> CoCoNutConfig:
    defaults = dict(d_model=D_MODEL, n_continuous_steps=4, dropout=0.0)
    defaults.update(kwargs)
    return CoCoNutConfig(**defaults)


def make_model(**kwargs) -> CoCoNut:
    return CoCoNut(make_config(**kwargs))


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = CoCoNutConfig()
    assert cfg.n_continuous_steps == 8
    assert cfg.dropout == 0.0
    assert cfg.d_model == 2048
    assert cfg.continuous_step_hidden is None
    assert cfg.use_layer_norm is True


# ---------------------------------------------------------------------------
# 2. test_reason_shape_2d
# ---------------------------------------------------------------------------


def test_reason_shape_2d():
    model = make_model()
    h = torch.randn(BATCH, D_MODEL)
    out = model.reason(h)
    assert out.shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 3. test_reason_shape_3d
# ---------------------------------------------------------------------------


def test_reason_shape_3d():
    model = make_model()
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = model.reason(h)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# 4. test_reason_changes_hidden
# ---------------------------------------------------------------------------


def test_reason_changes_hidden():
    """Model must actually transform the input (not be identity)."""
    model = make_model()
    model.eval()
    with torch.no_grad():
        h = torch.randn(BATCH, D_MODEL)
        out = model.reason(h)
    # With randomly initialised weights + LayerNorm the output must differ
    assert not torch.allclose(out, h), "reason() returned the input unchanged"


# ---------------------------------------------------------------------------
# 5. test_n_steps_1
# ---------------------------------------------------------------------------


def test_n_steps_1():
    model = make_model(n_continuous_steps=1)
    h = torch.randn(BATCH, D_MODEL)
    out = model.reason(h)
    assert out.shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 6. test_n_steps_0
# ---------------------------------------------------------------------------


def test_n_steps_0():
    """Zero steps: reason() must return the input tensor unchanged."""
    model = make_model(n_continuous_steps=0)
    h = torch.randn(BATCH, D_MODEL)
    out = model.reason(h)
    assert out.shape == h.shape
    assert torch.allclose(out, h), "n_continuous_steps=0 must be identity"


# ---------------------------------------------------------------------------
# 7. test_reason_with_trace_length
# ---------------------------------------------------------------------------


def test_reason_with_trace_length():
    n = 6
    model = make_model(n_continuous_steps=n)
    h = torch.randn(BATCH, D_MODEL)
    _, trace = model.reason_with_trace(h)
    assert len(trace) == n


# ---------------------------------------------------------------------------
# 8. test_reason_with_trace_shapes
# ---------------------------------------------------------------------------


def test_reason_with_trace_shapes():
    model = make_model(n_continuous_steps=4)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    _, trace = model.reason_with_trace(h)
    for t in trace:
        assert t.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# 9. test_reason_with_trace_final
# ---------------------------------------------------------------------------


def test_reason_with_trace_final():
    """The last trace entry must equal the returned final hidden state."""
    model = make_model(n_continuous_steps=4)
    model.eval()
    h = torch.randn(BATCH, D_MODEL)
    with torch.no_grad():
        final, trace = model.reason_with_trace(h)
    assert torch.allclose(trace[-1], final)


# ---------------------------------------------------------------------------
# 10. test_gradients_flow
# ---------------------------------------------------------------------------


def test_gradients_flow():
    model = make_model()
    h = torch.randn(BATCH, D_MODEL, requires_grad=True)
    out = model.reason(h)
    loss = out.sum()
    loss.backward()
    # At least one parameter in the step projections must have received a grad
    grads = [p.grad for step in model.steps for p in step.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed to step parameters"


# ---------------------------------------------------------------------------
# 11. test_forward_alias
# ---------------------------------------------------------------------------


def test_forward_alias():
    """forward() must return the same result as reason() for identical input."""
    model = make_model()
    model.eval()
    h = torch.randn(BATCH, D_MODEL)
    with torch.no_grad():
        out_reason = model.reason(h.clone())
        out_forward = model.forward(h.clone())
    assert torch.allclose(out_reason, out_forward)


# ---------------------------------------------------------------------------
# 12. test_determinism
# ---------------------------------------------------------------------------


def test_determinism():
    """In eval mode with dropout=0, identical inputs must produce identical outputs."""
    model = make_model(dropout=0.0)
    model.eval()
    h = torch.randn(BATCH, D_MODEL)
    with torch.no_grad():
        out1 = model.reason(h)
        out2 = model.reason(h)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# 13. test_batch_independence
# ---------------------------------------------------------------------------


def test_batch_independence():
    """Output for one sample must match a single-sample batch."""
    model = make_model()
    model.eval()
    h_single = torch.randn(1, D_MODEL)
    h_batch = torch.cat([h_single, torch.randn(1, D_MODEL)], dim=0)
    with torch.no_grad():
        out_single = model.reason(h_single)
        out_batch = model.reason(h_batch)
    assert torch.allclose(out_single[0], out_batch[0], atol=1e-5)


# ---------------------------------------------------------------------------
# 14. test_hidden_size_override
# ---------------------------------------------------------------------------


def test_hidden_size_override():
    """A different continuous_step_hidden must still produce d_model output."""
    model = make_model(continuous_step_hidden=128)
    h = torch.randn(BATCH, D_MODEL)
    out = model.reason(h)
    assert out.shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------


def test_registry():
    assert "coconut" in DECODER_REGISTRY
    assert DECODER_REGISTRY["coconut"] is CoCoNut
