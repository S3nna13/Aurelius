"""
Tests for src/model/sliding_window_attention_v2.py — 10 tests.
Pure PyTorch, tiny tensors, no external deps.
"""

import torch

from src.model.sliding_window_attention_v2 import SlidingWindowAttention, SWAConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
B, T, D = 2, 16, 32
H = 4
HD = D // H  # head_dim = 8


def make_cfg(**kwargs) -> SWAConfig:
    defaults = dict(d_model=D, n_heads=H, head_dim=HD, window_size=4)
    defaults.update(kwargs)
    return SWAConfig(**defaults)


def make_model(**kwargs) -> SlidingWindowAttention:
    return SlidingWindowAttention(make_cfg(**kwargs))


# ---------------------------------------------------------------------------
# 1. SWAConfig instantiates
# ---------------------------------------------------------------------------


def test_swa_config_instantiates():
    cfg = SWAConfig(d_model=64, n_heads=4, head_dim=16)
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.head_dim == 16
    assert cfg.window_size == 512
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# 2. SlidingWindowAttention instantiates
# ---------------------------------------------------------------------------


def test_sliding_window_attention_instantiates():
    model = make_model()
    assert isinstance(model, SlidingWindowAttention)
    assert hasattr(model, "q_proj")
    assert hasattr(model, "k_proj")
    assert hasattr(model, "v_proj")
    assert hasattr(model, "o_proj")


# ---------------------------------------------------------------------------
# 3. Forward returns shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_forward_output_shape():
    model = make_model()
    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 4. Output is finite
# ---------------------------------------------------------------------------


def test_forward_output_is_finite():
    model = make_model()
    x = torch.randn(B, T, D)
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 5. _build_window_mask returns BoolTensor of shape (T, T)
# ---------------------------------------------------------------------------


def test_build_window_mask_shape_and_dtype():
    model = make_model(window_size=4)
    mask = model._build_window_mask(T, 4, torch.device("cpu"))
    assert mask.shape == (T, T), f"Expected ({T}, {T}), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected bool dtype, got {mask.dtype}"


# ---------------------------------------------------------------------------
# 6. Diagonal is True (token attends to itself)
# ---------------------------------------------------------------------------


def test_build_window_mask_diagonal_true():
    model = make_model(window_size=4)
    mask = model._build_window_mask(T, 4, torch.device("cpu"))
    diag = torch.diagonal(mask)
    assert diag.all(), "Every diagonal element should be True (self-attention)"


# ---------------------------------------------------------------------------
# 7. Position (5, 0) is False when window_size=2
# ---------------------------------------------------------------------------


def test_build_window_mask_far_position_false():
    model = make_model(window_size=2)
    # T must be > 5; use T=10
    mask = model._build_window_mask(10, 2, torch.device("cpu"))
    assert mask[5, 0].item() is False, (
        "Position (5, 0) should be False with window_size=2 (too far back)"
    )


# ---------------------------------------------------------------------------
# 8. Position (2, 0) is True when window_size=4
# ---------------------------------------------------------------------------


def test_build_window_mask_within_window_true():
    model = make_model(window_size=4)
    mask = model._build_window_mask(10, 4, torch.device("cpu"))
    assert mask[2, 0].item() is True, (
        "Position (2, 0) should be True with window_size=4 (within window)"
    )


# ---------------------------------------------------------------------------
# 9. Gradient flows to Q projection weight
# ---------------------------------------------------------------------------


def test_gradient_flows_to_q_proj():
    model = make_model()
    x = torch.randn(B, T, D)
    out = model(x)
    out.sum().backward()
    assert model.q_proj.weight.grad is not None, "q_proj.weight should have gradient"
    assert torch.isfinite(model.q_proj.weight.grad).all(), "q_proj.weight grad must be finite"


# ---------------------------------------------------------------------------
# 10. Works with T=1
# ---------------------------------------------------------------------------


def test_forward_single_token():
    model = make_model()
    x = torch.randn(1, 1, D)
    out = model(x)
    assert out.shape == (1, 1, D), f"Expected (1, 1, {D}), got {out.shape}"
    assert torch.isfinite(out).all(), "T=1 output should be finite"
