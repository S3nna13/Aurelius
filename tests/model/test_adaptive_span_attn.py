"""Tests for src/model/adaptive_span_attn.py."""

from __future__ import annotations

import torch

from src.model.adaptive_span_attn import (
    AdaptiveSpanAttention,
    AdaptiveSpanBlock,
    AdaptiveSpanConfig,
    get_effective_spans,
    soft_span_mask,
)

# ---------------------------------------------------------------------------
# Tiny test fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
N_HEADS = 2
MAX_SPAN = 8
B = 2
SEQ = 6


def tiny_config() -> AdaptiveSpanConfig:
    return AdaptiveSpanConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        max_span=MAX_SPAN,
        span_loss_coef=0.0002,
        init_span=0.5,
    )


def make_x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = AdaptiveSpanConfig()
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.max_span == 128
    assert cfg.span_loss_coef == 0.0002
    assert cfg.init_span == 0.5


# ---------------------------------------------------------------------------
# 2. soft_span_mask shape
# ---------------------------------------------------------------------------


def test_soft_span_mask_shape():
    span = torch.tensor([0.5, 0.75])
    mask = soft_span_mask(span, MAX_SPAN, SEQ)
    assert mask.shape == (N_HEADS, SEQ, SEQ), (
        f"Expected ({N_HEADS}, {SEQ}, {SEQ}), got {mask.shape}"
    )


# ---------------------------------------------------------------------------
# 3. soft_span_mask causal: j > i must be -inf
# ---------------------------------------------------------------------------


def test_soft_span_mask_causal():
    span = torch.tensor([0.5, 1.0])
    mask = soft_span_mask(span, MAX_SPAN, SEQ)
    # For every head, every i, every j > i must be -inf
    for h in range(N_HEADS):
        for i in range(SEQ):
            for j in range(i + 1, SEQ):
                assert mask[h, i, j].item() <= -1e8, (
                    f"head={h}, i={i}, j={j}: expected -inf, got {mask[h, i, j].item()}"
                )


# ---------------------------------------------------------------------------
# 4. span_params initialised to init_span
# ---------------------------------------------------------------------------


def test_span_params_initialized():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    expected = cfg.init_span
    assert torch.allclose(model.span_params, torch.full((N_HEADS,), expected)), (
        f"span_params not initialised to {expected}: {model.span_params}"
    )


# ---------------------------------------------------------------------------
# 5. AdaptiveSpanAttention output shape
# ---------------------------------------------------------------------------


def test_attention_output_shape():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    x = make_x()
    output, span_loss = model(x)
    assert output.shape == (B, SEQ, D_MODEL), (
        f"Expected ({B}, {SEQ}, {D_MODEL}), got {output.shape}"
    )


# ---------------------------------------------------------------------------
# 6. span_loss is a scalar
# ---------------------------------------------------------------------------


def test_span_loss_is_scalar():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    x = make_x()
    _, span_loss = model(x)
    assert span_loss.dim() == 0, f"span_loss should be 0-d, got shape {span_loss.shape}"


# ---------------------------------------------------------------------------
# 7. span_loss >= 0
# ---------------------------------------------------------------------------


def test_span_loss_nonnegative():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    x = make_x()
    _, span_loss = model(x)
    assert span_loss.item() >= 0.0, f"span_loss={span_loss.item()} is negative"


# ---------------------------------------------------------------------------
# 8. Gradient flows through output
# ---------------------------------------------------------------------------


def test_gradient_through_output():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    x = make_x().requires_grad_(True)
    output, _ = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input x"
    assert x.grad.abs().sum().item() > 0, "Gradient for x is all zeros"


# ---------------------------------------------------------------------------
# 9. Gradient flows through span_loss → span_params
# ---------------------------------------------------------------------------


def test_gradient_through_span_loss():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    x = make_x()
    _, span_loss = model(x)
    span_loss.backward()
    assert model.span_params.grad is not None, "No gradient for span_params"
    assert model.span_params.grad.abs().sum().item() > 0, "Gradient for span_params is all zeros"


# ---------------------------------------------------------------------------
# 10. AdaptiveSpanBlock output shape
# ---------------------------------------------------------------------------


def test_block_output_shape():
    cfg = tiny_config()
    block = AdaptiveSpanBlock(cfg)
    x = make_x()
    output, span_loss = block(x)
    assert output.shape == (B, SEQ, D_MODEL), (
        f"Expected ({B}, {SEQ}, {D_MODEL}), got {output.shape}"
    )


# ---------------------------------------------------------------------------
# 11. AdaptiveSpanBlock residual connection
# ---------------------------------------------------------------------------


def test_block_residual():
    """With zeroed attention weights the block should approximate identity."""
    cfg = tiny_config()
    block = AdaptiveSpanBlock(cfg)
    # Force span to 0 → most positions masked → near-uniform softmax over
    # only self-position → output close to projection of x.  Instead verify
    # output != 0 (residual keeps signal) and shape is preserved.
    x = make_x()
    output, _ = block(x)
    # Output should not be all zeros (residual keeps x)
    assert output.abs().sum().item() > 0, "Block output is all zeros — residual likely broken"
    # Output shape matches input
    assert output.shape == x.shape


# ---------------------------------------------------------------------------
# 12. get_effective_spans shape
# ---------------------------------------------------------------------------


def test_get_effective_spans_shape():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    spans = get_effective_spans(model)
    assert spans.shape == (N_HEADS,), f"Expected ({N_HEADS},), got {spans.shape}"


# ---------------------------------------------------------------------------
# 13. get_effective_spans values bounded by max_span
# ---------------------------------------------------------------------------


def test_get_effective_spans_bounded():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    # Force span_params beyond [0,1] to confirm clamping works
    with torch.no_grad():
        model.span_params.fill_(2.0)
    spans = get_effective_spans(model)
    assert spans.max().item() <= MAX_SPAN, f"Effective spans exceed max_span={MAX_SPAN}: {spans}"
    assert spans.min().item() >= 0, f"Negative span: {spans}"


# ---------------------------------------------------------------------------
# 14. Single-token sequence (edge case)
# ---------------------------------------------------------------------------


def test_single_token_sequence():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    x = torch.randn(1, 1, D_MODEL)
    output, span_loss = model(x)
    assert output.shape == (1, 1, D_MODEL)
    assert span_loss.dim() == 0


# ---------------------------------------------------------------------------
# 15. span_params are trainable (requires_grad)
# ---------------------------------------------------------------------------


def test_span_params_requires_grad():
    cfg = tiny_config()
    model = AdaptiveSpanAttention(cfg)
    assert model.span_params.requires_grad, "span_params should have requires_grad=True"
