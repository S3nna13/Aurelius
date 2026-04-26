"""Unit tests for VisionCrossAttention.

Tiny config: d_text=64, d_vision=128, n_heads=4, n_kv_heads=2, head_dim=16.
"""

from __future__ import annotations

import pytest
import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.vision_cross_attention import VisionCrossAttention, VisionCrossAttnConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_TEXT = 64
D_VISION = 128
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16

B = 2
T_TEXT = 8
T_VIS = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_cfg() -> VisionCrossAttnConfig:
    return VisionCrossAttnConfig(
        d_text=D_TEXT,
        d_vision=D_VISION,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dropout=0.0,
        use_gated=False,
    )


@pytest.fixture
def tiny_model(tiny_cfg: VisionCrossAttnConfig) -> VisionCrossAttention:
    m = VisionCrossAttention(tiny_cfg)
    m.eval()
    return m


@pytest.fixture
def text_hidden() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T_TEXT, D_TEXT)


@pytest.fixture
def vision_tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, T_VIS, D_VISION)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = VisionCrossAttnConfig()
    assert cfg.d_text == 2048
    assert cfg.d_vision == 1024
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.dropout == 0.0
    assert cfg.use_gated is False
    assert cfg.groups_per_kv == 2


# ---------------------------------------------------------------------------
# 2. test_output_shape
# ---------------------------------------------------------------------------


def test_output_shape(
    tiny_model: VisionCrossAttention,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> None:
    with torch.no_grad():
        out = tiny_model(text_hidden, vision_tokens)
    assert out.shape == (B, T_TEXT, D_TEXT)


# ---------------------------------------------------------------------------
# 3. test_output_dtype
# ---------------------------------------------------------------------------


def test_output_dtype(
    tiny_model: VisionCrossAttention,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> None:
    with torch.no_grad():
        out = tiny_model(text_hidden, vision_tokens)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. test_vision_mask_effect
# ---------------------------------------------------------------------------


def test_vision_mask_effect(
    tiny_model: VisionCrossAttention,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> None:
    """Attention weight on a masked (invalid) vision token should be ~0."""
    mask = torch.ones(B, T_VIS, dtype=torch.bool)
    mask[:, -1] = False  # mask out the last vision position

    with torch.no_grad():
        weights = tiny_model.attention_weights(text_hidden, vision_tokens, vision_mask=mask)

    masked_attn = weights[:, :, :, -1]
    assert masked_attn.abs().max().item() < 1e-6, (
        f"Expected near-zero attention on masked token, got {masked_attn.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 5. test_gqa_kv_repeat
# ---------------------------------------------------------------------------


def test_gqa_kv_repeat() -> None:
    """n_kv_heads=2, n_heads=4 — each KV head serves 2 query heads."""
    cfg = VisionCrossAttnConfig(
        d_text=D_TEXT,
        d_vision=D_VISION,
        n_heads=4,
        n_kv_heads=2,
        head_dim=HEAD_DIM,
    )
    m = VisionCrossAttention(cfg)
    m.eval()

    text = torch.randn(B, T_TEXT, D_TEXT)
    vision = torch.randn(B, T_VIS, D_VISION)

    with torch.no_grad():
        out = m(text, vision)

    assert out.shape == (B, T_TEXT, D_TEXT)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. test_attention_weights_shape
# ---------------------------------------------------------------------------


def test_attention_weights_shape(
    tiny_model: VisionCrossAttention,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> None:
    with torch.no_grad():
        weights = tiny_model.attention_weights(text_hidden, vision_tokens)
    assert weights.shape == (B, N_HEADS, T_TEXT, T_VIS)


# ---------------------------------------------------------------------------
# 7. test_attention_weights_sum_to_one
# ---------------------------------------------------------------------------


def test_attention_weights_sum_to_one(
    tiny_model: VisionCrossAttention,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> None:
    """Each query position's attention distribution must sum to 1 over T_vis."""
    with torch.no_grad():
        weights = tiny_model.attention_weights(text_hidden, vision_tokens)
    row_sums = weights.sum(dim=-1)  # (B, n_heads, T_text)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Attention row sums not close to 1; max deviation: {(row_sums - 1).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 8. test_gate_zero_init
# ---------------------------------------------------------------------------


def test_gate_zero_init() -> None:
    """With use_gated=True and zero-init gate, tanh(0)=0 so output is zero."""
    cfg = VisionCrossAttnConfig(
        d_text=D_TEXT,
        d_vision=D_VISION,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        use_gated=True,
    )
    m = VisionCrossAttention(cfg)
    m.eval()

    assert m.gate is not None
    assert m.gate.item() == pytest.approx(0.0)

    text = torch.randn(B, T_TEXT, D_TEXT)
    vision = torch.randn(B, T_VIS, D_VISION)

    with torch.no_grad():
        out = m(text, vision)

    assert out.abs().max().item() < 1e-6, (
        f"Expected zero output at gate=0, got max={out.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 9. test_gate_effect
# ---------------------------------------------------------------------------


def test_gate_effect() -> None:
    """A non-zero gate value produces a different output than the ungated model."""
    cfg_gated = VisionCrossAttnConfig(
        d_text=D_TEXT,
        d_vision=D_VISION,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        use_gated=True,
    )
    cfg_plain = VisionCrossAttnConfig(
        d_text=D_TEXT,
        d_vision=D_VISION,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        use_gated=False,
    )

    torch.manual_seed(42)
    m_gated = VisionCrossAttention(cfg_gated)
    m_plain = VisionCrossAttention(cfg_plain)

    # Copy shared weights so only the gate differs
    m_gated.load_state_dict(m_plain.state_dict(), strict=False)

    # Set gate to 1.0 so tanh(1) ≈ 0.7616
    with torch.no_grad():
        m_gated.gate.fill_(1.0)

    m_gated.eval()
    m_plain.eval()

    text = torch.randn(B, T_TEXT, D_TEXT)
    vision = torch.randn(B, T_VIS, D_VISION)

    with torch.no_grad():
        out_gated = m_gated(text, vision)
        out_plain = m_plain(text, vision)

    assert not torch.allclose(out_gated, out_plain), (
        "Gated (gate=1) and plain outputs should differ."
    )


# ---------------------------------------------------------------------------
# 10. test_gradient_flows
# ---------------------------------------------------------------------------


def test_gradient_flows(tiny_cfg: VisionCrossAttnConfig) -> None:
    """Gradients must flow back through both text and vision inputs."""
    m = VisionCrossAttention(tiny_cfg)
    m.train()

    text = torch.randn(B, T_TEXT, D_TEXT, requires_grad=True)
    vision = torch.randn(B, T_VIS, D_VISION, requires_grad=True)

    out = m(text, vision)
    out.sum().backward()

    assert text.grad is not None, "No gradient for text_hidden"
    assert text.grad.shape == text.shape
    assert vision.grad is not None, "No gradient for vision_tokens"
    assert vision.grad.shape == vision.shape

    for name, param in m.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"


# ---------------------------------------------------------------------------
# 11. test_determinism
# ---------------------------------------------------------------------------


def test_determinism(
    tiny_cfg: VisionCrossAttnConfig,
    text_hidden: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> None:
    m = VisionCrossAttention(tiny_cfg)
    m.eval()

    with torch.no_grad():
        out1 = m(text_hidden, vision_tokens)
        out2 = m(text_hidden, vision_tokens)

    assert torch.allclose(out1, out2), "Non-deterministic outputs detected."


# ---------------------------------------------------------------------------
# 12. test_no_vision_tokens_empty_mask
# ---------------------------------------------------------------------------


def test_no_vision_tokens_empty_mask(tiny_cfg: VisionCrossAttnConfig) -> None:
    """T_vis=1 with all positions masked → output is near-zero."""
    m = VisionCrossAttention(tiny_cfg)
    m.eval()

    text = torch.randn(B, T_TEXT, D_TEXT)
    vision = torch.randn(B, 1, D_VISION)
    vision_mask = torch.zeros(B, 1, dtype=torch.bool)  # all masked

    with torch.no_grad():
        out = m(text, vision, vision_mask=vision_mask)

    assert out.abs().max().item() < 1e-6, (
        f"Expected near-zero output when all vision tokens masked; got {out.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 13. test_text_seq_independence
# ---------------------------------------------------------------------------


def test_text_seq_independence(
    tiny_model: VisionCrossAttention,
    vision_tokens: torch.Tensor,
) -> None:
    """Perturbing one text position must not affect other text positions."""
    torch.manual_seed(99)
    text_a = torch.randn(1, T_TEXT, D_TEXT)
    text_b = text_a.clone()
    text_b[0, 3, :] += 10.0  # perturb only position 3

    with torch.no_grad():
        out_a = tiny_model(text_a, vision_tokens[:1])
        out_b = tiny_model(text_b, vision_tokens[:1])

    # Position 3 must change
    assert not torch.allclose(out_a[0, 3], out_b[0, 3]), (
        "Output at position 3 should differ after perturbing text at position 3."
    )

    # All other positions must remain unchanged
    for pos in range(T_TEXT):
        if pos == 3:
            continue
        assert torch.allclose(out_a[0, pos], out_b[0, pos], atol=1e-5), (
            f"Output at position {pos} changed unexpectedly."
        )


# ---------------------------------------------------------------------------
# 14. test_vision_d_different_from_text
# ---------------------------------------------------------------------------


def test_vision_d_different_from_text() -> None:
    """d_vision != d_text must work — they use separate projections."""
    cfg = VisionCrossAttnConfig(
        d_text=32,
        d_vision=256,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    m = VisionCrossAttention(cfg)
    m.eval()

    text = torch.randn(B, T_TEXT, 32)
    vision = torch.randn(B, T_VIS, 256)

    with torch.no_grad():
        out = m(text, vision)

    assert out.shape == (B, T_TEXT, 32)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------


def test_registry() -> None:
    assert "vision_cross_attention" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["vision_cross_attention"] is VisionCrossAttention


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_integration_forward_backward() -> None:
    """Integration: B=2, T_text=8, T_vis=16, d_text=64, d_vision=128 — fwd + bwd."""
    cfg = VisionCrossAttnConfig(
        d_text=64,
        d_vision=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        dropout=0.0,
        use_gated=False,
    )
    m = VisionCrossAttention(cfg)
    m.train()

    torch.manual_seed(7)
    text = torch.randn(2, 8, 64, requires_grad=True)
    vision = torch.randn(2, 16, 128, requires_grad=True)

    # Partial mask: first 12 vision tokens valid, last 4 padded
    vision_mask = torch.ones(2, 16, dtype=torch.bool)
    vision_mask[:, 12:] = False

    out = m(text, vision, vision_mask=vision_mask)

    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()

    assert text.grad is not None
    assert vision.grad is not None
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(vision.grad).all()
