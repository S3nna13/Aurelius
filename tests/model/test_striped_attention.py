"""Unit tests for StripedAttention (Brandon et al. 2023).

Tiny test config: d_model=64, n_heads=4, window_size=4.
"""

from __future__ import annotations

import pytest
import torch

from src.model.striped_attention import StripedAttention, StripedAttentionConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
WINDOW_SIZE = 4
BATCH = 2
SEQ = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> StripedAttentionConfig:
    return StripedAttentionConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        window_size=WINDOW_SIZE,
    )


@pytest.fixture
def model(cfg: StripedAttentionConfig) -> StripedAttention:
    m = StripedAttention(cfg)
    m.train(False)
    return m


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """Default config uses window_size=256 and n_heads=16."""
    cfg = StripedAttentionConfig()
    assert cfg.window_size == 256
    assert cfg.n_heads == 16


# ---------------------------------------------------------------------------
# 2. test_output_shape
# ---------------------------------------------------------------------------


def test_output_shape(model: StripedAttention, x: torch.Tensor) -> None:
    """Forward pass output shape matches [B, T, d_model]."""
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 3. test_is_global_head
# ---------------------------------------------------------------------------


def test_is_global_head(model: StripedAttention) -> None:
    """Even-indexed heads are global; odd-indexed heads are local."""
    for i in range(N_HEADS):
        if i % 2 == 0:
            assert model._is_global_head(i) is True, f"head {i} should be global"
        else:
            assert model._is_global_head(i) is False, f"head {i} should be local"


# ---------------------------------------------------------------------------
# 4. test_head_type_mask
# ---------------------------------------------------------------------------


def test_head_type_mask(model: StripedAttention) -> None:
    """head_type_mask returns alternating ['global', 'local', ...]."""
    mask = model.head_type_mask()
    assert len(mask) == N_HEADS
    for i, label in enumerate(mask):
        expected = "global" if i % 2 == 0 else "local"
        assert label == expected, f"head {i}: expected {expected!r}, got {label!r}"


# ---------------------------------------------------------------------------
# 5. test_full_attention_causal
# ---------------------------------------------------------------------------


def test_full_attention_causal(model: StripedAttention) -> None:
    """Full attention: position i cannot attend to j > i (upper-triangle masked).

    We verify causality by confirming that the output of _full_attention changes
    when we zero-out a future token, and stays the same when we zero-out a past one.
    """
    torch.manual_seed(1)
    T = 8
    q = torch.randn(1, T, HEAD_DIM)
    k = torch.randn(1, T, HEAD_DIM)
    v = torch.randn(1, T, HEAD_DIM)

    out_orig = model._full_attention(q, k, v)  # [1, T, head_dim]

    # Modify a future token's value: position 6, observed from position 3
    # Position 3 should NOT see position 6 — output at position 3 should be unchanged
    v_future_mod = v.clone()
    v_future_mod[0, 6] += 999.0
    out_future_mod = model._full_attention(q, k, v_future_mod)

    # Position 3 output should be the same (it can't see position 6)
    assert torch.allclose(out_orig[0, 3], out_future_mod[0, 3], atol=1e-5), (
        "Causal violation: position 3 should not see position 6"
    )

    # Modify a past token's value: position 1, observed from position 3
    # Position 3 CAN see position 1 — output at position 3 should change
    v_past_mod = v.clone()
    v_past_mod[0, 1] += 999.0
    out_past_mod = model._full_attention(q, k, v_past_mod)
    assert not torch.allclose(out_orig[0, 3], out_past_mod[0, 3], atol=1e-5), (
        "Expected position 3 output to change when a visible past token changes"
    )


# ---------------------------------------------------------------------------
# 6. test_local_attention_window
# ---------------------------------------------------------------------------


def test_local_attention_window(model: StripedAttention) -> None:
    """Local head: position i cannot attend to j < i - window_size + 1."""
    torch.manual_seed(2)
    T = 12
    window = WINDOW_SIZE  # 4
    q = torch.randn(1, T, HEAD_DIM)
    k = torch.randn(1, T, HEAD_DIM)
    v = torch.randn(1, T, HEAD_DIM)

    out_orig = model._local_attention(q, k, v, window)

    # Position 8 should NOT see position 2 (8 - 2 = 6 > window=4)
    # Modifying position 2's value should not change position 8's output
    v_out_of_window = v.clone()
    v_out_of_window[0, 2] += 999.0
    out_mod = model._local_attention(q, k, v_out_of_window, window)
    assert torch.allclose(out_orig[0, 8], out_mod[0, 8], atol=1e-5), (
        "Local attention: position 8 should not see position 2 (outside window)"
    )

    # Position 8 CAN see position 5 (8 - 5 = 3 < window=4)
    v_in_window = v.clone()
    v_in_window[0, 5] += 999.0
    out_in = model._local_attention(q, k, v_in_window, window)
    assert not torch.allclose(out_orig[0, 8], out_in[0, 8], atol=1e-5), (
        "Local attention: position 8 should see position 5 (inside window)"
    )


# ---------------------------------------------------------------------------
# 7. test_local_vs_full_differ
# ---------------------------------------------------------------------------


def test_local_vs_full_differ(model: StripedAttention) -> None:
    """Local and full attention produce different outputs for long sequences."""
    torch.manual_seed(3)
    T = 20  # much larger than window_size=4
    q = torch.randn(1, T, HEAD_DIM)
    k = torch.randn(1, T, HEAD_DIM)
    v = torch.randn(1, T, HEAD_DIM)

    out_full = model._full_attention(q, k, v)
    out_local = model._local_attention(q, k, v, WINDOW_SIZE)

    # They should NOT be equal since local can't see far-back positions
    assert not torch.allclose(out_full, out_local, atol=1e-4), (
        "Local and full attention should differ for long sequences"
    )


# ---------------------------------------------------------------------------
# 8. test_gradients_flow
# ---------------------------------------------------------------------------


def test_gradients_flow(cfg: StripedAttentionConfig) -> None:
    """Gradients flow through the forward pass without NaN or None."""
    model = StripedAttention(cfg)
    model.train()
    torch.manual_seed(4)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ---------------------------------------------------------------------------
# 9. test_short_seq_within_window
# ---------------------------------------------------------------------------


def test_short_seq_within_window(model: StripedAttention) -> None:
    """When T <= window_size, local attention is effectively the same as full."""
    torch.manual_seed(5)
    T = WINDOW_SIZE  # T == window_size -> all positions are within window
    q = torch.randn(1, T, HEAD_DIM)
    k = torch.randn(1, T, HEAD_DIM)
    v = torch.randn(1, T, HEAD_DIM)

    out_full = model._full_attention(q, k, v)
    out_local = model._local_attention(q, k, v, WINDOW_SIZE)

    assert torch.allclose(out_full, out_local, atol=1e-5), (
        "For T <= window_size, local attention should match full causal attention"
    )


# ---------------------------------------------------------------------------
# 10. test_batch_size_one
# ---------------------------------------------------------------------------


def test_batch_size_one(model: StripedAttention) -> None:
    """Works correctly with batch size 1."""
    torch.manual_seed(6)
    x = torch.randn(1, SEQ, D_MODEL)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 11. test_seq_len_one
# ---------------------------------------------------------------------------


def test_seq_len_one(model: StripedAttention) -> None:
    """Works correctly with a single-token sequence (T=1)."""
    torch.manual_seed(7)
    x = torch.randn(BATCH, 1, D_MODEL)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, 1, D_MODEL)


# ---------------------------------------------------------------------------
# 12. test_n_heads_2
# ---------------------------------------------------------------------------


def test_n_heads_2() -> None:
    """n_heads=2 yields one global head (idx 0) and one local head (idx 1)."""
    cfg = StripedAttentionConfig(d_model=64, n_heads=2, window_size=4)
    model = StripedAttention(cfg)
    model.train(False)

    assert model._is_global_head(0) is True
    assert model._is_global_head(1) is False

    mask = model.head_type_mask()
    assert mask == ["global", "local"]

    torch.manual_seed(8)
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 8, 64)


# ---------------------------------------------------------------------------
# 13. test_determinism
# ---------------------------------------------------------------------------


def test_determinism(model: StripedAttention, x: torch.Tensor) -> None:
    """In inference mode, the same input always yields the same output."""
    model.train(False)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Output should be deterministic in inference mode"


# ---------------------------------------------------------------------------
# 14. test_output_not_all_zeros
# ---------------------------------------------------------------------------


def test_output_not_all_zeros(model: StripedAttention, x: torch.Tensor) -> None:
    """Output is non-trivial (not all-zero) for a random input."""
    with torch.no_grad():
        out = model(x)
    assert out.abs().max() > 1e-6, "Output should not be all zeros"


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------


def test_registry() -> None:
    """MODEL_COMPONENT_REGISTRY['striped_attention'] is StripedAttention."""
    from src.model import MODEL_COMPONENT_REGISTRY

    assert "striped_attention" in MODEL_COMPONENT_REGISTRY, (
        "'striped_attention' not found in MODEL_COMPONENT_REGISTRY"
    )
    assert MODEL_COMPONENT_REGISTRY["striped_attention"] is StripedAttention
