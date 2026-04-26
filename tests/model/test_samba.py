"""Tests for SAMBA: Simple Hybrid State Space Models.

Reference: Ren et al., 2024, arXiv:2406.07522.

Covers SambaMambaBlock, SambaSWABlock, and SambaModel.
"""

from __future__ import annotations

import torch
from aurelius.model.samba import SambaMambaBlock, SambaModel, SambaSWABlock

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, T, D = 2, 16, 32  # batch, seq_len, d_model
N_HEADS = 4
D_STATE = 8
WINDOW = 8


def make_mamba(d_model: int = D) -> SambaMambaBlock:
    return SambaMambaBlock(d_model=d_model, d_state=D_STATE, d_conv=4, expand=2)


def make_swa(d_model: int = D, n_heads: int = N_HEADS, window_size: int = WINDOW) -> SambaSWABlock:
    return SambaSWABlock(d_model=d_model, n_heads=n_heads, window_size=window_size)


def make_model(n_layers: int = 6) -> SambaModel:
    return SambaModel(
        vocab_size=256,
        d_model=D,
        n_heads=N_HEADS,
        n_layers=n_layers,
        d_state=D_STATE,
        window_size=WINDOW,
        swa_every=3,
    )


def rand_input(b: int = B, t: int = T, d: int = D) -> torch.Tensor:
    return torch.randn(b, t, d)


def set_inference_mode(model: torch.nn.Module) -> torch.nn.Module:
    """Put model in inference mode (equivalent to model.eval())."""
    model.training = False
    for m in model.modules():
        m.training = False
    return model


# ---------------------------------------------------------------------------
# SambaMambaBlock tests (tests 1-4)
# ---------------------------------------------------------------------------


def test_mamba_block_output_shape():
    """Test 1: SambaMambaBlock output shape matches (B, T, d_model)."""
    block = make_mamba()
    x = rand_input()
    out = block(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


def test_mamba_block_output_finite():
    """Test 2: SambaMambaBlock output contains only finite values."""
    block = make_mamba()
    x = rand_input()
    out = block(x)
    assert torch.isfinite(out).all(), "Mamba block output contains non-finite values"


def test_mamba_block_gradient_flows():
    """Test 3: Gradients flow through SambaMambaBlock (no dead params)."""
    block = make_mamba()
    x = rand_input().requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input"
    assert x.grad.shape == x.shape
    param_grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed"


def test_mamba_block_causal():
    """Test 4: Mamba block is causal — extending the sequence does not change the prefix output."""
    block = make_mamba()
    set_inference_mode(block)

    x_short = rand_input(b=1, t=T)
    x_long = torch.cat([x_short, rand_input(b=1, t=4)], dim=1)

    with torch.no_grad():
        out_short = block(x_short)
        out_long = block(x_long)

    assert torch.allclose(out_short, out_long[:, :T], atol=1e-5), (
        "Mamba block is not causal: prefix output changed when sequence was extended"
    )


# ---------------------------------------------------------------------------
# SambaSWABlock tests (tests 5-9)
# ---------------------------------------------------------------------------


def test_swa_block_output_shape():
    """Test 5: SambaSWABlock output shape matches (B, T, d_model)."""
    block = make_swa()
    x = rand_input()
    out = block(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


def test_swa_block_output_finite():
    """Test 6: SambaSWABlock output contains only finite values."""
    block = make_swa()
    x = rand_input()
    out = block(x)
    assert torch.isfinite(out).all(), "SWA block output contains non-finite values"


def test_swa_block_gradient_flows():
    """Test 7: Gradients flow through SambaSWABlock."""
    block = make_swa()
    x = rand_input().requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input"
    param_grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed"


def test_swa_block_causal():
    """Test 8: SWA is causal — token t does NOT attend to token t+1.

    Modify x at position t+1 and verify that out[t] is unchanged.
    """
    block = make_swa()
    set_inference_mode(block)

    t_query = 5
    t_future = 6

    x1 = rand_input(b=1, t=T)
    x2 = x1.clone()
    x2[0, t_future] = x2[0, t_future] + torch.randn(D)

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    assert torch.allclose(out1[0, t_query], out2[0, t_query], atol=1e-5), (
        f"SWA is not causal: out[{t_query}] changed when x[{t_future}] was modified"
    )


def test_swa_window_limit():
    """Test 9: SWA window limit — out[query_pos] is unchanged when tokens beyond the window are modified.

    Token at query_pos can only attend to tokens in (query_pos-window_size, query_pos].
    Modifying a token at beyond_pos (distance >= window_size away) must not change the output.
    """  # noqa: E501
    block = make_swa(window_size=WINDOW)
    set_inference_mode(block)

    query_pos = WINDOW + 2
    beyond_pos = 0  # distance = query_pos - 0 = WINDOW+2 >= WINDOW: outside window

    x1 = rand_input(b=1, t=T)
    x2 = x1.clone()
    x2[0, beyond_pos] = x2[0, beyond_pos] + torch.randn(D)

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    assert torch.allclose(out1[0, query_pos], out2[0, query_pos], atol=1e-5), (
        f"SWA window limit violated: out[{query_pos}] changed when x[{beyond_pos}] "
        f"(distance={query_pos - beyond_pos} >= window_size={WINDOW}) was modified"
    )


# ---------------------------------------------------------------------------
# SambaModel tests (tests 10-14)
# ---------------------------------------------------------------------------


def test_model_output_shape():
    """Test 10: SambaModel output shape matches (B, T, d_model)."""
    model = make_model()
    ids = torch.randint(0, 256, (B, T))
    out = model(ids)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


def test_model_output_finite():
    """Test 11: SambaModel output contains only finite values."""
    model = make_model()
    ids = torch.randint(0, 256, (B, T))
    out = model(ids)
    assert torch.isfinite(out).all(), "SambaModel output contains non-finite values"


def test_model_gradient_flows():
    """Test 12: Gradients flow through SambaModel."""
    model = make_model()
    ids = torch.randint(0, 256, (B, T))
    out = model(ids)
    loss = out.sum()
    loss.backward()
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed in SambaModel"


def test_model_swa_block_count():
    """Test 13: n_layers=3 with swa_every=3 produces exactly 1 SWA block at index 2."""
    model = SambaModel(
        vocab_size=256,
        d_model=D,
        n_heads=N_HEADS,
        n_layers=3,
        d_state=D_STATE,
        window_size=WINDOW,
        swa_every=3,
    )
    swa_count = 0
    swa_indices = []
    for idx, layer in enumerate(model.layers):
        if isinstance(layer.block, SambaSWABlock):
            swa_count += 1
            swa_indices.append(idx)

    assert swa_count == 1, f"Expected 1 SWA block, found {swa_count}"
    assert swa_indices == [2], f"Expected SWA at index 2, got {swa_indices}"


def test_model_edge_case_batch1_seqlen1():
    """Test 14: SambaModel handles batch=1, seq_len=1 edge case without error."""
    model = make_model()
    set_inference_mode(model)
    ids = torch.randint(0, 256, (1, 1))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (1, 1, D), f"Expected (1, 1, {D}), got {out.shape}"
    assert torch.isfinite(out).all(), "Output not finite for batch=1, seq_len=1"
