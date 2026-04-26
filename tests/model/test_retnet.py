"""Tests for RetNet — Retention Networks (Sun et al. 2023)."""

import math

import torch

from src.model.config import AureliusConfig
from src.model.retnet import MultiScaleRetention, RetNetBlock, SimpleRetention

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_config():
    """Tiny AureliusConfig for block-level tests."""
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


# ---------------------------------------------------------------------------
# SimpleRetention — parallel mode shape
# ---------------------------------------------------------------------------


def test_simple_retention_parallel_shape():
    """forward_parallel: (2, 16, 64) -> (2, 16, head_dim)."""
    head_dim = 16
    sr = SimpleRetention(d_model=64, head_dim=head_dim, gamma=0.9)
    x = torch.randn(2, 16, 64)
    out = sr.forward_parallel(x)
    assert out.shape == (2, 16, head_dim), f"Expected (2, 16, {head_dim}), got {out.shape}"


# ---------------------------------------------------------------------------
# Decay mask
# ---------------------------------------------------------------------------


def test_decay_mask_causal():
    """D[i, j] == 0 for all i < j (strict upper triangle is zero)."""
    sr = SimpleRetention(d_model=32, head_dim=8, gamma=0.9)
    L = 8
    D = sr._decay_mask(L, device=torch.device("cpu"))
    assert D.shape == (L, L)
    for i in range(L):
        for j in range(i + 1, L):
            assert D[i, j].item() == 0.0, f"D[{i},{j}] = {D[i, j].item()} should be 0 (causal)"


def test_decay_mask_diagonal():
    """D[i, i] == 1.0 for all i (gamma^0 = 1)."""
    sr = SimpleRetention(d_model=32, head_dim=8, gamma=0.85)
    L = 6
    D = sr._decay_mask(L, device=torch.device("cpu"))
    for i in range(L):
        assert abs(D[i, i].item() - 1.0) < 1e-5, f"D[{i},{i}] = {D[i, i].item()} should be 1.0"


# ---------------------------------------------------------------------------
# Recurrent mode
# ---------------------------------------------------------------------------


def test_recurrent_matches_parallel():
    """For L=1 a single recurrent step must match parallel mode output."""
    torch.manual_seed(42)
    head_dim = 8
    sr = SimpleRetention(d_model=16, head_dim=head_dim, gamma=0.9)
    sr.eval()

    x = torch.randn(2, 1, 16)  # (B=2, L=1, d_model=16)

    with torch.no_grad():
        out_par = sr.forward_parallel(x)
        out_rec, _ = sr.forward_recurrent(x, state=None)

    torch.testing.assert_close(
        out_par,
        out_rec,
        atol=1e-5,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# MultiScaleRetention
# ---------------------------------------------------------------------------


def test_multi_scale_retention_shape():
    """MultiScaleRetention: (2, 8, 64) -> (2, 8, 64)."""
    msr = MultiScaleRetention(d_model=64, n_heads=4)
    x = torch.randn(2, 8, 64)
    out = msr(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"


def test_different_gammas():
    """Each head in MultiScaleRetention must have a distinct gamma value."""
    n_heads = 4
    msr = MultiScaleRetention(d_model=64, n_heads=n_heads)
    gammas = [h.gamma for h in msr.heads]

    assert len(set(gammas)) == n_heads, (
        f"Expected {n_heads} distinct gammas, got {len(set(gammas))}: {gammas}"
    )
    for i, h in enumerate(msr.heads):
        expected = 1 - 2 ** (-5 - math.floor(8 * i / n_heads))
        assert abs(h.gamma - expected) < 1e-9, f"Head {i}: expected gamma {expected}, got {h.gamma}"


# ---------------------------------------------------------------------------
# RetNetBlock
# ---------------------------------------------------------------------------


def test_retnet_block_shape():
    """RetNetBlock: (2, 8, 64) -> (2, 8, 64)."""
    config = make_config()
    block = RetNetBlock(config)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"


def test_retnet_recurrent_mode():
    """RetNetBlock.forward(x, recurrent=True) returns shape (2, 8, 64)."""
    config = make_config()
    block = RetNetBlock(config)
    x = torch.randn(2, 8, 64)
    out = block(x, recurrent=True)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64) in recurrent mode, got {out.shape}"
