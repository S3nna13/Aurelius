"""Tests for Vision Mamba (Vim) — src/model/vim.py.

Tiny config: img_size=32, patch_size=8 → N=16 patches,
             d_model=32, n_layers=2, n_classes=10, d_state=8.
"""

from __future__ import annotations

import torch
from aurelius.model.vim import BidirectionalSSM, PatchEmbed, VimBlock, VisionMamba

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
IMG_SIZE = 32
PATCH_SIZE = 8
IN_CHANS = 3
D_MODEL = 32
N_LAYERS = 2
N_CLASSES = 10
D_STATE = 8
N_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 16

BATCH = 2
T = 10  # arbitrary sequence length for SSM / VimBlock tests


def _img(B: int = BATCH) -> torch.Tensor:
    return torch.randn(B, IN_CHANS, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# PatchEmbed tests
# ---------------------------------------------------------------------------


def test_patch_embed_output_shape():
    """PatchEmbed output must be (B, N, d_model) with N=(H//P)*(W//P)."""
    pe = PatchEmbed(IMG_SIZE, PATCH_SIZE, IN_CHANS, D_MODEL)
    out = pe(_img())
    assert out.shape == (BATCH, N_PATCHES, D_MODEL)


def test_patch_embed_non_square_image():
    """PatchEmbed should handle rectangular images (H != W)."""
    pe = PatchEmbed(img_size=32, patch_size=8, in_chans=IN_CHANS, d_model=D_MODEL)
    x = torch.randn(BATCH, IN_CHANS, 32, 64)
    out = pe(x)
    expected_n = (32 // 8) * (64 // 8)  # 4 * 8 = 32
    assert out.shape == (BATCH, expected_n, D_MODEL)


# ---------------------------------------------------------------------------
# BidirectionalSSM tests
# ---------------------------------------------------------------------------


def test_bidir_ssm_output_shape():
    """BidirectionalSSM output must be (B, T, d_model)."""
    ssm = BidirectionalSSM(D_MODEL, d_state=D_STATE)
    x = torch.randn(BATCH, T, D_MODEL)
    out = ssm(x)
    assert out.shape == (BATCH, T, D_MODEL)


def test_bidir_ssm_output_finite():
    """BidirectionalSSM output must contain only finite values."""
    ssm = BidirectionalSSM(D_MODEL, d_state=D_STATE)
    x = torch.randn(BATCH, T, D_MODEL)
    out = ssm(x)
    assert torch.isfinite(out).all()


def test_bidir_ssm_both_directions_contribute():
    """Forward and backward scans both contribute: output != forward-only output."""
    ssm = BidirectionalSSM(D_MODEL, d_state=D_STATE)
    x = torch.randn(BATCH, T, D_MODEL)

    # Full bidirectional output
    out_bidir = ssm(x)

    # Forward-only: zero out backward projection so backward scan produces zeros
    with torch.no_grad():
        saved_in_bwd = ssm.in_proj_bwd.weight.clone()
        ssm.in_proj_bwd.weight.zero_()

        out_fwd_only = ssm(x)

        # Restore
        ssm.in_proj_bwd.weight.copy_(saved_in_bwd)

    assert not torch.allclose(out_bidir, out_fwd_only), (
        "Expected bidirectional output to differ from forward-only output"
    )


def test_bidir_ssm_gradient_flows():
    """Gradient must flow through BidirectionalSSM to the input."""
    ssm = BidirectionalSSM(D_MODEL, d_state=D_STATE)
    x = torch.randn(BATCH, T, D_MODEL, requires_grad=True)
    out = ssm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# VimBlock tests
# ---------------------------------------------------------------------------


def test_vim_block_output_shape():
    """VimBlock output must be (B, T, d_model)."""
    block = VimBlock(D_MODEL, d_state=D_STATE)
    x = torch.randn(BATCH, T, D_MODEL)
    out = block(x)
    assert out.shape == (BATCH, T, D_MODEL)


def test_vim_block_output_finite():
    """VimBlock output must contain only finite values."""
    block = VimBlock(D_MODEL, d_state=D_STATE)
    x = torch.randn(BATCH, T, D_MODEL)
    out = block(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# VisionMamba tests
# ---------------------------------------------------------------------------


def _model() -> VisionMamba:
    return VisionMamba(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_classes=N_CLASSES,
        d_state=D_STATE,
    )


def test_vision_mamba_output_shape():
    """VisionMamba output must be (B, n_classes)."""
    m = _model()
    out = m(_img())
    assert out.shape == (BATCH, N_CLASSES)


def test_vision_mamba_output_finite():
    """VisionMamba output must contain only finite values."""
    m = _model()
    out = m(_img())
    assert torch.isfinite(out).all()


def test_vision_mamba_gradient_flows():
    """Gradient must flow through the full VisionMamba model to the input."""
    m = _model()
    x = _img().requires_grad_(True)
    out = m(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_vision_mamba_small_config():
    """VisionMamba must work with patch_size=8, img_size=32 (non-default config)."""
    m = VisionMamba(
        img_size=32,
        patch_size=8,
        in_chans=3,
        d_model=16,
        n_layers=1,
        n_classes=5,
        d_state=4,
    )
    x = torch.randn(1, 3, 32, 32)
    out = m(x)
    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()


def test_vision_mamba_batch_size_one():
    """VisionMamba must work with batch size 1."""
    m = _model()
    out = m(_img(B=1))
    assert out.shape == (1, N_CLASSES)
    assert torch.isfinite(out).all()


def test_vision_mamba_cls_uses_all_patches():
    """CLS-based output must depend on all patch content.

    Strongly perturbing a spatial region must change the output logits,
    confirming all tokens influence the CLS representation via the SSM.
    """
    m = _model()
    m.train(False)  # inference mode, no dropout etc.

    x1 = torch.randn(1, IN_CHANS, IMG_SIZE, IMG_SIZE)
    x2 = x1.clone()
    # Perturb the bottom-right quadrant strongly
    x2[:, :, IMG_SIZE // 2 :, IMG_SIZE // 2 :] += 10.0

    with torch.no_grad():
        logits1 = m(x1)
        logits2 = m(x2)

    assert not torch.allclose(logits1, logits2), (
        "Logits should change when a spatial patch region is strongly perturbed"
    )
