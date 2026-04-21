"""Unit tests for MoonVitPatchPacker and MoonVitPatchPackerConfig.

Tiny test config: patch_size=4 (overriding default 16 to keep tensors small).
All tests use pure PyTorch — no transformers, einops, flash_attn, etc.

Coverage target: 15 unit tests (spec requires 10-16).
"""

from __future__ import annotations

import pytest
import torch
import warnings

from src.model.moonvit_patch_packer import MoonVitPatchPacker, MoonVitPatchPackerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_packer(patch_size: int = 4, num_frames: int = 2, channels: int = 3,
                max_patches: int = 4096) -> MoonVitPatchPacker:
    """Construct a packer with the given tiny-test config."""
    cfg = MoonVitPatchPackerConfig(
        patch_size=patch_size,
        num_frames=num_frames,
        channels=channels,
        max_patches=max_patches,
    )
    return MoonVitPatchPacker(cfg)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """Default config values: patch_size=16, num_frames=4, channels=3, patch_dim=768."""
    cfg = MoonVitPatchPackerConfig()
    assert cfg.patch_size == 16
    assert cfg.num_frames == 4
    assert cfg.channels == 3
    assert cfg.max_patches == 4096
    assert cfg.patch_dim == 16 * 16 * 3


# ---------------------------------------------------------------------------
# 2. test_output_shapes_fixed_res
# ---------------------------------------------------------------------------

def test_output_shapes_fixed_res():
    """Input [1, 4, 3, 64, 64] with patch_size=16 → patches [1, 64, 768]."""
    cfg = MoonVitPatchPackerConfig(patch_size=16, num_frames=4, channels=3)
    packer = MoonVitPatchPacker(cfg)
    x = torch.randn(1, 4, 3, 64, 64)
    patches, positions, mask = packer(x)

    # T=4, H=64/16=4 rows, W=64/16=4 cols → 4*4*4 = 64 patches
    assert patches.shape == (1, 64, 768), f"patches shape {patches.shape}"
    assert positions.shape == (1, 64, 3), f"positions shape {positions.shape}"
    assert mask.shape == (1, 64), f"mask shape {mask.shape}"


# ---------------------------------------------------------------------------
# 3. test_output_shapes_small
# ---------------------------------------------------------------------------

def test_output_shapes_small():
    """Input [2, 2, 3, 8, 8] with patch_size=4 → patches [2, 8, 48], mask valid."""
    packer = make_packer(patch_size=4, num_frames=2, channels=3)
    x = torch.randn(2, 2, 3, 8, 8)
    patches, positions, mask = packer(x)

    # T=2, H/ps=2, W/ps=2 → 2*2*2 = 8 patches
    assert patches.shape == (2, 8, 48), f"patches shape {patches.shape}"
    assert mask.shape == (2, 8)
    assert positions.shape == (2, 8, 3)


# ---------------------------------------------------------------------------
# 4. test_patch_dim_correct
# ---------------------------------------------------------------------------

def test_patch_dim_correct():
    """patch_dim == patch_size^2 * channels for various configs."""
    for ps, ch in [(4, 1), (4, 3), (8, 2), (16, 3)]:
        cfg = MoonVitPatchPackerConfig(patch_size=ps, channels=ch)
        assert cfg.patch_dim == ps * ps * ch, (
            f"patch_dim mismatch for ps={ps} ch={ch}: {cfg.patch_dim}"
        )


# ---------------------------------------------------------------------------
# 5. test_positions_frame_index
# ---------------------------------------------------------------------------

def test_positions_frame_index():
    """positions[:, :, 0] should be correct frame indices (0..T-1 repeated)."""
    packer = make_packer(patch_size=4, num_frames=2, channels=1)
    x = torch.randn(1, 2, 1, 8, 8)  # T=2, H=W=8 → 4 patches/frame
    patches, positions, mask = packer(x)

    # First 4 patches: frame 0; next 4 patches: frame 1
    frame_indices = positions[0, :, 0]  # [8]
    expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert torch.equal(frame_indices, expected), (
        f"frame indices mismatch: {frame_indices.tolist()}"
    )


# ---------------------------------------------------------------------------
# 6. test_positions_row_col
# ---------------------------------------------------------------------------

def test_positions_row_col():
    """positions[:, :, 1] and [:, :, 2] are valid grid row/col indices."""
    packer = make_packer(patch_size=4, num_frames=1, channels=1)
    x = torch.randn(1, 1, 1, 8, 8)  # T=1, 2x2 grid → 4 patches
    patches, positions, mask = packer(x)

    rows = positions[0, :, 1]
    cols = positions[0, :, 2]
    n_rows = 8 // 4
    n_cols = 8 // 4

    assert rows.min() >= 0 and rows.max() < n_rows, (
        f"rows out of range [0, {n_rows}): {rows.tolist()}"
    )
    assert cols.min() >= 0 and cols.max() < n_cols, (
        f"cols out of range [0, {n_cols}): {cols.tolist()}"
    )
    # All (row, col) combinations should be present exactly once
    pairs = set(zip(rows.tolist(), cols.tolist()))
    expected_pairs = {(r, c) for r in range(n_rows) for c in range(n_cols)}
    assert pairs == expected_pairs, f"Missing grid positions: {expected_pairs - pairs}"


# ---------------------------------------------------------------------------
# 7. test_mask_all_ones_uniform
# ---------------------------------------------------------------------------

def test_mask_all_ones_uniform():
    """When all frames have the same size, mask is all 1s (no padding slots)."""
    packer = make_packer(patch_size=4, num_frames=2, channels=3)
    x = torch.randn(3, 2, 3, 8, 8)
    patches, positions, mask = packer(x)

    assert mask.shape == (3, 8)
    assert (mask == 1.0).all(), f"mask should be all 1s, got: {mask}"


# ---------------------------------------------------------------------------
# 8. test_padding_to_patch_multiple
# ---------------------------------------------------------------------------

def test_padding_to_patch_multiple():
    """Input H=6 (not multiple of 4): packs correctly after padding to H=8."""
    packer = make_packer(patch_size=4, num_frames=1, channels=1)
    x = torch.randn(1, 1, 1, 6, 8)  # H=6 → padded to 8 (2 rows), W=8 (2 cols)
    patches, positions, mask = packer(x)

    # ceil(6/4)=2 rows, ceil(8/4)=2 cols → 4 patches total
    assert patches.shape == (1, 4, 1 * 4 * 4), f"unexpected shape {patches.shape}"
    assert (mask == 1.0).all()


# ---------------------------------------------------------------------------
# 9. test_single_frame
# ---------------------------------------------------------------------------

def test_single_frame():
    """T=1 input works correctly."""
    packer = make_packer(patch_size=4, num_frames=1, channels=3)
    x = torch.randn(1, 1, 3, 8, 8)
    patches, positions, mask = packer(x)

    # 1 frame × 2×2 patches = 4
    assert patches.shape == (1, 4, 48)
    assert (positions[:, :, 0] == 0).all(), "All frame indices should be 0 for T=1"


# ---------------------------------------------------------------------------
# 10. test_batch_size_one
# ---------------------------------------------------------------------------

def test_batch_size_one():
    """B=1 works correctly."""
    packer = make_packer(patch_size=4, num_frames=2, channels=3)
    x = torch.randn(1, 2, 3, 8, 8)
    patches, positions, mask = packer(x)
    assert patches.shape[0] == 1
    assert positions.shape[0] == 1
    assert mask.shape[0] == 1


# ---------------------------------------------------------------------------
# 11. test_batch_size_two
# ---------------------------------------------------------------------------

def test_batch_size_two():
    """B=2 works with same-shape inputs."""
    packer = make_packer(patch_size=4, num_frames=2, channels=3)
    x = torch.randn(2, 2, 3, 8, 8)
    patches, positions, mask = packer(x)
    assert patches.shape[0] == 2
    assert positions.shape[0] == 2
    assert mask.shape[0] == 2
    # Both batch items should have identical mask (same resolution)
    assert torch.equal(mask[0], mask[1])


# ---------------------------------------------------------------------------
# 12. test_no_grad_positions
# ---------------------------------------------------------------------------

def test_no_grad_positions():
    """positions tensor has no gradient (it's an integer index tensor)."""
    packer = make_packer(patch_size=4, num_frames=2, channels=3)
    x = torch.randn(1, 2, 3, 8, 8, requires_grad=True)
    patches, positions, mask = packer(x)

    assert positions.dtype == torch.long, (
        f"positions should be long, got {positions.dtype}"
    )
    assert not positions.requires_grad, "positions should not require grad"


# ---------------------------------------------------------------------------
# 13. test_max_patches_config
# ---------------------------------------------------------------------------

def test_max_patches_config():
    """max_patches truncates output when total patches would exceed the limit."""
    # 2 frames × 4 patches/frame = 8 normally; cap at 6
    packer = make_packer(patch_size=4, num_frames=2, channels=1, max_patches=6)
    x = torch.randn(1, 2, 1, 8, 8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        patches, positions, mask = packer(x)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert "max_patches" in str(w[0].message).lower()

    assert patches.shape[1] == 6, f"Expected 6 patches, got {patches.shape[1]}"


# ---------------------------------------------------------------------------
# 14. test_determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Same input produces identical output on two forward passes."""
    torch.manual_seed(42)
    packer = make_packer(patch_size=4, num_frames=2, channels=3)
    x = torch.randn(2, 2, 3, 8, 8)

    patches_a, positions_a, mask_a = packer(x)
    patches_b, positions_b, mask_b = packer(x)

    assert torch.equal(patches_a, patches_b), "patches should be deterministic"
    assert torch.equal(positions_a, positions_b), "positions should be deterministic"
    assert torch.equal(mask_a, mask_b), "mask should be deterministic"


# ---------------------------------------------------------------------------
# 15. test_tiny_config
# ---------------------------------------------------------------------------

def test_tiny_config():
    """patch_size=4, num_frames=2, channels=1, input [1,2,1,8,8] → patches [1,8,16]."""
    cfg = MoonVitPatchPackerConfig(patch_size=4, num_frames=2, channels=1)
    packer = MoonVitPatchPacker(cfg)
    x = torch.randn(1, 2, 1, 8, 8)
    patches, positions, mask = packer(x)

    # T=2, 2×2 grid, ch=1, patch_dim=4*4*1=16 → 2*4 = 8 patches
    assert patches.shape == (1, 8, 16), f"unexpected shape {patches.shape}"
    assert positions.shape == (1, 8, 3)
    assert mask.shape == (1, 8)
    assert (mask == 1.0).all()


# ---------------------------------------------------------------------------
# 16. test_pack_single_shape
# ---------------------------------------------------------------------------

def test_pack_single_shape():
    """pack_single returns correct shape for a single frame."""
    cfg = MoonVitPatchPackerConfig(patch_size=4, channels=3)
    packer = MoonVitPatchPacker(cfg)
    frame = torch.randn(3, 12, 8)  # C=3, H=12, W=8 → 3 rows × 2 cols = 6 patches
    out = packer.pack_single(frame)
    assert out.shape == (6, 48), f"unexpected shape {out.shape}"
