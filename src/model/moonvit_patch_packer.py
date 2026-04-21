"""MoonViT Patch Packer — NaViT-style spatiotemporal patch packing.

Implements the patch packing strategy from Kimi K2.5 (arXiv:2602.02276 §4).
Converts 2D/3D image frames into a 1D sequence of patch tokens for use as
visual tokens prepended to the language model sequence.

Key features:
- Variable-resolution images via NaViT-style packing (no fixed grid required)
- Spatial patch size: configurable (default 16×16 pixels)
- Temporal: packs T frames into sequence
- Output: (patches, positions, mask) triple
- patch_positions encodes (frame_idx, row_idx, col_idx) per patch
- attention_mask marks valid vs padding patches when batching variable-size inputs

Usage::

    cfg = MoonVitPatchPackerConfig()
    packer = MoonVitPatchPacker(cfg)
    pixel_values = torch.randn(2, 4, 3, 64, 64)   # [B, T, C, H, W]
    patches, positions, mask = packer(pixel_values)
    # patches:   [2, 64, 768]
    # positions: [2, 64, 3]
    # mask:      [2, 64]
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoonVitPatchPackerConfig:
    """Configuration for MoonVitPatchPacker.

    Attributes:
        patch_size:  Side length (pixels) of each square spatial patch.
        num_frames:  Number of temporal frames to pack per clip.
        channels:    Number of input image channels (e.g. 3 for RGB).
        max_patches: Maximum number of patches per sequence. Sequences that
                     would exceed this are truncated (with a warning).
                     Defaults to 4096.  Set to 0 to disable the limit.
        enabled:     Feature toggle — defaults to True.  Set False to skip
                     packing (returns raw pixel values unchanged).
    """

    patch_size: int = 16
    num_frames: int = 4
    channels: int = 3
    max_patches: int = 4096
    enabled: bool = True

    @property
    def patch_dim(self) -> int:
        """Flattened patch dimensionality: patch_size² × channels."""
        return self.patch_size * self.patch_size * self.channels


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class MoonVitPatchPacker(nn.Module):
    """NaViT spatiotemporal patch packer.

    Converts a batch of video clips [B, T, C, H, W] into packed patch
    sequences with positional bookkeeping.

    Args:
        config: :class:`MoonVitPatchPackerConfig` instance.

    Inputs (forward):
        pixel_values: Float tensor ``[B, T, C, H, W]``.
            *T* need not equal ``config.num_frames``; the module packs
            whatever T frames are provided.  H and W may be any positive
            integer — they are padded to the nearest multiple of
            ``patch_size`` before extraction.

    Returns:
        A 3-tuple ``(patches, positions, mask)``:

        * **patches** — ``[B, N, patch_dim]`` float tensor of flattened
          patch pixels.
        * **positions** — ``[B, N, 3]`` long tensor encoding
          ``(frame_idx, row_idx, col_idx)`` for every patch slot.  Padding
          slots carry the last valid position (content is masked out).
        * **mask** — ``[B, N]`` float tensor; 1 for valid patches, 0 for
          padding patches injected to make the batch rectangular.
    """

    def __init__(self, config: MoonVitPatchPackerConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack_single(self, frame: Tensor) -> Tensor:
        """Extract and flatten all patches from a single image frame.

        Args:
            frame: ``[C, H, W]`` float tensor.

        Returns:
            ``[n_patches, patch_dim]`` float tensor where
            ``n_patches = ceil(H/patch_size) × ceil(W/patch_size)``.
        """
        ps = self.config.patch_size
        C, H, W = frame.shape

        # Pad H and W to multiples of patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            # F.pad pads in reverse-dim order: (left, right, top, bottom)
            frame = torch.nn.functional.pad(frame, (0, pad_w, 0, pad_h))

        H_pad = H + pad_h
        W_pad = W + pad_w
        n_rows = H_pad // ps
        n_cols = W_pad // ps

        # Reshape into patches: [C, n_rows, ps, n_cols, ps]
        frame = frame.view(C, n_rows, ps, n_cols, ps)
        # → [n_rows, n_cols, C, ps, ps]
        frame = frame.permute(1, 3, 0, 2, 4).contiguous()
        # → [n_rows * n_cols, patch_dim]
        frame = frame.view(n_rows * n_cols, C * ps * ps)
        return frame

    def forward(
        self,
        pixel_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pack a batch of video clips into patch sequences.

        Args:
            pixel_values: ``[B, T, C, H, W]`` float tensor.

        Returns:
            ``(patches, positions, mask)`` — see class docstring.
        """
        B, T, C, H, W = pixel_values.shape
        ps = self.config.patch_size
        patch_dim = self.config.patch_dim

        # Compute expected number of patches per frame (after padding)
        n_rows = math.ceil(H / ps)
        n_cols = math.ceil(W / ps)
        patches_per_frame = n_rows * n_cols
        total_patches = T * patches_per_frame

        # max_patches guard
        max_p = self.config.max_patches
        if max_p > 0 and total_patches > max_p:
            warnings.warn(
                f"MoonVitPatchPacker: total_patches={total_patches} exceeds "
                f"max_patches={max_p}. Truncating to {max_p} patches.",
                stacklevel=2,
            )
            total_patches = max_p

        # Allocate output tensors
        patches_out = pixel_values.new_zeros(B, total_patches, patch_dim)
        positions_out = pixel_values.new_zeros(B, total_patches, 3, dtype=torch.long)
        mask_out = pixel_values.new_zeros(B, total_patches)

        for b in range(B):
            slot = 0
            for t in range(T):
                frame = pixel_values[b, t]   # [C, H, W]
                frame_patches = self.pack_single(frame)  # [patches_per_frame, patch_dim]
                fp = frame_patches.shape[0]

                remaining = total_patches - slot
                take = min(fp, remaining)
                if take <= 0:
                    break

                patches_out[b, slot:slot + take] = frame_patches[:take]
                mask_out[b, slot:slot + take] = 1.0

                # Build positions for this frame's patches
                for idx in range(take):
                    patch_in_frame = idx   # local index within frame
                    row_idx = patch_in_frame // n_cols
                    col_idx = patch_in_frame % n_cols
                    positions_out[b, slot + idx, 0] = t
                    positions_out[b, slot + idx, 1] = row_idx
                    positions_out[b, slot + idx, 2] = col_idx

                slot += take
                if slot >= total_patches:
                    break

        return patches_out, positions_out, mask_out


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------
# MODEL_COMPONENT_REGISTRY lives in src/model/__init__.py.
# Registration is performed there (see __init__.py) to avoid circular imports.
# The registry key is "moonvit_patch_packer".
# ---------------------------------------------------------------------------

__all__ = [
    "MoonVitPatchPacker",
    "MoonVitPatchPackerConfig",
]
