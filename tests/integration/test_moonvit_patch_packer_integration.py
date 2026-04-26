"""Integration tests for MoonVitPatchPacker via MODEL_COMPONENT_REGISTRY.

Verifies:
  1. "moonvit_patch_packer" key present in MODEL_COMPONENT_REGISTRY.
  2. Construct from registry, forward pass, (patches, positions, mask) shapes correct.
  3. Output is a 3-tuple of tensors with expected dtypes.
  4. Existing MODEL_COMPONENT_REGISTRY entries are still present (regression guard).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 1. Registry membership
# ---------------------------------------------------------------------------


def test_moonvit_patch_packer_in_registry():
    from src.model import MODEL_COMPONENT_REGISTRY

    assert "moonvit_patch_packer" in MODEL_COMPONENT_REGISTRY, (
        "'moonvit_patch_packer' not found in MODEL_COMPONENT_REGISTRY"
    )


# ---------------------------------------------------------------------------
# 2. Registry class identity
# ---------------------------------------------------------------------------


def test_registry_class_is_moonvit_patch_packer():
    from src.model import MODEL_COMPONENT_REGISTRY
    from src.model.moonvit_patch_packer import MoonVitPatchPacker

    cls = MODEL_COMPONENT_REGISTRY["moonvit_patch_packer"]
    assert cls is MoonVitPatchPacker, f"Expected MoonVitPatchPacker, got {cls}"


# ---------------------------------------------------------------------------
# 3. Construct from registry + forward pass — output shapes
# ---------------------------------------------------------------------------


def test_registry_forward_shapes():
    """Simulate AureliusConfig dict pattern: build from config dict, run forward."""
    from src.model import MODEL_COMPONENT_REGISTRY
    from src.model.moonvit_patch_packer import MoonVitPatchPackerConfig

    cls = MODEL_COMPONENT_REGISTRY["moonvit_patch_packer"]

    # Build config from dict (simulating AureliusConfig pattern)
    cfg_kwargs = {
        "patch_size": 16,
        "num_frames": 4,
        "channels": 3,
        "max_patches": 4096,
    }
    cfg = MoonVitPatchPackerConfig(**cfg_kwargs)
    packer = cls(cfg)

    torch.manual_seed(0)
    pixel_values = torch.randn(1, 4, 3, 64, 64)
    output = packer(pixel_values)

    # Must return a 3-tuple
    assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"
    assert len(output) == 3, f"Expected 3-tuple, got length {len(output)}"

    patches, positions, mask = output

    # T=4, H/16=4, W/16=4 → N = 4*4*4 = 64 patches, patch_dim = 16*16*3 = 768
    assert patches.shape == (1, 64, 768), f"patches shape mismatch: {patches.shape}"
    assert positions.shape == (1, 64, 3), f"positions shape mismatch: {positions.shape}"
    assert mask.shape == (1, 64), f"mask shape mismatch: {mask.shape}"


# ---------------------------------------------------------------------------
# 4. Output tensor dtypes
# ---------------------------------------------------------------------------


def test_registry_output_dtypes():
    """patches/mask are float, positions are long integer indices."""
    from src.model import MODEL_COMPONENT_REGISTRY
    from src.model.moonvit_patch_packer import MoonVitPatchPackerConfig

    cls = MODEL_COMPONENT_REGISTRY["moonvit_patch_packer"]
    cfg = MoonVitPatchPackerConfig(patch_size=16, num_frames=4, channels=3)
    packer = cls(cfg)

    x = torch.randn(1, 4, 3, 64, 64)
    patches, positions, mask = packer(x)

    assert patches.is_floating_point(), f"patches should be float, got {patches.dtype}"
    assert positions.dtype == torch.long, f"positions should be long, got {positions.dtype}"
    assert mask.is_floating_point(), f"mask should be float, got {mask.dtype}"


# ---------------------------------------------------------------------------
# 5. Regression guard: existing MODEL_COMPONENT_REGISTRY keys still present
# ---------------------------------------------------------------------------


def test_existing_registry_keys_intact():
    from src.model import MODEL_COMPONENT_REGISTRY

    expected_keys = {"dsa_attention", "mtp_shared", "dp_aware_moe_routing", "mla_256"}
    for key in expected_keys:
        assert key in MODEL_COMPONENT_REGISTRY, (
            f"Existing registry key '{key}' is missing — regression!"
        )
