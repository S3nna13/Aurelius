"""Tests for src/model/vision_encoder.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.vision_encoder import (
    MultimodalEmbedding,
    PatchEmbedding,
    VisionConfig,
    VisionTransformer,
    VisualProjection,
    create_visual_attention_mask,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# Small config used throughout tests to keep things fast.
SMALL_CFG = VisionConfig(
    image_size=32,
    patch_size=16,
    n_channels=3,
    d_vision=32,
    n_vision_layers=1,
    n_vision_heads=2,
    d_lm=64,
)

# 32x32 image → (32//16)^2 = 4 patches + 1 CLS = 5 visual tokens
N_PATCHES_SMALL = (32 // 16) ** 2  # 4
N_VISUAL_SMALL = N_PATCHES_SMALL + 1  # 5  (includes CLS)

torch.manual_seed(0)


def _make_images(B: int = 1, cfg: VisionConfig = SMALL_CFG) -> torch.Tensor:
    return torch.randn(B, cfg.n_channels, cfg.image_size, cfg.image_size)


def _make_text_embed(vocab: int = 100, cfg: VisionConfig = SMALL_CFG) -> nn.Embedding:
    return nn.Embedding(vocab, cfg.d_lm)


# ---------------------------------------------------------------------------
# 1. test_vision_config_defaults
# ---------------------------------------------------------------------------


def test_vision_config_defaults():
    cfg = VisionConfig()
    assert cfg.image_size == 224
    assert cfg.patch_size == 16
    assert cfg.n_channels == 3
    assert cfg.d_vision == 256
    assert cfg.n_vision_layers == 2
    assert cfg.n_vision_heads == 4
    assert cfg.d_lm == 64
    assert cfg.max_visual_tokens == 196


# ---------------------------------------------------------------------------
# 2. test_patch_embedding_output_shape
# ---------------------------------------------------------------------------


def test_patch_embedding_output_shape():
    torch.manual_seed(0)
    embed = PatchEmbedding(SMALL_CFG)
    images = _make_images(B=2)
    out = embed(images)
    assert out.shape == (2, N_VISUAL_SMALL, SMALL_CFG.d_vision), (
        f"Expected (2, {N_VISUAL_SMALL}, {SMALL_CFG.d_vision}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 3. test_patch_embedding_cls_token
# ---------------------------------------------------------------------------


def test_patch_embedding_cls_token():
    """The first token position should be the CLS token (from cls_token parameter)."""
    torch.manual_seed(0)
    embed = PatchEmbedding(SMALL_CFG)
    # Zero out positional embeddings so we can isolate the CLS token value.
    with torch.no_grad():
        embed.pos_embed.zero_()
        # Set CLS token to a recognisable value.
        embed.cls_token.fill_(99.0)

    images = torch.zeros(1, SMALL_CFG.n_channels, SMALL_CFG.image_size, SMALL_CFG.image_size)
    out = embed(images)

    # CLS token is at index 0; with zero pos_embed and zero image it should be 99.
    assert torch.allclose(out[0, 0], torch.full((SMALL_CFG.d_vision,), 99.0)), (
        "First token should be the CLS token value"
    )


# ---------------------------------------------------------------------------
# 4. test_vision_transformer_output_shape
# ---------------------------------------------------------------------------


def test_vision_transformer_output_shape():
    torch.manual_seed(0)
    vit = VisionTransformer(SMALL_CFG)
    images = _make_images(B=2)
    out = vit(images)
    assert out.shape == (2, N_VISUAL_SMALL, SMALL_CFG.d_vision)


# ---------------------------------------------------------------------------
# 5. test_visual_projection_output_shape
# ---------------------------------------------------------------------------


def test_visual_projection_output_shape():
    torch.manual_seed(0)
    proj = VisualProjection(SMALL_CFG)
    visual_tokens = torch.randn(2, N_VISUAL_SMALL, SMALL_CFG.d_vision)
    out = proj(visual_tokens)
    assert out.shape == (2, N_VISUAL_SMALL, SMALL_CFG.d_lm)


# ---------------------------------------------------------------------------
# 6. test_multimodal_embedding_text_only
# ---------------------------------------------------------------------------


def test_multimodal_embedding_text_only():
    torch.manual_seed(0)
    text_embed = _make_text_embed()
    mm = MultimodalEmbedding(text_embed, SMALL_CFG)

    B, T = 2, 10
    input_ids = torch.randint(0, 100, (B, T))
    out = mm(input_ids, images=None)

    assert out.shape == (B, T, SMALL_CFG.d_lm)


# ---------------------------------------------------------------------------
# 7. test_multimodal_embedding_with_image
# ---------------------------------------------------------------------------


def test_multimodal_embedding_with_image():
    torch.manual_seed(0)
    text_embed = _make_text_embed()
    mm = MultimodalEmbedding(text_embed, SMALL_CFG)

    B, T = 1, 5
    input_ids = torch.randint(0, 100, (B, T))
    images = _make_images(B=B)
    out = mm(input_ids, images=images)

    expected_len = N_VISUAL_SMALL + T
    assert out.shape == (B, expected_len, SMALL_CFG.d_lm), (
        f"Expected (B, {expected_len}, {SMALL_CFG.d_lm}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 8. test_visual_attention_mask_shape
# ---------------------------------------------------------------------------


def test_visual_attention_mask_shape():
    n_visual, n_text = 5, 10
    mask = create_visual_attention_mask(n_visual, n_text)
    T_total = n_visual + n_text
    assert mask.shape == (T_total, T_total)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 9. test_visual_attention_mask_text_causal
# ---------------------------------------------------------------------------


def test_visual_attention_mask_text_causal():
    """Text tokens must be causal: token i attends to text tokens 0..i only."""
    n_visual, n_text = 3, 4
    mask = create_visual_attention_mask(n_visual, n_text)

    # Check text-to-text sub-block (lower-triangular)
    text_block = mask[n_visual:, n_visual:]  # (n_text, n_text)

    for i in range(n_text):
        for j in range(n_text):
            expected = j <= i  # causal: can attend to same or earlier position
            assert text_block[i, j].item() == expected, f"text_block[{i},{j}] should be {expected}"

    # All text tokens must be able to attend to all visual tokens.
    text_to_visual = mask[n_visual:, :n_visual]
    assert text_to_visual.all(), "All text tokens should attend to all visual tokens"


# ---------------------------------------------------------------------------
# 10. test_patch_embedding_different_image_sizes
# ---------------------------------------------------------------------------


def test_patch_embedding_different_image_sizes():
    """PatchEmbedding must work with image_size=32, patch_size=16 → 4 patches."""
    torch.manual_seed(0)
    cfg = VisionConfig(image_size=32, patch_size=16, d_vision=32, n_vision_heads=2)
    embed = PatchEmbedding(cfg)
    images = torch.randn(1, cfg.n_channels, 32, 32)
    out = embed(images)
    expected_patches = (32 // 16) ** 2  # 4
    assert out.shape == (1, expected_patches + 1, cfg.d_vision)


# ---------------------------------------------------------------------------
# 11. test_vision_transformer_gradient_flow
# ---------------------------------------------------------------------------


def test_vision_transformer_gradient_flow():
    """Backward pass through the full VisionTransformer must not error."""
    torch.manual_seed(0)
    vit = VisionTransformer(SMALL_CFG)
    images = _make_images(B=1)
    images.requires_grad_(True)

    out = vit(images)
    loss = out.sum()
    loss.backward()  # should not raise

    assert images.grad is not None, "Gradient should flow back to images"
    assert images.grad.shape == images.shape


# ---------------------------------------------------------------------------
# 12. test_multimodal_embedding_no_image_vs_image
# ---------------------------------------------------------------------------


def test_multimodal_embedding_no_image_vs_image():
    """Output sequence length must differ when image is provided vs. not."""
    torch.manual_seed(0)
    text_embed = _make_text_embed()
    mm = MultimodalEmbedding(text_embed, SMALL_CFG)

    B, T = 1, 7
    input_ids = torch.randint(0, 100, (B, T))
    images = _make_images(B=B)

    out_text_only = mm(input_ids, images=None)
    out_with_image = mm(input_ids, images=images)

    assert out_text_only.shape[1] == T, "Text-only seq len should equal T"
    assert out_with_image.shape[1] == N_VISUAL_SMALL + T, (
        f"With-image seq len should equal {N_VISUAL_SMALL + T}"
    )
    assert out_with_image.shape[1] > out_text_only.shape[1], (
        "Image input must increase sequence length"
    )
