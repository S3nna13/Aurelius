"""Vision encoder for multimodal Aurelius — CLIP-style patch embedding + visual tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """Configuration for the vision encoder."""

    image_size: int = 224       # input image size (H == W assumed)
    patch_size: int = 16        # size of each image patch
    n_channels: int = 3         # RGB input channels
    d_vision: int = 256         # vision encoder hidden dim
    n_vision_layers: int = 2    # number of transformer layers in vision encoder
    n_vision_heads: int = 4     # number of attention heads in vision encoder
    d_lm: int = 64              # language model embedding dim (for projection)
    max_visual_tokens: int = 196  # (224 // 16) ** 2 = 196 patches


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch tokens.

    Uses a strided Conv2d to extract non-overlapping patches, then prepends a
    learnable CLS token and adds learned positional embeddings.

    Args:
        config: VisionConfig instance.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        n_patches = (config.image_size // config.patch_size) ** 2

        # Conv2d with kernel == stride extracts non-overlapping patches.
        self.proj = nn.Conv2d(
            config.n_channels,
            config.d_vision,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Learnable CLS token prepended before patch tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_vision))

        # Positional embeddings: one per patch + one for CLS.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, config.d_vision))

    def forward(self, images: Tensor) -> Tensor:
        """Embed images as patch token sequences.

        Args:
            images: (B, C, H, W) float tensor.

        Returns:
            (B, n_patches + 1, d_vision) — CLS token at position 0, then patches.
        """
        B = images.shape[0]

        # (B, d_vision, H//patch_size, W//patch_size)
        x = self.proj(images)

        # Flatten spatial dimensions -> (B, d_vision, n_patches) -> (B, n_patches, d_vision)
        x = x.flatten(2).transpose(1, 2)

        # Expand CLS token across the batch and prepend.
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_vision)
        x = torch.cat([cls, x], dim=1)           # (B, n_patches + 1, d_vision)

        # Add positional embeddings.
        x = x + self.pos_embed

        return x


# ---------------------------------------------------------------------------
# VisionTransformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """Lightweight Vision Transformer (ViT) encoder.

    Processes image patch tokens through a small stack of standard transformer
    encoder layers, then applies a final layer-norm.

    Args:
        config: VisionConfig instance.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(config)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_vision,
                nhead=config.n_vision_heads,
                dim_feedforward=config.d_vision * 4,
                batch_first=True,
            )
            for _ in range(config.n_vision_layers)
        ])

        self.norm = nn.LayerNorm(config.d_vision)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images into visual token sequences.

        Args:
            images: (B, C, H, W) float tensor.

        Returns:
            (B, n_patches + 1, d_vision) — all tokens, CLS first.
        """
        x = self.patch_embed(images)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# VisualProjection
# ---------------------------------------------------------------------------

class VisualProjection(nn.Module):
    """Project visual tokens from vision encoder space to LM embedding space.

    Args:
        config: VisionConfig instance.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.d_vision, config.d_lm)
        self.norm = nn.LayerNorm(config.d_lm)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        """Project visual tokens into LM token space.

        Args:
            visual_tokens: (B, n_visual_tokens, d_vision).

        Returns:
            (B, n_visual_tokens, d_lm).
        """
        return self.norm(self.proj(visual_tokens))


# ---------------------------------------------------------------------------
# MultimodalEmbedding
# ---------------------------------------------------------------------------

class MultimodalEmbedding(nn.Module):
    """Combine text and visual tokens for multimodal language model input.

    Visual tokens (if present) are prepended before the text token sequence,
    giving the LM full visibility of image context at every text position.

    Args:
        text_embedding: shared nn.Embedding lookup for text tokens.
        vision_config: VisionConfig instance.
    """

    def __init__(self, text_embedding: nn.Embedding, vision_config: VisionConfig) -> None:
        super().__init__()
        self.text_embedding = text_embedding
        self.vision_encoder = VisionTransformer(vision_config)
        self.projection = VisualProjection(vision_config)

    def forward(
        self,
        input_ids: Tensor,
        images: Tensor | None = None,
    ) -> Tensor:
        """Produce combined token embeddings for LM input.

        Args:
            input_ids: (B, T) integer token ids.
            images: (B, C, H, W) float tensor, or None for text-only input.

        Returns:
            (B, T_total, d_lm) where T_total = n_visual_tokens + T if images
            are provided, otherwise T_total = T.
        """
        # Text embeddings: (B, T, d_lm)
        text_embeds = self.text_embedding(input_ids)

        if images is None:
            return text_embeds

        # Visual tokens: (B, n_v, d_vision) -> (B, n_v, d_lm)
        visual_tokens = self.vision_encoder(images)      # (B, n_v, d_vision)
        visual_embeds = self.projection(visual_tokens)   # (B, n_v, d_lm)

        # Prepend visual tokens to text tokens.
        combined = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, n_v + T, d_lm)
        return combined


# ---------------------------------------------------------------------------
# Visual attention mask
# ---------------------------------------------------------------------------

def create_visual_attention_mask(n_visual: int, n_text: int) -> Tensor:
    """Create a combined attention mask for visual + text tokens.

    Attention rules:
      - Visual tokens attend to all other visual tokens (bidirectional).
      - Text tokens attend to all visual tokens + all preceding text tokens
        (causal within the text portion).

    Args:
        n_visual: number of visual tokens prepended before text.
        n_text: number of text tokens.

    Returns:
        Boolean tensor of shape (T_total, T_total) where True means the
        query at row i is allowed to attend to the key at column j.
    """
    T = n_visual + n_text
    mask = torch.zeros(T, T, dtype=torch.bool)

    # Visual tokens: attend to all visual tokens (rows 0..n_visual-1, cols 0..n_visual-1)
    mask[:n_visual, :n_visual] = True

    # Text tokens attending to visual tokens (rows n_visual..T-1, cols 0..n_visual-1)
    mask[n_visual:, :n_visual] = True

    # Text tokens: causal self-attention (lower-triangular within text block)
    for i in range(n_text):
        mask[n_visual + i, n_visual: n_visual + i + 1] = True

    return mask
