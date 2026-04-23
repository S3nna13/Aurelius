"""Audio-speech fusion encoder for Aurelius multimodal surface.

Inspired by BLIP-2/Q-Former (Salesforce, Apache-2.0, 2301.12597), Gemini 2.5 cross-modal
grounding (Google DeepMind 2025), Whisper cross-attention fusion (OpenAI, MIT),
clean-room reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpeechFusionConfig:
    """Configuration for audio-text speech fusion."""

    audio_d_model: int = 256
    text_d_model: int = 512
    fused_d_model: int = 512
    n_heads: int = 8
    n_fusion_layers: int = 2
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# AudioTextAligner
# ---------------------------------------------------------------------------

class AudioTextAligner(nn.Module):
    """Single-layer audio-text alignment: text tokens attend to audio tokens via cross-attention.

    Projects audio and text to a shared fused space, then applies cross-attention
    where text is the query and audio is the key/value source.

    Args:
        audio_d_model: input dimension of audio features.
        text_d_model: input dimension of text features.
        fused_d_model: shared projection dimension for the fused space.
        n_heads: number of attention heads.
        dropout: dropout probability.
    """

    def __init__(
        self,
        audio_d_model: int,
        text_d_model: int,
        fused_d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_d_model, fused_d_model)
        self.text_proj = nn.Linear(text_d_model, fused_d_model)
        self.cross_attn = nn.MultiheadAttention(fused_d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(fused_d_model)

    def forward(self, audio: Tensor, text: Tensor) -> Tensor:
        """Fuse audio into text via cross-attention.

        Args:
            audio: (B, T_a, audio_d_model) — audio feature sequence.
            text:  (B, T_t, text_d_model) — text feature sequence.

        Returns:
            (B, T_t, fused_d_model) — text features enriched with audio context.
        """
        # Project both modalities to shared fused space
        audio_fused = self.audio_proj(audio)   # (B, T_a, fused_d_model)
        text_fused = self.text_proj(text)       # (B, T_t, fused_d_model)

        # Cross-attention: text queries attend to audio keys/values
        ca_out, _ = self.cross_attn(text_fused, audio_fused, audio_fused)

        # Residual + LayerNorm
        out = self.ln(text_fused + ca_out)      # (B, T_t, fused_d_model)

        return out


# ---------------------------------------------------------------------------
# SpeechFusionEncoder
# ---------------------------------------------------------------------------

class SpeechFusionEncoder(nn.Module):
    """Stack of AudioTextAligner layers for deep audio-speech fusion.

    Progressively fuses audio context into text representations across
    multiple alignment layers, followed by a final linear projection.

    Args:
        audio_d_model: input audio feature dimension.
        text_d_model: input text feature dimension.
        fused_d_model: fused hidden dimension.
        n_heads: number of attention heads per layer.
        n_fusion_layers: number of AudioTextAligner stacked layers.
        dropout: dropout probability.
    """

    def __init__(
        self,
        audio_d_model: int,
        text_d_model: int,
        fused_d_model: int,
        n_heads: int,
        n_fusion_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.audio_d_model = audio_d_model
        self.text_d_model = text_d_model
        self.fused_d_model = fused_d_model

        # First layer bridges from raw audio/text dims to fused dim
        self.layers = nn.ModuleList()
        for i in range(n_fusion_layers):
            # After the first layer, text input is already fused_d_model
            t_dim = text_d_model if i == 0 else fused_d_model
            self.layers.append(
                AudioTextAligner(
                    audio_d_model=audio_d_model,
                    text_d_model=t_dim,
                    fused_d_model=fused_d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            )

        self.output_proj = nn.Linear(fused_d_model, fused_d_model)

    @classmethod
    def from_config(cls, cfg: SpeechFusionConfig) -> "SpeechFusionEncoder":
        """Construct a SpeechFusionEncoder from a SpeechFusionConfig.

        Args:
            cfg: SpeechFusionConfig dataclass instance.

        Returns:
            Initialized SpeechFusionEncoder module.
        """
        return cls(
            audio_d_model=cfg.audio_d_model,
            text_d_model=cfg.text_d_model,
            fused_d_model=cfg.fused_d_model,
            n_heads=cfg.n_heads,
            n_fusion_layers=cfg.n_fusion_layers,
            dropout=cfg.dropout,
        )

    def forward(self, audio: Tensor, text: Tensor) -> Tensor:
        """Fuse audio features into text representations.

        Args:
            audio: (B, T_a, audio_d_model) — audio feature sequence.
            text:  (B, T_t, text_d_model) — text feature sequence.

        Returns:
            (B, T_t, fused_d_model) — audio-fused text representations.
        """
        x = text
        for layer in self.layers:
            x = layer(audio, x)

        out = self.output_proj(x)  # (B, T_t, fused_d_model)
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPEECH_FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "audio_text_aligner": SpeechFusionEncoder,
}

# Register into the shared MODALITY_PROJECTOR_REGISTRY
from src.multimodal.multimodal_registry import register_modality_projector  # noqa: E402

register_modality_projector("SpeechFusionEncoder", SpeechFusionEncoder)


__all__ = [
    "SpeechFusionConfig",
    "AudioTextAligner",
    "SpeechFusionEncoder",
    "SPEECH_FUSION_REGISTRY",
]
