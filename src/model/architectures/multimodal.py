"""Multimodal/Vision-Language: CLIP, Flamingo, LLaVA, DALL-E, GPT-4V patterns.

Papers: Radford 2021, Alayrac 2022, Liu 2023, Ramesh 2022, OpenAI GPT-4V 2023.
"""

from __future__ import annotations

import math
import random

from .cnn_vision import VisionTransformer
from .registry import register


class CLIPEncoder:
    """CLIP dual encoder (Radford et al. 2021)."""

    def __init__(self, d_model: int = 512) -> None:
        self.image_encoder = VisionTransformer(224, 16, d_model, 6, 8)
        s = 1.0 / math.sqrt(d_model)
        self.text_proj = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]
        self.image_proj = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]

    def encode_text(self, text_emb: list[float]) -> list[float]:
        return [
            sum(self.text_proj[i][j] * text_emb[j] for j in range(len(text_emb)))
            for i in range(len(self.text_proj))
        ]

    def encode_image(self, image: list[list[list[list[float]]]]) -> list[float]:
        img_tokens = self.image_encoder.forward(image)
        # Flatten token embeddings into a single vector
        flat = [v for tok in img_tokens for v in tok]
        flat = flat[: len(self.image_proj[0])] + [0.0] * max(0, len(self.image_proj[0]) - len(flat))
        return [
            sum(self.image_proj[i][j] * flat[j] for j in range(len(flat)))
            for i in range(len(self.image_proj))
        ]

    def similarity(self, text_emb: list[float], img_emb: list[float]) -> float:
        dot = sum(t * i for t, i in zip(text_emb, img_emb, strict=True))
        nt = math.sqrt(sum(t**2 for t in text_emb))
        ni = math.sqrt(sum(i**2 for i in img_emb))
        return dot / (nt * ni + 1e-8)


register("multimodal.clip", CLIPEncoder)


class LLaVAConnector:
    """LLaVA vision-language connector (Liu et al. 2023). Projects vision features to LLM space."""

    def __init__(self, vision_dim: int = 768, llm_dim: int = 4096) -> None:
        s = 1.0 / math.sqrt(vision_dim)
        self.W = [[random.gauss(0, s) for _ in range(vision_dim)] for _ in range(llm_dim)]

    def forward(self, vision_features: list[list[float]]) -> list[list[float]]:
        return [
            [sum(self.W[i][k] * vf[k] for k in range(len(vf))) for i in range(len(self.W))]
            for vf in vision_features
        ]


register("multimodal.llava_connector", LLaVAConnector)


class MultiModalFusion:
    """GPT-4V style multimodal fusion — projects and interleaves modalities."""

    def __init__(self, d_model: int = 4096) -> None:
        self.text_proj = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.image_proj = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.audio_proj = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(d_model)]

    def fuse(
        self,
        text: list[list[float]],
        images: list[list[list[float]]] | None = None,
        audio: list[list[float]] | None = None,
    ) -> list[list[float]]:
        tokens = [list(t) for t in text]
        if images:
            for img in images:
                img_tokens = [
                    [
                        sum(self.image_proj[i][k] * img[k] for k in range(len(img)))  # type: ignore[operator]
                        for i in range(len(self.image_proj))
                    ]
                ]
                tokens.extend(img_tokens)
        if audio:
            aud_tokens = [
                [
                    sum(self.audio_proj[i][k] * a[k] for k in range(len(a)))
                    for i in range(len(self.audio_proj))
                ]
                for a in audio
            ]
            tokens.extend(aud_tokens)
        return tokens


register("multimodal.fusion", MultiModalFusion)
