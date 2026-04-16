"""Kirchenbauer et al. 2023 green/red list watermarking for LLM detection.

Implements:
  - GreenListWatermark: biases generation by boosting green-list token logits
  - watermark_sample: samples a token with the green-list bias applied
  - generate_watermarked: autoregressive generation with watermarking
  - detect_watermark: z-score based detection on a generated sequence
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class WatermarkConfig:
    """Configuration for green/red list watermark generation and detection."""

    key: int = 42                    # secret key for green/red list generation
    gamma: float = 0.25              # fraction of green-list tokens (0 < gamma < 1)
    delta: float = 2.0               # logit boost for green tokens
    vocab_size: int = 256            # vocabulary size
    seeding_scheme: str = "simple"   # "simple" (hash prev token) or "selfhash"
    detection_threshold: float = 4.0 # z-score threshold for detection


class GreenListWatermark:
    """Green/Red list watermarking (Kirchenbauer et al. 2023).

    For each token position:
    1. Use previous token (or key) to seed a hash
    2. Randomly partition vocab into green (gamma fraction) and red lists
    3. Add delta to green-list token logits before sampling
    """

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config

    def get_green_list(self, prev_token: int) -> Tensor:
        """Get boolean mask of green tokens for this context.

        Returns (vocab_size,) bool tensor.
        Uses hash(key XOR prev_token) to seed a torch.Generator.
        """
        seed = hash(self.config.key ^ prev_token) & 0xFFFFFFFF
        generator = torch.Generator()
        generator.manual_seed(seed)

        n_green = int(self.config.vocab_size * self.config.gamma)
        perm = torch.randperm(self.config.vocab_size, generator=generator)
        green_indices = perm[:n_green]

        mask = torch.zeros(self.config.vocab_size, dtype=torch.bool)
        mask[green_indices] = True
        return mask

    def apply(self, logits: Tensor, prev_token: int) -> Tensor:
        """Apply green-list bias to logits.

        Args:
            logits: (vocab_size,) next-token logit scores
            prev_token: the previous token id used to seed the green list

        Returns:
            Modified logits with delta added to green-list token positions.
        """
        green_mask = self.get_green_list(prev_token)
        logits = logits.clone()
        logits[green_mask] += self.config.delta
        return logits

    def score_token(self, token: int, prev_token: int) -> int:
        """Score 1 if token is on green list for this context, 0 otherwise."""
        green_mask = self.get_green_list(prev_token)
        return int(green_mask[token].item())

    def score_sequence(self, token_ids: Tensor) -> dict[str, float]:
        """Score a full sequence for watermark presence.

        Args:
            token_ids: (T,) sequence of token ids

        Returns:
            dict with keys:
              - 'green_fraction': fraction of tokens that landed on green list
              - 'z_score': z-score under null hypothesis of no watermark
              - 'is_watermarked': bool, True if z_score > detection_threshold
        """
        token_ids = token_ids.long()
        T = len(token_ids)
        if T < 2:
            return {"green_fraction": 0.0, "z_score": 0.0, "is_watermarked": False}

        # Score tokens starting from index 1 (need prev_token for each)
        n_scored = T - 1
        n_green = 0
        for i in range(1, T):
            prev_token = int(token_ids[i - 1].item())
            curr_token = int(token_ids[i].item())
            n_green += self.score_token(curr_token, prev_token)

        gamma = self.config.gamma
        green_fraction = n_green / n_scored

        # z-score under H0: each token independently green with prob gamma
        z_score = (n_green - gamma * n_scored) / math.sqrt(
            n_scored * gamma * (1.0 - gamma)
        )
        is_watermarked = bool(z_score > self.config.detection_threshold)

        return {
            "green_fraction": float(green_fraction),
            "z_score": float(z_score),
            "is_watermarked": is_watermarked,
        }


def watermark_sample(
    logits: Tensor,
    watermark: GreenListWatermark,
    prev_token: int,
    temperature: float = 1.0,
) -> int:
    """Sample a token with watermark bias applied.

    Args:
        logits: (vocab_size,) next-token logits
        watermark: GreenListWatermark instance
        prev_token: previous token id (used to generate green list)
        temperature: sampling temperature (greater than 0)

    Returns:
        Sampled token id as int.
    """
    biased_logits = watermark.apply(logits, prev_token)
    if temperature != 1.0:
        biased_logits = biased_logits / temperature
    probs = torch.softmax(biased_logits, dim=-1)
    token_id = int(torch.multinomial(probs, num_samples=1).item())
    return token_id


def generate_watermarked(
    model: nn.Module,
    input_ids: Tensor,
    watermark: GreenListWatermark,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
) -> Tensor:
    """Generate a token sequence with green/red list watermarking.

    Args:
        model: AureliusTransformer (or compatible) forward returns
               (loss, logits, present_key_values) where logits is (B, T, vocab_size)
        input_ids: (T,) prompt token ids (1-D)
        watermark: GreenListWatermark instance
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature

    Returns:
        (max_new_tokens,) tensor of generated token ids.
    """
    model.eval()
    generated: list[int] = []

    # Running context: start with the prompt
    current_ids = input_ids.long().unsqueeze(0)  # (1, T)

    # Determine the last token of the prompt for initial green-list seeding
    prev_token = int(input_ids[-1].item())

    with torch.no_grad():
        for _ in range(max_new_tokens):
            _, logits, _ = model(current_ids)
            # logits: (1, T, vocab_size) -- take last position
            next_logits = logits[0, -1, :]  # (vocab_size,)

            token_id = watermark_sample(
                next_logits, watermark, prev_token, temperature=temperature
            )
            generated.append(token_id)

            # Append new token and advance
            new_token_tensor = torch.tensor([[token_id]], dtype=torch.long)
            current_ids = torch.cat([current_ids, new_token_tensor], dim=1)
            prev_token = token_id

    return torch.tensor(generated, dtype=torch.long)


def detect_watermark(
    token_ids: Tensor,
    watermark: GreenListWatermark,
) -> dict[str, float]:
    """Run watermark detection on a token sequence.

    Args:
        token_ids: (T,) token sequence to test
        watermark: GreenListWatermark instance (same config used at generation)

    Returns:
        dict with keys: 'green_fraction', 'z_score', 'is_watermarked'
    """
    return watermark.score_sequence(token_ids)
