"""Kirchenbauer et al. 2023 LLM watermarking via green/red token list bias.

Implements a logit processor that biases generation toward a deterministic
"green list" of tokens, plus a detector that checks whether text was generated
with this bias by computing a z-score on the green token fraction.
"""

import hashlib
import math
from dataclasses import dataclass

import torch


@dataclass
class WatermarkConfig:
    """Configuration for watermark generation and detection."""

    gamma: float = 0.25  # green list fraction
    delta: float = 2.0  # logit bias for green list tokens
    vocab_size: int = 256
    seed: int = 42


def compute_green_list(
    prev_token: int,
    vocab_size: int,
    gamma: float,
    seed: int,
) -> set[int]:
    """Hash prev_token with seed to deterministically split vocab into green/red.

    Returns a set of green token indices whose size is approximately
    gamma * vocab_size.
    """
    key = f"{seed}:{prev_token}".encode()
    hash_val = int(hashlib.sha256(key).hexdigest(), 16)

    rng = torch.Generator()
    rng.manual_seed(hash_val % (2**63))
    perm = torch.randperm(vocab_size, generator=rng)

    n_green = int(vocab_size * gamma)
    return set(perm[:n_green].tolist())


def apply_watermark(
    logits: torch.Tensor,
    prev_token: int,
    config: WatermarkConfig,
) -> torch.Tensor:
    """Add delta to green list token logits. Returns modified logits (same shape)."""
    green_set = compute_green_list(prev_token, config.vocab_size, config.gamma, config.seed)
    out = logits.clone()
    green_indices = torch.tensor(sorted(green_set), dtype=torch.long)
    out[..., green_indices] += config.delta
    return out


def detect_watermark(
    token_ids: list[int] | torch.Tensor,
    config: WatermarkConfig,
) -> dict:
    """For each token check if it is in the green list given its predecessor.

    Returns dict with green_fraction, z_score, is_watermarked (z > 2.0).
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    total = len(token_ids) - 1
    if total <= 0:
        return {
            "green_fraction": 0.0,
            "z_score": 0.0,
            "is_watermarked": False,
        }

    green_count = 0
    for i in range(1, len(token_ids)):
        green_set = compute_green_list(
            token_ids[i - 1], config.vocab_size, config.gamma, config.seed
        )
        if token_ids[i] in green_set:
            green_count += 1

    green_fraction = green_count / total
    expected = total * config.gamma
    std = math.sqrt(total * config.gamma * (1 - config.gamma))
    z_score = (green_count - expected) / (std + 1e-8)

    return {
        "green_fraction": green_fraction,
        "z_score": z_score,
        "is_watermarked": z_score > 2.0,
    }


class WatermarkGenerator:
    """Generates text with watermark bias applied at each step."""

    def __init__(self, model: torch.nn.Module, config: WatermarkConfig) -> None:
        self.model = model
        self.config = config

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens with watermark bias. Returns 1-D token tensor."""
        # Ensure input_ids is 2-D (B, S)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            _loss, logits, _pkv = self.model(generated)
            # logits shape: (B, S, V) -- take last position
            next_logits = logits[:, -1, :]  # (B, V)

            # Apply watermark bias for each batch element
            prev_token = generated[0, -1].item()
            next_logits = apply_watermark(next_logits, prev_token, self.config)

            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=-1)

        # Return flattened 1-D tensor for the first batch element
        return generated[0]


class WatermarkDetector:
    """Detects watermarked text using z-score on green token fraction."""

    def __init__(self, config: WatermarkConfig) -> None:
        self.config = config

    def detect(self, token_ids: list[int] | torch.Tensor) -> dict:
        """Returns detection dict with green_fraction, z_score, is_watermarked."""
        return detect_watermark(token_ids, self.config)
