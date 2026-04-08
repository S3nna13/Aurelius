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
    vocab_size: int
    green_list_fraction: float = 0.5  # fraction of vocab in green list
    delta: float = 2.0  # logit boost for green list tokens
    seeding_scheme: str = "hash"  # "hash" only for now
    seed: int = 42  # base seed for reproducibility


def _get_green_list(cfg: WatermarkConfig, prev_token_id: int) -> torch.Tensor:
    """Return boolean tensor of shape (vocab_size,) marking green tokens.

    Deterministic given prev_token_id: hash(seed || prev_token_id) -> shuffle -> take top fraction.
    """
    # Hash prev_token_id with seed to get a deterministic permutation
    key = f"{cfg.seed}:{prev_token_id}".encode()
    hash_val = int(hashlib.sha256(key).hexdigest(), 16)

    # Use hash_val to seed a generator, shuffle vocab indices
    rng = torch.Generator()
    rng.manual_seed(hash_val % (2**63))
    perm = torch.randperm(cfg.vocab_size, generator=rng)

    # First green_list_fraction * vocab_size indices are green
    n_green = int(cfg.vocab_size * cfg.green_list_fraction)
    green_tokens = perm[:n_green]

    # Return boolean mask
    mask = torch.zeros(cfg.vocab_size, dtype=torch.bool)
    mask[green_tokens] = True
    return mask


class WatermarkLogitProcessor:
    """Boosts logits of green-list tokens at each generation step.

    Green list is determined by hashing the previous token ID.
    This is a LogitProcessor-compatible __call__ interface.
    """

    def __init__(self, cfg: WatermarkConfig) -> None:
        self.cfg = cfg

    def _get_green_list(self, prev_token_id: int) -> torch.Tensor:
        """Return boolean tensor of shape (vocab_size,) marking green tokens."""
        return _get_green_list(self.cfg, prev_token_id)

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Add delta to green list tokens based on last token in input_ids.

        If input_ids is empty, use seed as prev_token (no conditioning).
        """
        logits = logits.clone()
        if len(input_ids) == 0:
            prev_token = self.cfg.seed  # use seed as proxy
        else:
            prev_token = input_ids[-1].item()
        green_mask = self._get_green_list(prev_token)
        logits[green_mask] += self.cfg.delta
        return logits


class WatermarkDetector:
    """Detects watermarked text using z-score on green token fraction.

    If text has significantly more green tokens than expected (green_list_fraction),
    it was likely watermarked.
    """

    def __init__(self, cfg: WatermarkConfig) -> None:
        self.cfg = cfg

    def _get_green_list(self, prev_token_id: int) -> torch.Tensor:
        """Return boolean tensor of shape (vocab_size,) marking green tokens."""
        return _get_green_list(self.cfg, prev_token_id)

    def detect(
        self,
        token_ids: list[int],
        z_threshold: float = 4.0,
    ) -> dict:
        """Run detection on a sequence of token IDs.

        For each token t[i] (i >= 1):
        - Compute green list based on t[i-1]
        - Check if t[i] is in the green list

        Returns dict with: green_count, total, z_score, is_watermarked, green_fraction.
        """
        p = self.cfg.green_list_fraction
        total = len(token_ids) - 1
        if total <= 0:
            return {
                "green_count": 0,
                "total": 0,
                "z_score": 0.0,
                "is_watermarked": False,
                "green_fraction": 0.0,
            }

        green_count = 0
        for i in range(1, len(token_ids)):
            green_mask = self._get_green_list(token_ids[i - 1])
            if green_mask[token_ids[i]]:
                green_count += 1

        expected = total * p
        std = math.sqrt(total * p * (1 - p))
        z_score = (green_count - expected) / (std + 1e-8)

        return {
            "green_count": green_count,
            "total": total,
            "z_score": z_score,
            "is_watermarked": z_score >= z_threshold,
            "green_fraction": green_count / total,
        }
