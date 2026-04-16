"""Sampling strategies v2 for autoregressive decoding in AureliusTransformer.

Implements a complete, composable set of token-sampling filters:
temperature, top-k, top-p (nucleus), min-p, locally-typical sampling,
and repetition penalty.  A SamplerPipeline class wires them together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """Configuration for all token-sampling filters."""

    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled
    min_p: float = 0.0      # 0.0 = disabled
    typical_p: float = 1.0  # 1.0 = disabled
    repetition_penalty: float = 1.0  # 1.0 = disabled
    do_sample: bool = True   # False → greedy (argmax)


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by *temperature*.

    Raises:
        ValueError: if temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature!r}"
        )
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-*k* logits; set the rest to -inf.

    k=0 disables filtering (all logits returned unchanged).
    """
    if k == 0:
        return logits
    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    # Threshold = smallest value in the top-k set
    threshold = top_k_values[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling filter.

    Tokens outside the smallest set whose cumulative probability >= *p*
    are set to -inf.  Always keeps at least the highest-probability token.
    p=1.0 returns logits unchanged.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumsum *before* adding this token already >= p.
    # Shift right by 1 so the token that pushes cumsum over p is kept.
    sorted_to_remove = cumulative_probs - sorted_probs >= p
    # Always keep at least the top token
    sorted_to_remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_to_remove, float("-inf"))

    # Scatter back to original ordering
    result = torch.empty_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p filtering.

    Tokens whose probability is below ``min_p * max_prob`` are masked to -inf.
    Always keeps at least one token (the argmax).
    min_p=0.0 returns logits unchanged.
    """
    if min_p == 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob

    to_remove = probs < threshold
    # Always preserve the top token
    argmax_idx = logits.argmax(dim=-1, keepdim=True)
    to_remove.scatter_(-1, argmax_idx, False)

    return logits.masked_fill(to_remove, float("-inf"))


def apply_typical_p(logits: torch.Tensor, mass: float) -> torch.Tensor:
    """Locally typical sampling.

    Computes the conditional entropy H of the distribution and keeps only
    the tokens whose |log p - H| is smallest (most typical), accumulating
    until their total probability mass >= *mass*.
    Always keeps at least one token.  mass=1.0 returns logits unchanged.
    """
    if mass >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-10))

    # Entropy H = -sum(p * log p)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

    # Deviation from entropy — most typical tokens have smallest deviation
    deviation = (log_probs - entropy).abs()

    # Sort ascending by deviation (most typical first)
    sorted_dev, sorted_indices = torch.sort(deviation, dim=-1)
    sorted_probs = probs.gather(-1, sorted_indices)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative mass *before* adding this token >= mass
    to_remove = cumulative - sorted_probs >= mass
    # Always keep the single most-typical token
    to_remove[..., 0] = False

    sorted_logits = logits.gather(-1, sorted_indices)
    sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))

    # Scatter back to original ordering
    result = torch.empty_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result


def apply_repetition_penalty_sampling(
    logits: torch.Tensor,
    past_token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits of previously seen tokens.

    For each token that appears in *past_token_ids*:
      - if logit > 0 → divide by penalty  (reduce positive encouragement)
      - if logit <= 0 → multiply by penalty  (increase negative discouragement)

    penalty=1.0 has no effect.
    """
    if penalty == 1.0:
        return logits

    result = logits.clone()
    unique_ids = past_token_ids.view(-1).unique()
    for token_id in unique_ids:
        idx = int(token_id.item())
        if idx < 0 or idx >= result.size(-1):
            continue
        if result[idx] > 0:
            result[idx] = result[idx] / penalty
        else:
            result[idx] = result[idx] * penalty
    return result


# ---------------------------------------------------------------------------
# Unified sampling entry point
# ---------------------------------------------------------------------------

def sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    past_ids: Optional[torch.Tensor] = None,
) -> int:
    """Apply all enabled filters then sample one token.

    Filter order: temperature → repetition_penalty → top_k → top_p →
                  min_p → typical_p

    If ``config.do_sample`` is False, returns the argmax (greedy decoding).

    Returns:
        int: sampled (or greedy) token id.
    """
    logits = logits.view(-1).float()

    # 1. Temperature
    logits = apply_temperature(logits, config.temperature)

    # 2. Repetition penalty
    if past_ids is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty_sampling(
            logits, past_ids.view(-1), config.repetition_penalty
        )

    # 3. Top-k
    if config.top_k > 0:
        logits = apply_top_k(logits, config.top_k)

    # 4. Top-p
    if config.top_p < 1.0:
        logits = apply_top_p(logits, config.top_p)

    # 5. Min-p
    if config.min_p > 0.0:
        logits = apply_min_p(logits, config.min_p)

    # 6. Typical-p
    if config.typical_p < 1.0:
        logits = apply_typical_p(logits, config.typical_p)

    # Safety: if all logits are -inf, fall back to uniform
    if torch.all(logits == float("-inf")):
        logits = torch.zeros_like(logits)

    if not config.do_sample:
        return int(logits.argmax().item())

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


# ---------------------------------------------------------------------------
# SamplerPipeline
# ---------------------------------------------------------------------------

class SamplerPipeline:
    """A stateless pipeline that applies all enabled sampling filters."""

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config

    def apply_filters(
        self,
        logits: torch.Tensor,
        past_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply all enabled filters and return the filtered logit tensor.

        The input logits shape is preserved (1-D or batched).
        """
        cfg = self.config
        out = logits.float()

        out = apply_temperature(out, cfg.temperature)

        if past_ids is not None and cfg.repetition_penalty != 1.0:
            # Apply row-wise if batched; simple 1-D otherwise
            if out.dim() == 1:
                out = apply_repetition_penalty_sampling(
                    out, past_ids.view(-1), cfg.repetition_penalty
                )
            else:
                # Batched: same past_ids applied to every row
                rows = []
                for row in out:
                    rows.append(
                        apply_repetition_penalty_sampling(
                            row, past_ids.view(-1), cfg.repetition_penalty
                        )
                    )
                out = torch.stack(rows, dim=0)

        if cfg.top_k > 0:
            out = apply_top_k(out, cfg.top_k)

        if cfg.top_p < 1.0:
            out = apply_top_p(out, cfg.top_p)

        if cfg.min_p > 0.0:
            out = apply_min_p(out, cfg.min_p)

        if cfg.typical_p < 1.0:
            out = apply_typical_p(out, cfg.typical_p)

        return out

    def sample(
        self,
        logits: torch.Tensor,
        past_ids: Optional[torch.Tensor] = None,
    ) -> int:
        """Apply filters then sample (or argmax) one token.

        Returns:
            int: sampled token id.
        """
        filtered = self.apply_filters(logits.view(-1), past_ids)

        # Safety fallback
        if torch.all(filtered == float("-inf")):
            filtered = torch.zeros_like(filtered)

        if not self.config.do_sample:
            return int(filtered.argmax().item())

        probs = F.softmax(filtered, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def batch_sample(
        self,
        logits: torch.Tensor,
        past_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one token per row of a batched logit tensor.

        Args:
            logits:   (B, vocab_size) float tensor.
            past_ids: optional 1-D past token ids shared across the batch.

        Returns:
            (B,) int64 tensor of sampled token ids.
        """
        assert logits.dim() == 2, "logits must be (B, vocab_size)"
        B = logits.size(0)
        tokens = []
        for b in range(B):
            tokens.append(self.sample(logits[b], past_ids))
        return torch.tensor(tokens, dtype=torch.long)
